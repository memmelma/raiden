"""Model-agnostic inference loop for the YAM bimanual robot.

Provides:

- ``ModelBridge`` — abstract interface for model-specific logic
- ``RaidenInferenceLoop`` — camera/motor infrastructure + control loop

The split: **raiden** owns hardware (cameras, motors, 14D joint commands).
The **model repo** owns everything model-specific (action space, preprocessing,
inference, action chunking).

Usage from a model repo::

    from raiden.inference import ModelBridge, RaidenInferenceLoop

    class MyBridge(ModelBridge):
        def load(self, ckpt_path, **kwargs): ...
        def predict(self, obs):              ...  # returns (14,)

    if __name__ == "__main__":
        bridge = MyBridge()
        loop = RaidenInferenceLoop(bridge, ckpt_path="...", action_hz=30)
        loop.run()

Or via CLI::

    rd infer --bridge my_repo.deployment.bridge:MyBridge \\
             --ckpt_path /path/to/checkpoint.pt

Action layout
-------------
``predict()`` must return a ``(14,)`` float32 array using raiden's bimanual
convention::

    [0:6]   right arm joint angles (rad)
    [6]     right gripper (0=open, 1=closed)
    [7:13]  left arm joint angles (rad)
    [13]    left gripper (0=open, 1=closed)

Thread layout (inherited from RaidenPolicyServer)::

    camera-<name>       : grabs frames at ~30 Hz (per camera)
    proprio             : reads joint state at ~100 Hz
    main thread         : inference loop (gated by bridge.predict() speed)
"""

import importlib
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import tqdm

import numpy as np
from chiral.types import Observation

from raiden.robot.rollout_logger import RolloutLogger
from raiden.server import RaidenPolicyServer

DOF = 7  # joints per arm (6 revolute + 1 gripper)

# Attribute names various bridges use for their fixed action-chunk length.
# Checked in order; the first one present on the bridge instance wins.
_CHUNK_SIZE_ATTRS = ("execute_len", "action_horizon", "_action_horizon", "chunk_size", "_chunk_size")


def _detect_chunk_size(bridge: "ModelBridge") -> Optional[int]:
    """Best-effort lookup of a bridge's fixed action-chunk length.

    Different bridges name this differently (``OpenPiBridge.execute_len``,
    a custom bridge's ``action_horizon``/``_action_horizon``, etc.) — try the
    common spellings before falling back to latency-based detection in
    :meth:`RaidenInferenceLoop.run`.
    """
    for attr in _CHUNK_SIZE_ATTRS:
        val = getattr(bridge, attr, None)
        if val:
            return int(val)
    return None


def _fmt_arm(arm_7d: np.ndarray) -> str:
    """Format one arm's command as ``j0..j5`` plus gripper, on a single line.

    Every commanded number is shown -- no summarization or small-value
    suppression -- so the printed stream is a complete record of what the
    motors were told to do.
    """
    joints = " ".join(f"{v:7.3f}" for v in arm_7d[:6])
    return f"=[{joints}] g={arm_7d[6]:5.3f}"


# ---------------------------------------------------------------------------
# Abstract bridge interface
# ---------------------------------------------------------------------------


class ModelBridge(ABC):
    """Abstract interface between a policy model and the YAM robot.

    Implement this in your model's repo to handle all model-specific logic:

    - Model loading (checkpoint format, framework imports)
    - Observation preprocessing (image resize, pointmaps, state assembly,
      normalization — whatever your model's action space requires)
    - Inference (forward pass)
    - Action postprocessing (denormalization, chunking, format conversion)

    The ``predict()`` method receives a raw ``Observation`` from the robot
    and must return a ``(14,)`` float32 array of motor commands::

        [0:6]   right arm joint angles (rad)
        [6]     right gripper (0=open, 1=closed)
        [7:13]  left arm joint angles (rad)
        [13]    left gripper (0=open, 1=closed)

    Action chunking, if used, should be managed internally — ``predict()``
    is called once per control step and should return exactly one action.
    """

    @abstractmethod
    def load(self, ckpt_path: str, **kwargs) -> None:
        """Load model from checkpoint.

        Args:
            ckpt_path: Path to model checkpoint file.
            **kwargs: Model-specific arguments (e.g. ``host``, ``port``,
                ``action_horizon``, ``prompt``).
        """

    @abstractmethod
    def predict(self, obs: Observation) -> np.ndarray:
        """Given a raw observation, return motor commands.

        Args:
            obs: Raw observation containing:

                - ``obs.cameras``: list of ``CameraInfo``
                  (each has ``.name``, ``.image``, ``.depth``,
                  ``.intrinsics``, ``.extrinsics``)
                - ``obs.proprios``: dict of ``(N,)`` float32 arrays
                  (``follower_r_joint_pos``, ``follower_l_joint_pos``, etc.)
                - ``obs.timestamp``: float

        Returns:
            ``(14,)`` float32 motor commands:
            ``[right_joints(6), right_grip(1), left_joints(6), left_grip(1)]``
        """

    def reset(self) -> None:
        """Called when the robot is homed. Override to reset internal state
        (e.g. clear action chunk buffer)."""


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


class RaidenInferenceLoop(RaidenPolicyServer):
    """Model-agnostic inference loop for the YAM robot.

    Reuses ``RaidenPolicyServer``'s camera capture, depth computation,
    proprioception reading, and motor control.  Delegates all model-specific
    logic to a :class:`ModelBridge` instance.

    Args:
        bridge: A ``ModelBridge`` implementation from the model repo.
        ckpt_path: Passed to ``bridge.load()``.
        action_hz: Control loop frequency in Hz (default 30, should match
            training data frame rate).
        bridge_kwargs: Extra keyword arguments forwarded to ``bridge.load()``.
        log_dir: Root directory for rollout debug logs (see ``RolloutLogger``).
        log_images: Also save a camera snapshot at every detected action-chunk
            boundary. Off by default — writing images to disk on the control
            thread has previously caused control-loop slowdowns / jerky
            motion, so only enable this when actively debugging.
        **kwargs: Forwarded to ``RaidenPolicyServer`` (camera_config_file,
            calibration_file, stereo_method, etc.).
    """

    def __init__(
        self,
        bridge: ModelBridge,
        ckpt_path: str,
        action_hz: float = 30.0,
        bridge_kwargs: Optional[dict] = None,
        horizon: Optional[int] = None,
        save_video: Optional[str] = None,
        log_dir: Optional[str] = None,
        log_images: bool = False,
        log_actions: Optional[str] = None,
        camera_fps_interval: float = 2.0,
        **kwargs,
    ):
        self._bridge = bridge
        self._action_hz = action_hz
        self._horizon = horizon
        self._save_video = Path(save_video) if save_video else None
        self._log_images = log_images
        logger_kwargs = {"log_root": Path(log_dir)} if log_dir else {}
        self._rollout_logger = RolloutLogger(
            source="infer",
            extra_meta={"log_images_enabled": log_images},
            **logger_kwargs,
        )
        self._log_actions = log_actions or None
        self._action_log_fh = None
        self._action_log_path: Optional[Path] = None
        self._video_t0: float = 0.0
        # Per-camera fps tracking (see _open_camera_fps_log).
        self._cam_names: list = []
        self._cam_last_counts: dict = {}
        self._cam_last_t: float = 0.0
        self._cam_totals_t0: float = 0.0
        self._cam_totals_start: dict = {}
        self._cam_log_fh = None
        self._cam_log_path: Optional[Path] = None
        self._cam_sample_s: float = float(camera_fps_interval)

        # Load model FIRST — this can take seconds (downloading weights,
        # building the network) and doesn't need hardware.  Loading after
        # robot init causes CAN timeouts on idle motor chains.
        print(f"\nLoading model from {ckpt_path}...")
        self._bridge.load(ckpt_path, **(bridge_kwargs or {}))
        print("Model loaded.\n")

        # Now initialize cameras, proprio threads, and robots (inherited).
        super().__init__(record_raw_images=(self._save_video is not None), **kwargs)

    def _safety_check_rl(self, action_14d: np.ndarray) -> np.ndarray:
        """Clip any joint delta that exceeds ``_max_joint_delta``.

        Uses raiden's bimanual inference layout: ``[r(7), l(7)]``.  This is
        independent of ``RaidenPolicyServer._check_joint_delta`` (which uses
        the ``[l, r]`` layout served over the WebSocket protocol).

        Clips ``action_14d`` in place (the per-arm slices are views) and returns
        the pre-clip copy so callers can log what the policy actually asked for
        alongside what was commanded.
        """
        policy_action = action_14d.copy()
        pairs = []
        if self._robot.follower_r:
            q_r = self._read_proprio("follower_r_joint_pos")
            if q_r is not None:
                pairs.append(("right", q_r, action_14d[:DOF]))
        if self._robot.follower_l:
            q_l = self._read_proprio("follower_l_joint_pos")
            if q_l is not None:
                pairs.append(("left", q_l, action_14d[DOF : DOF * 2]))

        for arm, current, commanded in pairs:
            # Only check the 6 arm joints — gripper uses a wider linear range.
            delta = np.abs(commanded[:6] - current[:6])
            max_delta = float(delta.max())
            if max_delta > self._max_joint_delta:
                joint_idx = int(delta.argmax())
                print(
                    f"[CLIP] {arm} arm joint {joint_idx}: "
                    f"{max_delta:.4f} > {self._max_joint_delta:.4f} rad. Clipping."
                )
                commanded[:6] = np.clip(
                    commanded[:6],
                    current[:6] - self._max_joint_delta,
                    current[:6] + self._max_joint_delta,
                )

        return policy_action

    def _open_action_log(self) -> None:
        """Open the action CSV and write its header.

        One row per control step, holding every value that was commanded plus
        the pre-clip policy request, so a rollout can be reconstructed exactly.
        Buffered and held open for the run -- reopening per step would add file
        I/O to a 50 Hz control loop.
        """
        if self._log_actions is None:
            return
        path = Path(self._log_actions)
        if path.is_dir():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path / f"actions_{stamp}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._action_log_path = path
        self._action_log_fh = open(path, "w", buffering=1 << 16)
        cols = ["step", "t_wall", "hz", "clipped"]
        for side in ("r", "l"):
            cols += [f"cmd_{side}_j{i}" for i in range(6)] + [f"cmd_{side}_grip"]
        for side in ("r", "l"):
            cols += [f"policy_{side}_j{i}" for i in range(6)] + [f"policy_{side}_grip"]
        # Measured gripper position, read back from the follower at the same
        # step. A grasp failure looks different depending on which of these
        # moves: policy_*_grip flat = the policy never asked to close;
        # obs_*_grip failing to track cmd_*_grip = it asked and the hardware
        # did not get there (blocked, out of range, or too slow).
        cols += ["obs_r_grip", "obs_l_grip", "err_r_grip", "err_l_grip"]
        self._action_log_fh.write(",".join(cols) + "\n")
        print(f"[actions] logging every action to {path}")

    def _log_action(
        self,
        step: int,
        hz: float,
        commanded: np.ndarray,
        policy_action: np.ndarray,
    ) -> None:
        if self._action_log_fh is None:
            return
        clipped = int(not np.allclose(commanded, policy_action))
        row = [str(step), f"{time.time():.6f}", f"{hz:.3f}", str(clipped)]
        row += [f"{v:.6f}" for v in commanded]
        row += [f"{v:.6f}" for v in policy_action]
        obs_r, obs_l = self._read_gripper_state()
        row += [
            "" if obs_r is None else f"{obs_r:.6f}",
            "" if obs_l is None else f"{obs_l:.6f}",
            "" if obs_r is None else f"{commanded[6] - obs_r:.6f}",
            "" if obs_l is None else f"{commanded[DOF * 2 - 1] - obs_l:.6f}",
        ]
        self._action_log_fh.write(",".join(row) + "\n")

    def _read_gripper_state(self) -> tuple:
        """Measured (right, left) gripper positions, or ``None`` per arm.

        Index 6 of each follower's joint vector is the gripper.
        """
        out = []
        for name in ("follower_r_joint_pos", "follower_l_joint_pos"):
            q = self._read_proprio(name)
            out.append(None if q is None or len(q) <= 6 else float(q[6]))
        return tuple(out)

    def _close_action_log(self) -> None:
        if self._action_log_fh is not None:
            self._action_log_fh.close()
            self._action_log_fh = None
            print(f"[actions] wrote action log -> {self._action_log_path}")

    def _open_camera_fps_log(self) -> None:
        """Start per-camera fps tracking, and open its CSV if action logging is on.

        The configured fps is only a request. A camera starved by USB bandwidth
        or per-frame work (depth alignment, frame buffering) delivers fewer, and
        the policy silently conditions on stale views. This measures what each
        camera actually produced, throughout the run.
        """
        self._cam_names = sorted(self.read_camera_frame_counts().keys())
        self._cam_last_counts = self.read_camera_frame_counts()
        self._cam_last_t = time.perf_counter()
        self._cam_totals_t0 = self._cam_last_t
        self._cam_totals_start = dict(self._cam_last_counts)

        if self._action_log_path is None:
            return
        path = self._action_log_path.with_name(
            self._action_log_path.stem.replace("actions_", "camera_fps_") + ".csv"
        )
        self._cam_log_fh = open(path, "w", buffering=1 << 14)
        self._cam_log_path = path
        self._cam_log_fh.write(
            ",".join(["t_wall", "elapsed_s", *(f"{c}_fps" for c in self._cam_names)]) + "\n"
        )
        print(f"[cams] logging per-camera fps to {path}")

    def _sample_camera_fps(self) -> None:
        """Compute per-camera fps since the last sample; print and log a row."""
        now = time.perf_counter()
        dt = now - self._cam_last_t
        if dt < self._cam_sample_s:
            return
        counts = self.read_camera_frame_counts()
        fps = {
            name: (counts[name] - self._cam_last_counts.get(name, 0)) / dt
            for name in self._cam_names
        }
        self._cam_last_counts = counts
        self._cam_last_t = now

        parts = "  ".join(f"{n}={fps[n]:5.1f}" for n in self._cam_names)
        line = f"[cams] {parts}"
        if self._horizon is not None:
            tqdm.tqdm.write(line)
        else:
            print(line)

        if self._cam_log_fh is not None:
            row = [f"{time.time():.6f}", f"{now - self._cam_totals_t0:.3f}"]
            row += [f"{fps[n]:.3f}" for n in self._cam_names]
            self._cam_log_fh.write(",".join(row) + "\n")

    def _close_camera_fps_log(self) -> None:
        """Print each camera's mean fps for the whole run, then close the CSV."""
        if not getattr(self, "_cam_names", None):
            return
        elapsed = max(time.perf_counter() - self._cam_totals_t0, 1e-6)
        counts = self.read_camera_frame_counts()
        print("\n[cams] mean capture rate over the rollout:")
        for name in self._cam_names:
            n = counts[name] - self._cam_totals_start.get(name, 0)
            mean = n / elapsed
            flag = "" if mean >= 20.0 else "   <-- STARVED"
            print(f"[cams]   {name:22s} {mean:5.1f} fps  ({n} frames){flag}")
        if self._cam_log_fh is not None:
            self._cam_log_fh.close()
            self._cam_log_fh = None
            print(f"[cams] wrote camera fps log -> {self._cam_log_path}")

    def _write_videos(self) -> None:
        """Write per-camera frame buffers to MP4 files under a timestamped directory.

        Runs from the ``finally`` of :meth:`run`, at the end of a rollout whose
        frames exist only in memory: a raised exception here destroys the whole
        recording. So every failure is caught and reported per camera, and one
        bad camera never costs the others.
        """
        try:
            import imageio.v3 as iio
        except ImportError:
            print("[video] imageio not installed — skipping video save.")
            return

        with self._video_buffers_lock:
            frame_buffers = {k: list(v) for k, v in self._video_frame_buffers.items()}

        frame_buffers = {k: v for k, v in frame_buffers.items() if v}
        if not frame_buffers:
            print(
                "[video] no frames were buffered — nothing to save. "
                "(Cameras may have failed to start.)"
            )
            return

        # Frames are appended by the camera threads at each camera's own capture
        # rate, which is NOT the control rate -- writing at action_hz would play
        # the rollout back at the wrong speed. Derive fps from what was actually
        # captured over the recording window.
        elapsed = max(time.time() - self._video_t0, 1e-6)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self._save_video / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)

        for cam_name, frames in frame_buffers.items():
            out_path = out_dir / f"{cam_name}.mp4"
            fps = max(len(frames) / elapsed, 1.0)
            try:
                # Frames were stored as BGR (OpenCV convention); flip to RGB.
                rgb_frames = [f[..., ::-1] for f in frames]
                iio.imwrite(
                    str(out_path),
                    rgb_frames,
                    plugin="pyav",
                    codec="h264",
                    fps=fps,
                )
                print(
                    f"[video] saved {len(frames)} frames @ {fps:.1f} fps → {out_path}"
                )
            except ImportError as e:
                print(
                    f"[video] cannot encode {cam_name}: {e}\n"
                    f"[video] install the encoder with:  uv pip install av"
                )
            except Exception as e:
                print(f"[video] failed to write {out_path}: {type(e).__name__}: {e}")

    def run(self) -> None:
        """Run the closed-loop inference loop.

        Runs for ``self._horizon`` steps when set, otherwise blocks until Ctrl+C.
        A tqdm progress bar is shown when a finite horizon is given.
        """
        dt = 1.0 / self._action_hz
        step = 0

        # Home the robot before inference so the first commanded action does
        # not cause a large jump (RaidenPolicyServer.__init__ does not home).
        print("Homing robot...")
        self._robot.move_to_home_positions(simultaneous=True)

        self._bridge.reset()

        # Detect the bridge's action-chunk size (e.g. OpenPiBridge.execute_len,
        # or a custom bridge's action_horizon) so logged rollouts can mark
        # chunk boundaries even for bridges that don't expose a
        # ``just_refilled`` flag.
        chunk_size = _detect_chunk_size(self._bridge)
        self._rollout_logger.chunk_size = chunk_size
        # Rolling baseline of predict() latency, used as a bridge-agnostic
        # fallback chunk-boundary detector (see below) for bridges that don't
        # expose ``just_refilled`` or ``execute_len`` at all (e.g. custom
        # bridges from other repos).
        predict_latencies: "deque[float]" = deque(maxlen=50)

        self._open_action_log()
        self._open_camera_fps_log()

        print(f"Starting inference at {self._action_hz} Hz")
        if self._horizon is not None:
            print(f"Horizon: {self._horizon} steps (~{self._horizon / self._action_hz:.1f}s)")
        else:
            print("Press Ctrl+C to stop.\n")

        if self._save_video is not None:
            self._video_t0 = time.time()
            self._video_recording = True

        try:
            iterator: object
            if self._horizon is not None:
                iterator = tqdm.trange(self._horizon, desc="rollout", unit="step")
            else:
                iterator = iter(int, 1)  # infinite iterator

            for _ in iterator:
                t_step = time.perf_counter()

                # Build observation from live sensors.
                obs = self._make_obs()

                # Ask bridge for next motor command (raiden layout: [r, l]).
                t_predict = time.perf_counter()
                action = self._bridge.predict(obs)
                predict_elapsed = time.perf_counter() - t_predict

                # Convert EE pose to joint command if needed.  Note: the
                # parent's ``_ee_pose_to_joint_cmd`` returns ``[l, r]``; for
                # a model that emits raiden-layout EE poses, callers should
                # set ``action_type="joint"`` and let the bridge produce the
                # ``[r, l]`` joint command directly.
                if self._action_type == "ee_pose":
                    init_cmd = self._last_joint_cmd
                    action_14d = self._ee_pose_to_joint_cmd(action, init_cmd)
                else:
                    action_14d = action

                # Safety check — clip any joint that jumps too far ([r, l]).
                # Returns the pre-clip request so the log can distinguish what
                # the policy asked for from what the motors were given.
                policy_action = self._safety_check_rl(action_14d)
                # Same value under the name the RolloutLogger call below uses.
                action_pred_14d = policy_action

                self._last_joint_cmd = action_14d

                # Log policy observation, actual robot state, and predicted vs.
                # commanded action for offline debugging (see rd infer's
                # RolloutLogger / scripts/visualize_rollout.py).
                q_r = obs.proprios.get("follower_r_joint_pos")
                q_l = obs.proprios.get("follower_l_joint_pos")
                if q_r is not None and q_l is not None:
                    state_14d = np.concatenate([q_r, q_l])
                    # Prefer the bridge's own signal for "just fetched a fresh
                    # chunk" (e.g. OpenPiEEBridge.just_refilled) since it's
                    # exact; fall back to step-modulo when the bridge exposes
                    # a fixed chunk length (e.g. OpenPiBridge.execute_len).
                    # For bridges that expose neither (custom/third-party
                    # bridges), detect refills generically: querying the
                    # policy for a fresh chunk is far slower than popping an
                    # already-buffered action off a local queue, so a latency
                    # spike relative to the rolling baseline is a reliable
                    # signal regardless of bridge internals.
                    if hasattr(self._bridge, "just_refilled"):
                        chunk_start = bool(self._bridge.just_refilled)
                    elif chunk_size:
                        chunk_start = step % chunk_size == 0
                    elif predict_latencies:
                        baseline = float(np.median(predict_latencies))
                        chunk_start = predict_elapsed > max(3.0 * baseline, 0.02)
                    else:
                        chunk_start = True  # first step always starts a chunk
                    predict_latencies.append(predict_elapsed)
                    self._rollout_logger.log_step(
                        obs=obs,
                        state_14d=state_14d,
                        action_pred_14d=action_pred_14d,
                        action_cmd_14d=action_14d,
                        chunk_start=chunk_start,
                    )
                    if chunk_start and self._log_images:
                        self._rollout_logger.log_chunk_images(step, obs.cameras)

                # Command motors (raiden inference layout: right first, then left).
                if self._robot.follower_r:
                    self._robot.follower_r.command_joint_pos(action_14d[:DOF])
                if self._robot.follower_l:
                    self._robot.follower_l.command_joint_pos(action_14d[DOF : DOF * 2])

                step += 1

                elapsed = time.perf_counter() - t_step
                hz = 1.0 / max(elapsed, 1e-6)

                self._log_action(step, hz, action_14d, policy_action)
                self._sample_camera_fps()

                # Print every commanded value: 6 arm joints + gripper per arm.
                # ``_fmt_arm`` keeps a step on one line -- at 50 Hz, multi-line
                # output makes stdout itself a source of control-loop jitter.
                clip_flag = "" if np.allclose(action_14d, policy_action) else "  [CLIPPED]"
                # Commanded gripper alone can't explain a failed grasp -- show
                # what the fingers actually did next to what they were told.
                obs_r_g, obs_l_g = self._read_gripper_state()
                grip_obs = (
                    f"  grip_obs R={'  n/a' if obs_r_g is None else f'{obs_r_g:5.3f}'}"
                    f" L={'  n/a' if obs_l_g is None else f'{obs_l_g:5.3f}'}"
                )
                line = (
                    f"step={step:5d} hz={hz:5.1f}"
                    f"  R{_fmt_arm(action_14d[:DOF])}"
                    f"  L{_fmt_arm(action_14d[DOF : DOF * 2])}"
                    f"{grip_obs}{clip_flag}"
                )
                if self._horizon is not None:
                    tqdm.tqdm.write(line)
                else:
                    print(line)

                # Sleep to maintain target Hz.
                elapsed = time.perf_counter() - t_step
                remaining = dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        except KeyboardInterrupt:
            print("\n\nStopping inference...")
        finally:
            unhealthy = self.get_unhealthy_cameras()
            if unhealthy:
                print(
                    f"[camera-health] WARNING: camera(s) were unhealthy at some "
                    f"point during this rollout: {unhealthy}. Treat this run's "
                    "policy behavior as suspect — see rollout meta.json "
                    "['unhealthy_cameras']."
                )
            self._rollout_logger.extra_meta["unhealthy_cameras"] = unhealthy
            self._rollout_logger.save()
            if self._save_video is not None:
                self._video_recording = False
            self._close_action_log()
            self._close_camera_fps_log()
            self._bridge.reset()
            if step > 0:
                print("Returning to home...")
                self._robot.move_to_home_positions(simultaneous=True)
            self._running = False
            time.sleep(0.1)
            self.close()
            if self._save_video is not None:
                self._write_videos()
            print("Done.")


# ---------------------------------------------------------------------------
# Dynamic bridge loading (used by ``rd infer``)
# ---------------------------------------------------------------------------


def load_bridge(bridge_spec: str, **kwargs) -> ModelBridge:
    """Import a ModelBridge class from a ``module.path:ClassName`` string.

    Extra *kwargs* are forwarded to the bridge constructor.

    Examples::

        load_bridge("deployment.openpi_bridge:OpenPiBridge")
        load_bridge("deployment.openpi_bridge:OpenPiBridge", action_horizon=10)
    """
    if ":" not in bridge_spec:
        raise ValueError(
            f"Bridge spec must be 'module.path:ClassName', got '{bridge_spec}'"
        )
    module_path, class_name = bridge_spec.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Cannot import bridge module '{module_path}'. "
            f"Make sure it is installed or on PYTHONPATH."
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{class_name}'"
        )
    bridge = cls(**kwargs)
    if not isinstance(bridge, ModelBridge):
        raise TypeError(f"'{bridge_spec}' is not a ModelBridge subclass")
    return bridge
