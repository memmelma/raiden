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

        # Load model FIRST — this can take seconds (downloading weights,
        # building the network) and doesn't need hardware.  Loading after
        # robot init causes CAN timeouts on idle motor chains.
        print(f"\nLoading model from {ckpt_path}...")
        self._bridge.load(ckpt_path, **(bridge_kwargs or {}))
        print("Model loaded.\n")

        # Now initialize cameras, proprio threads, and robots (inherited).
        super().__init__(record_raw_images=(self._save_video is not None), **kwargs)

    def _safety_check_rl(self, action_14d: np.ndarray) -> None:
        """Trip the e-stop if any joint delta exceeds ``_max_joint_delta``.

        Uses raiden's bimanual inference layout: ``[r(7), l(7)]``.  This is
        independent of ``RaidenPolicyServer._check_joint_delta`` (which uses
        the ``[l, r]`` layout served over the WebSocket protocol).
        """
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

    def _write_videos(self) -> None:
        """Write per-camera frame buffers to MP4 files under a timestamped directory."""
        try:
            import imageio.v3 as iio
        except ImportError:
            print("[video] imageio not installed — skipping video save.")
            return

        with self._video_buffers_lock:
            frame_buffers = {k: list(v) for k, v in self._video_frame_buffers.items()}

        if not frame_buffers:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self._save_video / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)

        for cam_name, frames in frame_buffers.items():
            if not frames:
                continue
            out_path = out_dir / f"{cam_name}.mp4"
            # Frames were stored as BGR (OpenCV convention); flip to RGB for imageio.
            rgb_frames = [f[..., ::-1] for f in frames]
            iio.imwrite(
                str(out_path),
                rgb_frames,
                plugin="pyav",
                codec="h264",
                fps=self._action_hz,
            )
            print(f"[video] saved {len(frames)} frames → {out_path}")

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

        print(f"Starting inference at {self._action_hz} Hz")
        if self._horizon is not None:
            print(f"Horizon: {self._horizon} steps (~{self._horizon / self._action_hz:.1f}s)")
        else:
            print("Press Ctrl+C to stop.\n")

        if self._save_video is not None:
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

                # Snapshot the predicted action before safety clipping mutates it.
                action_pred_14d = action_14d.copy()

                # Safety check — abort if any joint jumps too far ([r, l]).
                self._safety_check_rl(action_14d)

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
                r_act = np.array2string(action_14d[:DOF], precision=3, suppress_small=True)
                l_act = np.array2string(
                    action_14d[DOF : DOF * 2], precision=3, suppress_small=True
                )
                if self._horizon is not None:
                    tqdm.tqdm.write(f"step={step:5d}  hz={hz:.1f}  r={r_act}  l={l_act}")
                else:
                    print(f"step={step:5d}  hz={hz:.1f}  r={r_act}  l={l_act}")

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
