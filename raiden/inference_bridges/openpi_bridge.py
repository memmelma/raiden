"""OpenPi WebSocket eval bridge for the YAM bimanual robot.

Connects to an openpi-style policy server (e.g. ``pi05_ethernet``) running the
WebSocket + msgpack-numpy protocol and drives the YAM follower arms with the
returned action chunks. Use this for closed-loop policy evaluation.

Wire format (per ``src/openpi/policies/yam_policy.py`` on the server):

    Request (each step):
        observation/state               (14,) float32   [L_joint(6), L_grip(1),
                                                          R_joint(6), R_grip(1)]
        observation/image               (H,W,3) uint8 RGB   (base/scene view)
        observation/left_image          (H,W,3) uint8 RGB   (left wrist)
        observation/right_image         (H,W,3) uint8 RGB   (right wrist)
        prompt                          str

    Response:
        actions       (action_horizon, 14) float32 — same layout as state
        server_timing dict

Raiden's bimanual inference layout is right-first ``[r(7), l(7)]``; this bridge
flips the policy's left-first action into raiden's right-first order before
returning it to :class:`raiden.inference.RaidenInferenceLoop`.

Run::

    rd infer \\
        --bridge raiden.inference_bridges.openpi_bridge:OpenPiBridge \\
        --ckpt_path unused \\
        --resize_images 224x224 \\
        --bridge-kwargs \\
            host=10.158.134.124 port=8000 \\
            prompt='put the box on the shelf and unplug the ethernet cable'
"""

from __future__ import annotations

import functools
import logging
import time
from collections import deque
from typing import Optional

import cv2
import msgpack
import numpy as np

from raiden.inference import ModelBridge
from raiden.intervention import (
    InterventionConfig,
    PositionTrackingInterventionController,
)
from raiden.robot.leader_io import BOTTOM_BIT, apply_leader_grippers

DOF = 7
IMG_HW = 224


def _truthy(v) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class _WebsocketClient:
    """Persistent sync websocket client matching openpi's wire protocol."""

    def __init__(self, host: str, port: int):
        from websockets.sync.client import connect

        uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        self._uri = uri
        self._conn = connect(uri, compression=None, max_size=None)
        # Server sends metadata as the first frame on connect.
        self.metadata = _unpackb(self._conn.recv())

    def infer(self, obs: dict) -> dict:
        self._conn.send(_packb(obs))
        resp = self._conn.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"server error:\n{resp}")
        return _unpackb(resp)

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


class OpenPiBridge(ModelBridge):
    """Closed-loop eval bridge for the YAM openpi policy server."""

    def __init__(self, **kwargs):
        self.host: str = kwargs.get("host", "10.158.134.124")
        self.port: int = int(kwargs.get("port", 8000))
        self.prompt: str = kwargs.get("prompt", "")
        # How many actions from each chunk to execute open-loop before
        # re-querying. pi05 returns ~10-50 actions per chunk; popping 8
        # at 30 Hz means re-querying every ~265 ms.
        self.execute_len: int = int(kwargs.get("execute_len", 8))

        # Intervention support — when enabled (requires --use-leaders so the
        # leader joint streams show up in the chiral Observation), the bridge
        # watches leader-vs-follower joint divergence and, while the operator
        # is moving the leader, commands the followers to mirror the leader
        # arm pose (gripper held from the last policy command).
        self.enable_intervention: bool = _truthy(kwargs.get("enable_intervention", "0"))
        # White-button toggle: press the white button on either leader to flip
        # in/out of intervention. Composed with position-tracking — either
        # signal can engage the followers-mirror-leader override.
        self.enable_button_intervention: bool = _truthy(
            kwargs.get("enable_button_intervention", "0")
        )
        # If True, the bridge starts in INTERVENTION mode (leader drives
        # followers). The first white-button press hands off to the policy;
        # subsequent presses toggle back and forth. Use this when the
        # operator wants to position the robot or hold control until they
        # explicitly release it.
        self.start_intervening: bool = _truthy(kwargs.get("start_intervening", "0"))
        self._button_force: bool = self.start_intervening
        # Which leader io_inputs bit the toggle button maps to. Defaults to the
        # BOTTOM (white) button per raiden.robot.leader_io (single source of
        # truth); override with button_bit=... only for non-standard handles.
        self.button_bit: int = int(kwargs.get("button_bit", BOTTOM_BIT))
        self._button_prev: dict[str, float] = {}
        # While intervening, drop leader PD gains to zero so the operator can
        # backdrive them (gravity-comp), then restore on release. Without this
        # the loop holds the leaders stiff at the policy pose and the operator
        # fights the PD controller. Disable for position-tracking-only mode,
        # where leaders must stay free the whole time.
        self.free_leaders_on_intervention: bool = _truthy(
            kwargs.get("free_leaders_on_intervention", "1")
        )
        self._robot = None  # set by attach_loop(); needed for button polling
        self.verbose_intervention: bool = _truthy(kwargs.get("verbose_intervention", "0"))
        self._verbose_every: int = int(kwargs.get("verbose_every", 10))
        self._verbose_tick: int = 0
        intervention_cfg = InterventionConfig(
            thresh_engage=float(kwargs.get(
                "intervention_thresh_engage", InterventionConfig.thresh_engage
            )),
            thresh_release=float(kwargs.get(
                "intervention_thresh_release", InterventionConfig.thresh_release
            )),
            debounce_ticks=int(kwargs.get(
                "intervention_debounce_ticks", InterventionConfig.debounce_ticks
            )),
        )
        self._intervention = PositionTrackingInterventionController(cfg=intervention_cfg)
        self._was_intervening: bool = False

        # Intervention-frequency telemetry (per episode). predict() counts every
        # tick (``_interv_total_steps``) vs. ticks under intervention
        # (``_interv_step_count``) and the number of distinct engage events
        # (``_interv_engage_count``); ``_interv_cur_ticks`` is the length of the
        # currently-active intervention. reset() logs a one-line summary and
        # zeroes them, so each eval episode reports how often / how long the
        # operator had to take over. ``_control_hz`` (set in attach_loop) converts
        # tick counts to seconds and interventions/min.
        self._interv_total_steps: int = 0
        self._interv_step_count: int = 0
        self._interv_engage_count: int = 0
        self._interv_cur_ticks: int = 0
        self._interv_episode_t0: Optional[float] = None
        self._control_hz: float = 30.0
        # Last action we emitted (raiden right-first layout). Used to hold
        # gripper positions during intervention since the leader arms have
        # no gripper sensor.
        self._last_action: Optional[np.ndarray] = None

        self._client: Optional[_WebsocketClient] = None
        self._chunk_queue: "deque[np.ndarray]" = deque()

    def load(self, ckpt_path: str, **kwargs) -> None:
        del ckpt_path  # weights live on the server
        print(f"[OpenPiBridge] connecting to ws://{self.host}:{self.port} ...")
        self._client = _WebsocketClient(self.host, self.port)
        print(f"[OpenPiBridge] connected. server metadata: {self._client.metadata}")
        if not self.prompt:
            logging.warning("[OpenPiBridge] no prompt set — server will use its default_prompt")

    def attach_loop(self, loop) -> None:
        """Stash a robot-controller reference for leader-button polling."""
        self._robot = getattr(loop, "_robot", None)
        # Loop control rate, used to report intervention durations in seconds.
        self._control_hz = float(getattr(loop, "_action_hz", 30.0) or 30.0)
        print(
            "[OpenPiBridge] features: "
            f"position_intervention={self.enable_intervention}  "
            f"button_intervention={self.enable_button_intervention}  "
            f"start_intervening={self.start_intervening}  "
            f"verbose={self.verbose_intervention}"
        )
        if self.enable_button_intervention:
            if self._robot is None:
                print(
                    "[OpenPiBridge] WARN button_intervention=true but no robot "
                    "controller — disabled"
                )
            else:
                has_r = self._robot.leader_r is not None
                has_l = self._robot.leader_l is not None
                print(f"[OpenPiBridge] leaders attached: leader_r={has_r} leader_l={has_l}")
                if not (has_r or has_l):
                    print(
                        "[OpenPiBridge] WARN no leaders initialized — did you "
                        "pass --use-leaders?"
                    )

    def _set_leaders_free(self, free: bool) -> None:
        """Zero leader PD gains (free/backdrivable) or restore them (stiff).
        No-op if the feature is disabled or no robot is attached."""
        if not self.free_leaders_on_intervention or self._robot is None:
            return
        for name, ldr in (
            ("leader_r", self._robot.leader_r),
            ("leader_l", self._robot.leader_l),
        ):
            if ldr is None:
                continue
            try:
                if free:
                    ldr.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                else:
                    kp = self._robot.kp_gains.get(name)
                    kd = self._robot.kd_gains.get(name)
                    if kp is not None and kd is not None:
                        ldr.update_kp_kd(kp=kp, kd=kd)
            except Exception as e:
                print(f"[OpenPiBridge] leader stiffness set failed ({name}): {e}")

    def _read_button_edge(self) -> Optional[str]:
        """Rising-edge detect on ``io_inputs[button_bit]`` across both leaders.
        Returns the leader name on a fresh press, else None. Uses
        ``get_encoder_io`` (no internal sleep) rather than ``get_info``."""
        fired: Optional[str] = None
        for name, ldr in (
            ("leader_r", self._robot.leader_r),
            ("leader_l", self._robot.leader_l),
        ):
            if ldr is None:
                continue
            try:
                io = ldr.get_encoder_io(encoder_idx=0)
                cur = float(io[self.button_bit]) if len(io) > self.button_bit else 0.0
            except Exception as e:
                if self.verbose_intervention:
                    print(f"[OpenPiBridge] button poll error ({name}): {e}")
                continue
            prev = self._button_prev.get(name, 0.0)
            self._button_prev[name] = cur
            if cur > 0.5 and prev < 0.5:
                fired = name
        return fired

    def _log_intervention_summary(self) -> None:
        """Print a one-line intervention-frequency summary for the episode just
        finished. Called from reset() (between episodes); a no-op if no steps
        ran, so the first reset() before any episode stays quiet."""
        if self._interv_total_steps == 0:
            return
        frac = 100.0 * self._interv_step_count / self._interv_total_steps
        interv_s = self._interv_step_count / self._control_hz
        elapsed_s = (
            (time.perf_counter() - self._interv_episode_t0)
            if self._interv_episode_t0 is not None
            else 0.0
        )
        rate = (self._interv_engage_count / elapsed_s * 60.0) if elapsed_s > 0 else 0.0
        print(
            f"[OpenPiBridge] intervention summary: {self._interv_engage_count} "
            f"interventions over {self._interv_total_steps} steps "
            f"({elapsed_s:.0f}s) — {self._interv_step_count} steps under "
            f"intervention ({frac:.1f}%, {interv_s:.1f}s), "
            f"{rate:.1f} interventions/min"
        )

    def reset(self) -> None:
        # Summarize the episode that just ended before zeroing the counters.
        self._log_intervention_summary()
        self._interv_total_steps = 0
        self._interv_step_count = 0
        self._interv_engage_count = 0
        self._interv_cur_ticks = 0
        self._interv_episode_t0 = None

        self._chunk_queue.clear()
        self._intervention.reset()
        self._was_intervening = False
        self._last_action = None
        self._button_force = self.start_intervening
        # Leaders stiff at reset (needed for homing); first predict() re-frees
        # them if we boot into intervention.
        self._set_leaders_free(False)

    def is_intervening(self) -> bool:
        """True when the last ``predict()`` returned a leader-driven command, so
        the inference loop skips its policy-jerk safety check (human leader
        motion must not trip the e-stop)."""
        return self._was_intervening

    def predict(self, obs) -> np.ndarray:
        assert self._client is not None, "load() must run before predict()"

        # Intervention-frequency bookkeeping: count every tick and stamp the
        # episode start on the first one (for the interventions/min rate).
        self._interv_total_steps += 1
        if self._interv_episode_t0 is None:
            self._interv_episode_t0 = time.perf_counter()

        # White-button toggle: rising edge on either leader flips force state.
        if self.enable_button_intervention and self._robot is not None:
            pressed = self._read_button_edge()
            if pressed is not None:
                self._button_force = not self._button_force
                print(
                    f"[OpenPiBridge] button ({pressed}, bit {self.button_bit}) → "
                    f"intervention {'ENGAGED' if self._button_force else 'RELEASED'}"
                )
            if self.verbose_intervention:
                self._verbose_tick += 1
                if self._verbose_tick % self._verbose_every == 0:
                    print(
                        f"[btn] prev={self._button_prev}  "
                        f"bit={self.button_bit}  force={self._button_force}"
                    )

        # Run intervention state machine if enabled and leader data is present.
        is_intervening = False
        leader_rl: Optional[np.ndarray] = None
        # Button override needs leader data too (we use the leader pose as cmd).
        need_leader = self.enable_intervention or (
            self.enable_button_intervention and self._button_force
        )
        if need_leader:
            leader_rl = self._build_leader_pose_rl(obs)
        if self.enable_intervention:
            if leader_rl is None and self.verbose_intervention:
                self._verbose_tick += 1
                if self._verbose_tick % self._verbose_every == 0:
                    proprios = (
                        obs.proprios if hasattr(obs, "proprios") else obs["proprios"]
                    )
                    print(
                        "[OpenPiBridge][intv] no leader data in obs.proprios; "
                        f"keys={sorted(proprios.keys())} — did you pass --use-leaders?"
                    )
            if leader_rl is not None:
                follower_rl = self._build_follower_pose_rl(obs)
                is_intervening, _ = self._intervention.step(leader_rl, follower_rl)
                if self.verbose_intervention:
                    self._verbose_tick += 1
                    if self._verbose_tick % self._verbose_every == 0:
                        idx = np.array([0, 1, 2, 3, 4, 5])
                        right_dev = np.abs(leader_rl[idx] - follower_rl[idx])
                        left_dev = np.abs(leader_rl[7 + idx] - follower_rl[7 + idx])
                        max_dev = float(max(right_dev.max(), left_dev.max()))
                        print(
                            f"[intv] max_dev={max_dev:.4f} rad  "
                            f"R={np.array2string(right_dev, precision=3)}  "
                            f"L={np.array2string(left_dev, precision=3)}  "
                            f"state={self._intervention.state.value}  "
                            f"eng_streak={self._intervention.engage_streak}"
                        )

        # Compose: position-tracking OR white-button toggle engages override.
        is_intervening = is_intervening or self._button_force

        if is_intervening:
            if not self._was_intervening:
                self._interv_engage_count += 1
                self._interv_cur_ticks = 0
                print(
                    f"[OpenPiBridge] intervention ENGAGED (#{self._interv_engage_count}) "
                    "— leader driving followers"
                )
                self._set_leaders_free(True)  # backdrivable so operator can move
            self._was_intervening = True
            self._interv_step_count += 1
            self._interv_cur_ticks += 1
            if leader_rl is None:
                # Asked to intervene but leader data missing. Hold the last
                # action so motors don't lurch.
                print(
                    "[OpenPiBridge] intervention requested but no leader data — "
                    "holding last action. Did you pass --use-leaders?"
                )
                action = (
                    self._last_action
                    if self._last_action is not None
                    else np.zeros(14, dtype=np.float32)
                )
            else:
                action = self._leader_command_rl(leader_rl)
            self._last_action = action
            return action

        if self._was_intervening:
            dur_s = self._interv_cur_ticks / self._control_hz
            frac = (
                100.0 * self._interv_step_count / self._interv_total_steps
                if self._interv_total_steps
                else 0.0
            )
            print(
                f"[OpenPiBridge] intervention RELEASED after {self._interv_cur_ticks} "
                f"ticks ({dur_s:.1f}s) — flushing stale chunk | cumulative: "
                f"{self._interv_engage_count} interventions, "
                f"{self._interv_step_count}/{self._interv_total_steps} steps "
                f"({frac:.1f}%)"
            )
            self._set_leaders_free(False)  # restiffen so leaders mirror policy
            self._chunk_queue.clear()
            self._was_intervening = False

        if not self._chunk_queue:
            self._refill_chunk(obs)
        action_lr = self._chunk_queue.popleft()  # openpi layout [L(7), R(7)]
        # Raiden's inference layout is right-first: [R(7), L(7)].
        action = np.concatenate([action_lr[DOF:], action_lr[:DOF]]).astype(
            np.float32, copy=False
        )
        self._last_action = action
        return action

    def _refill_chunk(self, obs) -> None:
        request = self._build_request(obs)
        t0 = time.perf_counter()
        resp = self._client.infer(request)
        dt_ms = (time.perf_counter() - t0) * 1e3
        actions = np.asarray(resp["actions"], dtype=np.float32)
        timing = resp.get("server_timing", {})
        print(
            f"[OpenPiBridge] chunk: shape={actions.shape} "
            f"infer={timing.get('infer_ms', 'n/a')}ms wire={dt_ms:.1f}ms"
        )
        n = min(self.execute_len, actions.shape[0])
        for i in range(n):
            self._chunk_queue.append(actions[i])

    def _build_follower_pose_rl(self, obs) -> np.ndarray:
        """Follower joint pose in raiden inference layout [R(7), L(7)]."""
        proprios = obs.proprios if hasattr(obs, "proprios") else obs["proprios"]
        right = proprios.get("follower_r_joint_pos", np.zeros(DOF, dtype=np.float32))
        left = proprios.get("follower_l_joint_pos", np.zeros(DOF, dtype=np.float32))
        return np.concatenate([right, left]).astype(np.float32, copy=False)

    def _build_leader_pose_rl(self, obs) -> Optional[np.ndarray]:
        """Leader joint pose in raiden inference layout [R(7), L(7)].
        Leaders are 6-DoF (arm only); the gripper slot is zero-filled — the
        intervention controller only looks at arm joints (indices 0..5 of
        each arm). Returns None if no leader data is in the observation."""
        proprios = obs.proprios if hasattr(obs, "proprios") else obs["proprios"]
        if (
            "leader_r_joint_pos" not in proprios
            and "leader_l_joint_pos" not in proprios
        ):
            return None
        pose = np.zeros(14, dtype=np.float32)
        if "leader_r_joint_pos" in proprios:
            pose[0:6] = proprios["leader_r_joint_pos"]
        if "leader_l_joint_pos" in proprios:
            pose[7:13] = proprios["leader_l_joint_pos"]
        return pose

    def _leader_command_rl(self, leader_rl: np.ndarray) -> np.ndarray:
        """Build a 14-D follower command from a leader pose.

        Arm joints come from the leader. Grippers follow the operator's
        teaching-handle trigger *absolutely* (read live from the leader): the
        instant an intervention engages, the gripper jumps to whatever the
        trigger commands — at rest that is fully open, so the gripper opens on
        handoff — and then tracks the trigger directly (squeeze → close, release
        → open). When a trigger isn't available it falls back to holding the
        last commanded value."""
        cmd = leader_rl.copy()
        # Fallback: hold grippers at their last commanded value.
        if self._last_action is not None:
            cmd[6] = self._last_action[6]    # right gripper
            cmd[13] = self._last_action[13]  # left gripper
        # Preferred: drive grippers directly from the live teaching-handle
        # triggers (absolute — opens on engage, then tracks the trigger).
        if self._robot is not None:
            right_grip, left_grip = self._robot.read_leader_grippers()
            cmd = apply_leader_grippers(cmd, right_grip, left_grip)
        return cmd

    def _build_request(self, obs) -> dict:
        # --- proprio -> state in left-first openpi layout ---
        proprios = obs.proprios if hasattr(obs, "proprios") else obs["proprios"]
        right = proprios.get("follower_r_joint_pos", np.zeros(DOF, dtype=np.float32))
        left = proprios.get("follower_l_joint_pos", np.zeros(DOF, dtype=np.float32))
        state = np.concatenate([left, right]).astype(np.float32, copy=False)

        # --- camera mapping: raiden -> openpi yam keys ---
        cameras = obs.cameras if hasattr(obs, "cameras") else obs.get("cameras", [])
        cam_by_name = {(c.name if hasattr(c, "name") else c["name"]): c for c in cameras}

        def _img(name_options):
            for name in name_options:
                if name in cam_by_name:
                    c = cam_by_name[name]
                    img = c.image if hasattr(c, "image") else c["image"]
                    if img is not None:
                        return self._prep_image(np.asarray(img))
            return np.zeros((IMG_HW, IMG_HW, 3), dtype=np.uint8)

        return {
            "observation/state": state,
            "observation/image": _img(["scene_camera", "head_camera", "scene"]),
            "observation/left_image": _img(["left_wrist_camera", "left_wrist"]),
            "observation/right_image": _img(["right_wrist_camera", "right_wrist"]),
            "prompt": self.prompt,
        }

    @staticmethod
    def _prep_image(img: np.ndarray) -> np.ndarray:
        """Ensure HWC uint8 RGB at 224x224."""
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        h, w = img.shape[:2]
        if (h, w) != (IMG_HW, IMG_HW):
            img = cv2.resize(img, (IMG_HW, IMG_HW), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img)

    def __del__(self):
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
