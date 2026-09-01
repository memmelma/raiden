"""OpenPi-beta (pi07 / BimanualARX) eval bridge for the YAM bimanual robot.

This is the beta counterpart to :mod:`raiden.inference_bridges.openpi_bridge`.
The websocket + msgpack transport is identical, so this subclasses
``OpenPiBridge`` and overrides only the parts that differ: the request payload,
the response parsing, and the image/gripper conventions.

Wire format (per ``src/openpi/policies/bimanual_arx_policy.py`` on the server)
differs from the pi05 ``yam_policy`` contract -- it is structured, not flat, and
is validated strictly server-side:

    Request (each step):
        state/left_arm        (6,)  float32  joint angles (rad)
        state/left_gripper    (1,)  float32  normalized, MUST be in [0, 1]
        state/right_arm       (6,)  float32
        state/right_gripper   (1,)  float32
        images/cam_high       (H,W,3) uint8 RGB  -- or JPEG bytes
        images/cam_left_wrist    "
        images/cam_right_wrist   "
        prompt                str

    Response:
        actions/left_arm      (H,6) float32  ABSOLUTE joint targets (rad)
        actions/left_gripper  (H,1) float32  normalized absolute, [0, 1]
        actions/right_arm     (H,6) float32
        actions/right_gripper (H,1) float32
        server_timing         dict

Notes that cost real rollout quality if ignored:

* **Images are sent at native resolution.** ``BimanualARXInputs`` resizes with
  *padding* to 448x448 server-side, preserving aspect ratio. Pre-squashing to a
  square client-side (raiden's ``--resize_images`` default is ``384x384``)
  destroys the aspect ratio before the server ever sees the frame, so run with
  ``--resize_images ''`` and let the server do it. This matches PI's own runtime.
* **Actions come back absolute.** The server config uses delta joint actions
  internally (``use_delta_joint_actions=True``) but ``AbsoluteActions`` undoes
  that before the response, so what arrives here is directly commandable.
* **Control rate is 50 Hz**, not raiden's 30 Hz default -- see
  ``packages/openpi-runtime/configs/yam_bimanual.toml`` (``control_hz = 50.0``).
  PI's reference eval executes 25 of the 50-step chunk before re-querying, which
  is the ``execute_len`` default here.
* **Gripper direction is a hardware fact, not a guess.** See
  ``robot_gripper_open_value`` below and ``scripts/probe_gripper_convention.py``.

Run::

    rd infer \\
        --bridge raiden.inference_bridges.openpi_beta_bridge:OpenPiBetaBridge \\
        --ckpt_path unused \\
        --action_hz 50 \\
        --resize_images '' \\
        --bridge-kwargs \\
            host=localhost port=8000 \\
            robot_gripper_open_value=1 \\
            prompt='Both arms pick up the green bowl and lift it'

The prompt is fixed for the whole run: it is read once at launch and every
request sends that same string. Change it by restarting with a different
``prompt=``.

That default string is not arbitrary -- it is the verbatim
``language_prompt`` in every PickGreenBowl episode
(``data/processed/PickGreenBowl/*/lowdim/*.pkl``), which
``scripts/convert_pickgreenbowl_to_lerobot_pi07.py`` copies straight into the
LeRobot ``task`` field. Serving a paraphrase feeds the policy a
conditioning vector it never saw in training, so it must match exactly.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from raiden.inference_bridges.openpi_bridge import DOF, OpenPiBridge

# Server-side camera keys. BimanualARXInputs raises unless the set matches
# exactly, so these are not merely conventional.
CAM_HIGH = "cam_high"
CAM_LEFT_WRIST = "cam_left_wrist"
CAM_RIGHT_WRIST = "cam_right_wrist"

# Raiden camera names to try for each server key, in priority order.
_CAM_SOURCES = {
    CAM_HIGH: ("scene_camera", "head_camera", "scene", "scene_1"),
    CAM_LEFT_WRIST: ("left_wrist_camera", "left_wrist"),
    CAM_RIGHT_WRIST: ("right_wrist_camera", "right_wrist"),
}

# Fallback frame size when a camera is missing. Matches the RealSense color
# stream raiden configures (see raiden/cameras/realsense.py).
_FALLBACK_HW = (480, 640)

# PI's reference eval (packages/openpi-runtime/src/openpi_runtime/evaluate.py)
# executes 25 steps of the 50-step horizon before re-querying.
_DEFAULT_EXECUTE_LEN = 25

# The exact ``language_prompt`` recorded in every PickGreenBowl episode, hence
# the exact ``task`` string the pi07_pickgreenbowl_lora checkpoint was trained
# on. Used only when the caller passes no ``prompt=``; the parent's default is
# the empty string, which the policy was never conditioned on.
_PICKGREENBOWL_PROMPT = "Both arms pick up the green bowl and lift it"


def _truthy(v) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


class OpenPiBetaBridge(OpenPiBridge):
    """Closed-loop eval bridge for an openpi-beta (pi07) BimanualARX server.

    Extra ``--bridge-kwargs`` beyond those of :class:`OpenPiBridge`:

    ``robot_gripper_open_value``
        Which normalized value raiden's follower reports/accepts for a *fully
        open* gripper -- ``1`` or ``0``. i2rt's ``JointMapper`` normalizes
        against ``gripper_limits`` ordered ``[closed, open]``, which implies
        ``1``; raiden's ``leader_io`` docstring claims ``0``. They contradict and
        are unresolved on this cell, so this stays configurable rather than
        hardcoded. Default ``1`` (the i2rt-implied value).

    ``policy_gripper_open_value``
        Which value the *policy* uses for open. Default ``1``: the pi07
        pickgreenbowl checkpoint was fine-tuned on raiden-native, *unflipped*
        grippers (the recorder/converter apply no flip), so the policy's gripper
        convention matches the robot's i2rt-native one (``1``=open). With both
        sides ``1`` the mapping is passthrough (no inversion). Set to ``0`` only
        for a base openpi checkpoint that uses the ``0``=open norm-stats
        convention.

    ``jpeg``
        Encode frames as JPEG before sending (default off). Cuts ~2.7 MB/request
        to ~100 KB; worth enabling for a remote GPU, unnecessary on localhost.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("execute_len", _DEFAULT_EXECUTE_LEN)
        super().__init__(**kwargs)

        # Fall back to the PickGreenBowl training prompt rather than the
        # parent's empty string. Flagged at load() so a run against some other
        # pi07 checkpoint can't silently inherit a bowl prompt.
        self._prompt_defaulted = not self.prompt
        if self._prompt_defaulted:
            self.prompt = _PICKGREENBOWL_PROMPT

        # Gripper direction. Both sides of the mapping are explicit so a wrong
        # assumption shows up in the startup banner rather than as a policy that
        # mysteriously refuses to grasp.
        self.robot_gripper_open_value = float(kwargs.get("robot_gripper_open_value", 1.0))
        # Default 1.0 (not 0.0): this checkpoint was trained on unflipped native
        # grippers, so policy convention == robot convention -> passthrough.
        self.policy_gripper_open_value = float(kwargs.get("policy_gripper_open_value", 1.0))
        self._invert_gripper = (
            self.robot_gripper_open_value != self.policy_gripper_open_value
        )

        self.jpeg: bool = _truthy(kwargs.get("jpeg", "0"))
        self.jpeg_quality: int = int(kwargs.get("jpeg_quality", 95))

        # Gripper trace. The action CSV in RaidenInferenceLoop only sees the
        # ~25 steps of each chunk that get executed, after the robot/policy
        # convention mapping. To tell "the policy never commands a close" apart
        # from "the close is mapped or truncated away", the raw server output
        # has to be recorded too -- the whole horizon, unmodified.
        self.gripper_log = kwargs.get("gripper_log", "rollout_logs")
        self._grip_log_fh = None
        self._grip_log_path = None
        self._grip_infer_idx = 0

    # -- lifecycle ---------------------------------------------------------

    def load(self, ckpt_path: str, **kwargs) -> None:
        super().load(ckpt_path, **kwargs)
        print(
            "[OpenPiBetaBridge] gripper mapping: robot "
            f"{self.robot_gripper_open_value:.0f}=open -> policy "
            f"{self.policy_gripper_open_value:.0f}=open "
            f"({'INVERTED' if self._invert_gripper else 'passthrough'})"
        )
        print(
            f"[OpenPiBetaBridge] execute_len={self.execute_len}  "
            f"jpeg={self.jpeg}  images=native (server pads to 448x448)"
        )
        origin = "PickGreenBowl training default" if self._prompt_defaulted else "explicit"
        print(f"[OpenPiBetaBridge] fixed prompt ({origin}): {self.prompt!r}")
        self._open_gripper_log()

    # -- gripper trace -----------------------------------------------------

    def _open_gripper_log(self) -> None:
        """Open the per-inference gripper trace CSV.

        One row per horizon step of every chunk (not just the executed
        prefix), so a chunk whose close happens after ``execute_len`` is still
        visible in the record.
        """
        if not self.gripper_log:
            return
        path = Path(self.gripper_log)
        if path.suffix != ".csv":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path / f"gripper_{stamp}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._grip_log_path = path
        self._grip_log_fh = open(path, "w", buffering=1 << 16)
        self._grip_log_fh.write(
            ",".join(
                [
                    "infer_idx",      # which server query this row came from
                    "t_wall",
                    "horizon_i",      # index within the returned chunk
                    "executed",       # 1 if horizon_i < execute_len
                    "state_r_grip",   # gripper sent TO the server, policy convention
                    "state_l_grip",
                    "raw_r_grip",     # gripper FROM the server, untouched
                    "raw_l_grip",
                    "cmd_r_grip",     # after robot/policy convention mapping
                    "cmd_l_grip",
                ]
            )
            + "\n"
        )
        print(f"[OpenPiBetaBridge] logging gripper trace to {path}")

    def _log_gripper_chunk(
        self,
        state_r: float,
        state_l: float,
        raw_right: np.ndarray,
        raw_left: np.ndarray,
        cmd_right: np.ndarray,
        cmd_left: np.ndarray,
    ) -> None:
        if self._grip_log_fh is None:
            return
        t = time.time()
        for i in range(raw_right.shape[0]):
            self._grip_log_fh.write(
                f"{self._grip_infer_idx},{t:.6f},{i},"
                f"{int(i < self.execute_len)},"
                f"{state_r:.6f},{state_l:.6f},"
                f"{float(raw_right[i]):.6f},{float(raw_left[i]):.6f},"
                f"{float(cmd_right[i]):.6f},{float(cmd_left[i]):.6f}\n"
            )
        self._grip_infer_idx += 1

    def close(self) -> None:
        if self._grip_log_fh is not None:
            self._grip_log_fh.close()
            self._grip_log_fh = None
            print(f"[OpenPiBetaBridge] wrote gripper trace -> {self._grip_log_path}")
        parent_close = getattr(super(), "close", None)
        if parent_close is not None:
            parent_close()

    # -- gripper conversion ------------------------------------------------

    def _robot_to_policy_grip(self, value) -> np.ndarray:
        g = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
        return (1.0 - g) if self._invert_gripper else g

    def _policy_to_robot_grip(self, value) -> np.ndarray:
        # The mapping is an involution (both ends are [0,1] with only the
        # direction possibly flipped), so this mirrors the forward conversion.
        g = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
        return (1.0 - g) if self._invert_gripper else g

    # -- request / response ------------------------------------------------

    def _build_request(self, obs) -> dict:
        proprios = obs.proprios if hasattr(obs, "proprios") else obs["proprios"]
        right = np.asarray(
            proprios.get("follower_r_joint_pos", np.zeros(DOF, dtype=np.float32)),
            dtype=np.float32,
        )
        left = np.asarray(
            proprios.get("follower_l_joint_pos", np.zeros(DOF, dtype=np.float32)),
            dtype=np.float32,
        )

        cameras = obs.cameras if hasattr(obs, "cameras") else obs.get("cameras", [])
        cam_by_name = {
            (c.name if hasattr(c, "name") else c["name"]): c for c in cameras
        }

        images = {}
        for key, sources in _CAM_SOURCES.items():
            images[key] = self._lookup_image(cam_by_name, sources, key)

        return {
            "state": {
                "left_arm": left[:6],
                "left_gripper": self._robot_to_policy_grip(left[6:7]),
                "right_arm": right[:6],
                "right_gripper": self._robot_to_policy_grip(right[6:7]),
            },
            "images": images,
            "prompt": self.prompt,
        }

    def _lookup_image(self, cam_by_name: dict, sources: tuple, key: str):
        for name in sources:
            if name in cam_by_name:
                c = cam_by_name[name]
                img = c.image if hasattr(c, "image") else c["image"]
                if img is not None:
                    return self._prep_image_native(np.asarray(img))
        if not getattr(self, f"_warned_{key}", False):
            print(
                f"[OpenPiBetaBridge] WARN no camera for {key} "
                f"(tried {sources}) — sending a black frame."
            )
            setattr(self, f"_warned_{key}", True)
        return self._prep_image_native(np.zeros((*_FALLBACK_HW, 3), dtype=np.uint8))

    def _prep_image_native(self, img: np.ndarray):
        """Return HWC uint8 RGB at native resolution (or JPEG bytes).

        Deliberately does not resize: the server resizes with padding, which
        preserves aspect ratio. Contrast with ``OpenPiBridge._prep_image``,
        which squashes to 224x224 for the pi05 contract.
        """
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)
        if not self.jpeg:
            return img
        # cv2 encodes BGR, and raiden hands us RGB (server.py converts on
        # capture), so flip for the encoder and the server decodes RGB back.
        ok, buf = cv2.imencode(
            ".jpg", img[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return img
        return buf.tobytes()

    def reset(self) -> None:
        # ``RaidenInferenceLoop`` calls reset() at the end of a rollout but
        # never close(), so this is the flush that actually happens.
        if self._grip_log_fh is not None:
            self._grip_log_fh.flush()
        super().reset()

    def _refill_chunk(self, obs) -> None:
        request = self._build_request(obs)
        t0 = time.perf_counter()
        resp = self._client.infer(request)
        dt_ms = (time.perf_counter() - t0) * 1e3

        actions = resp["actions"]
        left_arm = np.asarray(actions["left_arm"], dtype=np.float32)
        left_grip = np.asarray(actions["left_gripper"], dtype=np.float32)
        right_arm = np.asarray(actions["right_arm"], dtype=np.float32)
        right_grip = np.asarray(actions["right_gripper"], dtype=np.float32)

        # Parent's predict() expects queue entries in openpi's left-first flat
        # layout [L(7), R(7)] and flips them to raiden's right-first order.
        chunk = np.concatenate(
            [
                left_arm,
                self._policy_to_robot_grip(left_grip),
                right_arm,
                self._policy_to_robot_grip(right_grip),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

        cmd_left_grip = self._policy_to_robot_grip(left_grip).reshape(-1)
        cmd_right_grip = self._policy_to_robot_grip(right_grip).reshape(-1)
        self._log_gripper_chunk(
            state_r=float(np.asarray(request["state"]["right_gripper"]).reshape(-1)[0]),
            state_l=float(np.asarray(request["state"]["left_gripper"]).reshape(-1)[0]),
            raw_right=right_grip.reshape(-1),
            raw_left=left_grip.reshape(-1),
            cmd_right=cmd_right_grip,
            cmd_left=cmd_left_grip,
        )

        timing = resp.get("server_timing", {})
        n_exec = min(self.execute_len, chunk.shape[0])
        print(
            f"[OpenPiBetaBridge] chunk: shape={chunk.shape} "
            f"infer={timing.get('infer_ms', 'n/a')}ms wire={dt_ms:.1f}ms"
        )
        # Range over the executed prefix vs the full horizon: a policy that
        # only commands the close in the tail of the chunk never gets to
        # execute it, and that shows up here as a flat prefix.
        print(
            "[OpenPiBetaBridge] grip cmd R "
            f"exec[{cmd_right_grip[:n_exec].min():.3f},{cmd_right_grip[:n_exec].max():.3f}] "
            f"full[{cmd_right_grip.min():.3f},{cmd_right_grip.max():.3f}]  L "
            f"exec[{cmd_left_grip[:n_exec].min():.3f},{cmd_left_grip[:n_exec].max():.3f}] "
            f"full[{cmd_left_grip.min():.3f},{cmd_left_grip.max():.3f}]"
        )

        n = min(self.execute_len, chunk.shape[0])
        for i in range(n):
            self._chunk_queue.append(chunk[i])
