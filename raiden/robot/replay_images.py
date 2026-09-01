"""Image sources for ``rd replay`` debug logging.

Two independent image sources are exposed so ``rd replay`` can log, at a low
sampling rate, what the *live* robot's cameras saw alongside what the
*recorded dataset* shows at the same point in the trajectory:

- :class:`LiveCameraGrabber` — opens the real cameras (via ``camera.json`` /
  :class:`~raiden.camera_config.CameraConfig`) and grabs single frames
  on demand. Much lighter than ``RaidenPolicyServer``'s camera threads: no
  background capture loop, no depth, no phase-offset sync — just
  open → grab → get_frame per call.
- Dataset readers — read back the *recorded* frames for the episode being
  replayed, indexed by control step:

  - :class:`ProcessedFrameReader` — for converted episodes (``rd convert``
    output): frames are individual PNGs under ``rgb/<camera>/``, an exact
    O(1) lookup.
  - :class:`RawFrameReader` — for raw episodes (``rd record`` output):
    frames live inside native ``.svo2``/``.bag`` containers with no random
    access, so this decodes sequentially and only ever advances forward,
    approximating "the frame at replay time t" from each container's total
    frame count and the recording's duration (camera frame timestamps are
    not necessarily aligned 1:1 with robot proprio timestamps).

All frames are returned as ``(H, W, 3)`` uint8 **RGB** arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from raiden.camera_config import CameraConfig


def list_recording_cameras(recording_dir: Path) -> list[str]:
    """Return camera names available in a recording directory (raw or processed)."""
    cameras_dir = recording_dir / "cameras"
    if cameras_dir.is_dir():
        return sorted({p.stem for p in cameras_dir.glob("*.svo2")} |
                      {p.stem for p in cameras_dir.glob("*.bag")})
    rgb_dir = recording_dir / "rgb"
    if rgb_dir.is_dir():
        return sorted(p.name for p in rgb_dir.iterdir() if p.is_dir())
    return []


class LiveCameraGrabber:
    """Opens a set of real cameras and grabs single frames on demand.

    Cameras are opened once (in ``__enter__``/construction) and closed once
    at the end of the rollout — grabbing a frame is just ``grab() +
    get_frame()``, no background threads.
    """

    def __init__(self, camera_names: list[str], config_file: Optional[str] = None):
        cfg = CameraConfig(config_file) if config_file else CameraConfig()
        self._cameras: dict[str, "object"] = {}
        for name in camera_names:
            if name not in cfg.list_camera_names():
                continue
            try:
                cam = cfg.create_camera(name)
                cam.open()
                self._cameras[name] = cam
            except Exception as e:  # noqa: BLE001 - best-effort, keep replay running
                print(f"[replay_images] failed to open live camera '{name}': {e}")
        if not self._cameras:
            print("[replay_images] no live cameras opened — live image logging disabled.")

    @property
    def camera_names(self) -> list[str]:
        return list(self._cameras.keys())

    def grab_all(self) -> dict[str, np.ndarray]:
        """Grab one frame from every opened camera. Returns {name: RGB array}."""
        images: dict[str, np.ndarray] = {}
        for name, cam in self._cameras.items():
            try:
                if not cam.grab():
                    continue
                frame = cam.get_frame()
                images[name] = frame.color[..., ::-1].copy()  # BGR -> RGB
            except Exception as e:  # noqa: BLE001
                print(f"[replay_images] grab failed for '{name}': {e}")
        return images

    def close(self) -> None:
        for cam in self._cameras.values():
            try:
                cam.close()
            except Exception:  # noqa: BLE001
                pass
        self._cameras.clear()


class ProcessedFrameReader:
    """Reads recorded frames for a *processed* (``rd convert``-output) episode.

    ``get_frame(control_step)`` maps a control-loop step index back to the
    original camera-rate keyframe index using the same ``upsample`` factor
    and ``stride`` the replay used, then loads that keyframe's PNG.
    """

    def __init__(
        self,
        recording_dir: Path,
        camera_names: list[str],
        upsample: int,
        stride: int = 1,
    ):
        self.recording_dir = recording_dir
        self.camera_names = camera_names
        self.upsample = max(1, upsample)
        self.stride = max(1, stride)

    def get_frame(self, control_step: int) -> dict[str, np.ndarray]:
        keyframe_idx = (control_step // self.upsample) * self.stride
        images: dict[str, np.ndarray] = {}
        for name in self.camera_names:
            path = self.recording_dir / "rgb" / name / f"{keyframe_idx:010d}.png"
            if not path.exists():
                continue
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            images[name] = bgr[..., ::-1].copy()
        return images

    def close(self) -> None:
        pass


class RawFrameReader:
    """Reads recorded frames for a *raw* (``rd record``-output) episode.

    SVO2/bag containers have no random access, so each camera is decoded
    strictly forward: ``get_frame(control_step)`` advances that camera to the
    frame index proportional to ``control_step``'s fraction of the total
    replay duration (``total_frames * control_step / total_control_steps``),
    grabbing forward as needed. Approximate — camera and robot capture aren't
    guaranteed frame-exact — but adequate for "did the robot roughly look
    like this" debugging.

    ZED SVO2 exposes ``get_total_frames()``; RealSense ``.bag`` does not, so
    for bags we estimate frame count from ``robot_data.npz`` duration × 30 fps.
    """

    def __init__(self, recording_dir: Path, camera_names: list[str], total_control_steps: int):
        self.total_control_steps = max(1, total_control_steps)
        self._decoders: dict[str, object] = {}
        self._total_frames: dict[str, int] = {}
        self._cur_frame: dict[str, int] = {}

        estimated = _estimate_raw_camera_frames(recording_dir)
        cameras_dir = recording_dir / "cameras"
        for name in camera_names:
            svo_path = cameras_dir / f"{name}.svo2"
            bag_path = cameras_dir / f"{name}.bag"
            try:
                if svo_path.exists():
                    from raiden.cameras.zed import ZedCamera

                    dec = ZedCamera.from_svo(name, svo_path, compute_sdk_depth=False)
                elif bag_path.exists():
                    from raiden.cameras.realsense import RealSenseCamera

                    dec = RealSenseCamera.from_bag(name, bag_path)
                else:
                    continue
            except Exception as e:  # noqa: BLE001
                print(f"[replay_images] failed to open dataset camera '{name}': {e}")
                continue
            self._decoders[name] = dec
            total = getattr(dec, "get_total_frames", lambda: 0)() or 0
            if total <= 0:
                total = estimated
            if total <= 0:
                print(
                    f"[replay_images] unknown frame count for '{name}' — "
                    "dataset image logging disabled for this camera."
                )
                try:
                    dec.close()
                except Exception:  # noqa: BLE001
                    pass
                del self._decoders[name]
                continue
            self._total_frames[name] = total
            self._cur_frame[name] = -1
            print(f"[replay_images] dataset camera '{name}': ~{total} frames")

    def get_frame(self, control_step: int) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for name, dec in self._decoders.items():
            total = self._total_frames.get(name, 0)
            if total <= 0:
                continue
            target = min(
                total - 1, int(total * control_step / self.total_control_steps)
            )
            cur = self._cur_frame[name]
            ok = True
            while cur < target and ok:
                ok = dec.grab()
                cur += 1
            self._cur_frame[name] = cur
            if not ok or cur < 0:
                continue
            try:
                frame = dec.get_frame()
            except Exception as e:  # noqa: BLE001
                print(f"[replay_images] get_frame failed for '{name}': {e}")
                continue
            images[name] = frame.color[..., ::-1].copy()  # BGR -> RGB
        return images

    def close(self) -> None:
        for dec in self._decoders.values():
            try:
                dec.close()
            except Exception:  # noqa: BLE001
                pass
        self._decoders.clear()


def _estimate_raw_camera_frames(recording_dir: Path, fps: float = 30.0) -> int:
    """Estimate camera frame count from robot_data.npz duration (bags have no count)."""
    npz_path = recording_dir / "robot_data.npz"
    if not npz_path.exists():
        return 0
    try:
        data = np.load(npz_path)
        ts = data["timestamps"]
        if len(ts) < 2:
            return 0
        duration_s = float(ts[-1] - ts[0]) / 1e9
        return max(1, int(round(duration_s * fps)))
    except Exception:  # noqa: BLE001
        return 0
