"""Camera health checks shared by ``rd record``, ``rd infer``, and ``rd serve``.

Motivating bug: during a real ``rd infer`` rollout, the ``left_wrist_camera``
RealSense stream never delivered a usable frame. ``RaidenPolicyServer``
pre-allocates each camera's image buffer as zeros and only overwrites it once
a capture thread successfully grabs a frame; the previous
``_wait_for_first_frames`` helper only *printed a warning* when a camera's
timestamp never advanced past zero, so the policy silently ran an entire
~200-step rollout conditioned on a solid-black wrist image with no visible
error — the failure only surfaced hours later, offline, in
``scripts/visualize_rollout.py``.

This module gives both the recorder and the policy server a way to:

1. Fail fast (raise :class:`CameraHealthError`) if a camera's first frame is
   black/degenerate, instead of silently proceeding.
2. Keep watching every subsequent frame (:class:`CameraHealthMonitor`) so a
   camera that goes bad *mid-session* (not just at startup) produces a loud,
   rate-limited warning instead of disappearing into a rollout log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Union

import numpy as np


class CameraHealthError(RuntimeError):
    """Raised when one or more cameras fail a startup health check."""


@dataclass
class FrameHealth:
    mean: float
    std: float
    frac_near_black: float
    is_black: bool
    is_frozen: bool = False

    @property
    def is_degenerate(self) -> bool:
        return self.is_black or self.is_frozen

    def reason(self) -> str:
        if self.is_black:
            return "black"
        if self.is_frozen:
            return "frozen"
        return "ok"


def check_frame(
    image: np.ndarray,
    black_level: float = 2.0,
    black_frac_thresh: float = 0.98,
) -> FrameHealth:
    """Flag a frame as (near-)black.

    Args:
        image: ``(H, W, C)`` (or any shape) array, any numeric dtype.
        black_level: per-channel intensity (0-255 scale) below which a pixel
            counts as "dark".
        black_frac_thresh: fraction of dark pixels above which the whole
            frame is flagged as black. Kept just under 1.0 so a frame with a
            few genuinely dark pixels (e.g. a shadow) doesn't false-positive.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        return FrameHealth(mean=0.0, std=0.0, frac_near_black=1.0, is_black=True)
    mean = float(arr.mean())
    std = float(arr.std())
    frac_dark = float((arr <= black_level).mean())
    is_black = frac_dark >= black_frac_thresh
    return FrameHealth(mean=mean, std=std, frac_near_black=frac_dark, is_black=is_black)


def check_cameras_at_startup(
    images: Dict[str, np.ndarray],
    black_level: float = 2.0,
    black_frac_thresh: float = 0.98,
    context: str = "",
) -> None:
    """Raise :class:`CameraHealthError` if any camera's frame is black.

    Call this right after every camera has delivered (what should be) its
    first real frame — before recording starts or before a policy is allowed
    to run — so a dead/miscabled/lens-capped camera aborts immediately
    instead of silently producing a corrupted episode or rollout.
    """
    bad: List[str] = []
    for name, image in images.items():
        health = check_frame(image, black_level, black_frac_thresh)
        if health.is_black:
            bad.append(
                f"{name} (mean={health.mean:.1f}, dark_frac={health.frac_near_black:.2f})"
            )
    if bad:
        prefix = f"[{context}] " if context else ""
        raise CameraHealthError(
            f"{prefix}Camera(s) produced a black/degenerate frame at startup — "
            f"refusing to proceed: {', '.join(bad)}. Check lens caps, physical "
            "camera/USB connections, USB bandwidth (RealSense: try --no-depth), "
            "and serials in camera.json."
        )


def _content_signature(image: np.ndarray) -> bytes:
    """Cheap content fingerprint that still changes under sensor noise.

    A global mean is *not* a usable freeze signal: a live but visually-stable
    scene (robot holding still, letterboxed black pads dominating the average)
    routinely drifts by ≪ 0.5 grey levels between samples while pixels keep
    updating. Striding a few rows/cols keeps this O(small) and bit-exact when
    the capture buffer is stuck on the same frame.
    """
    arr = np.ascontiguousarray(image)
    # ~24×24 samples from a 384×384 frame; still catches an exact stuck buffer.
    return arr[::16, ::16].tobytes()


class CameraHealthMonitor:
    """Tracks per-camera frame health over time to catch cameras that go bad
    (or never come up) mid-session, not just at startup.

    Call :meth:`update` once per captured frame (or at a lower sample rate to
    reduce overhead). Returns a human-readable warning string the moment a
    camera's status flips healthy -> unhealthy, then again at most every
    ``min_repeat_interval_s`` while it stays unhealthy (so logs don't flood),
    or ``None`` otherwise.

    Freeze detection prefers a monotonically advancing capture ``timestamp``
    (Unix ns from the camera SDK) when provided — that is the reliable signal
    that the capture thread is still writing. Without a timestamp, falls back
    to a strided content fingerprint (not mean brightness).
    """

    def __init__(
        self,
        black_level: float = 2.0,
        black_frac_thresh: float = 0.98,
        freeze_after_s: float = 1.0,
        min_repeat_interval_s: float = 5.0,
    ) -> None:
        self._black_level = black_level
        self._black_frac_thresh = black_frac_thresh
        self._freeze_after_s = freeze_after_s
        self._min_repeat_interval_s = min_repeat_interval_s

        self._last_signature: Dict[str, Hashable] = {}
        self._last_change_ts: Dict[str, float] = {}
        self._was_healthy: Dict[str, bool] = {}
        self._last_warn_ts: Dict[str, float] = {}
        self._ever_unhealthy: Dict[str, bool] = {}

    def update(
        self,
        name: str,
        image: np.ndarray,
        now: Optional[float] = None,
        timestamp: Optional[Union[int, float]] = None,
    ) -> Optional[str]:
        now = now if now is not None else time.monotonic()
        health = check_frame(image, self._black_level, self._black_frac_thresh)

        # Prefer capture-timestamp advancement (server path). Fall back to a
        # content fingerprint when no timestamp is available (recorder path).
        if timestamp is not None:
            sig: Hashable = timestamp
        else:
            sig = _content_signature(image)

        prev_sig = self._last_signature.get(name)
        if prev_sig is None or prev_sig != sig:
            self._last_change_ts[name] = now
        self._last_signature[name] = sig
        health.is_frozen = (now - self._last_change_ts.get(name, now)) >= self._freeze_after_s

        healthy = not health.is_degenerate
        was_healthy = self._was_healthy.get(name, True)
        self._was_healthy[name] = healthy
        if not healthy:
            self._ever_unhealthy[name] = True

        if healthy:
            return None

        last_warn = self._last_warn_ts.get(name, -1e9)
        if was_healthy or (now - last_warn) >= self._min_repeat_interval_s:
            self._last_warn_ts[name] = now
            return (
                f"[camera-health] '{name}' looks {health.reason()} "
                f"(mean={health.mean:.1f} std={health.std:.1f} "
                f"dark_frac={health.frac_near_black:.2f}) — "
                "check the physical camera / USB connection."
            )
        return None

    def is_healthy(self, name: str) -> bool:
        return self._was_healthy.get(name, True)

    def unhealthy_cameras(self) -> List[str]:
        """Cameras currently flagged black/frozen."""
        return [n for n, ok in self._was_healthy.items() if not ok]

    def ever_unhealthy_cameras(self) -> List[str]:
        """Cameras that were flagged at any point since this monitor was created."""
        return [n for n, bad in self._ever_unhealthy.items() if bad]
