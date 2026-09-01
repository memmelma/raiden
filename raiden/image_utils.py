"""Aspect-ratio-preserving image resize shared across raiden's camera capture
loops and model bridges.

Mirrors ``openpi.shared.image_tools.resize_with_pad`` (letterbox: uniform
scale + centered zero-pad) so a live camera frame served to a policy goes
through the *same* geometric transform as the frames the policy was trained
on. Plain ``cv2.resize(img, (w_out, h_out))`` to a different aspect ratio
(e.g. camera-native 640x480 -> square 384x384) *stretches* the image
non-uniformly instead, which is a real train/inference distribution shift —
see the investigation that led to this module.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _resize_pad_geometry(
    h_src: int, w_src: int, h_out: int, w_out: int
) -> Tuple[float, int, int, int, int, int, int]:
    """Compute the letterbox geometry for resizing (h_src, w_src) -> (h_out, w_out).

    Returns:
        scale, resized_h, resized_w, pad_h0, pad_h1, pad_w0, pad_w1
    """
    ratio = max(w_src / w_out, h_src / h_out)
    resized_h = max(1, int(h_src / ratio))
    resized_w = max(1, int(w_src / ratio))
    pad_h0, rem_h = divmod(h_out - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(w_out - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    scale = 1.0 / ratio
    return scale, resized_h, resized_w, pad_h0, pad_h1, pad_w0, pad_w1


def resize_with_pad(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: float = 0,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize ``image`` to ``(target_h, target_w)`` without distorting its
    aspect ratio, letterboxing with ``pad_value`` (default: black).

    Works for both color (H, W, C) and single-channel (H, W) arrays (e.g.
    depth maps), so color and depth from the same camera stay pixel-aligned
    when resized independently with the same target size.
    """
    h_src, w_src = image.shape[:2]
    scale, resized_h, resized_w, pad_h0, pad_h1, pad_w0, pad_w1 = _resize_pad_geometry(
        h_src, w_src, target_h, target_w
    )
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    pad_width = ((pad_h0, pad_h1), (pad_w0, pad_w1)) + ((0, 0),) * (image.ndim - 2)
    return np.pad(resized, pad_width, mode="constant", constant_values=pad_value)


def scale_intrinsics_for_resize_with_pad(
    K: np.ndarray, h_src: int, w_src: int, h_out: int, w_out: int
) -> np.ndarray:
    """Adjust a 3x3 camera intrinsics matrix for :func:`resize_with_pad`.

    Unlike a plain non-uniform resize (independent fx/fy scale factors), a
    letterbox resize uses one uniform scale plus a constant pixel offset for
    the principal point (from the pad margins).
    """
    scale, _resized_h, _resized_w, pad_h0, _pad_h1, pad_w0, _pad_w1 = _resize_pad_geometry(
        h_src, w_src, h_out, w_out
    )
    K = K.copy()
    K[0, 0] *= scale  # fx
    K[1, 1] *= scale  # fy
    K[0, 2] = K[0, 2] * scale + pad_w0  # cx
    K[1, 2] = K[1, 2] * scale + pad_h0  # cy
    return K
