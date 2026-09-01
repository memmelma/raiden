"""Structured logging of inference rollouts for offline debugging.

``RolloutLogger`` accumulates per-step policy observations, robot states, and
predicted actions in memory during a ``rd infer`` rollout, then dumps
everything to a single ``.npz`` file (plus a small JSON sidecar with
metadata) under a timestamped directory in ``LOG_ROOT``.

Use ``scripts/visualize_rollout.py`` (repo root) to plot the result.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from chiral.types import Observation

LOG_ROOT = Path("/home/reward/Projects/logs")

DOF = 7  # joints per arm (6 revolute + 1 gripper)
JOINT_NAMES = [f"right_j{i}" for i in range(6)] + ["right_gripper"] + [
    f"left_j{i}" for i in range(6)
] + ["left_gripper"]


class RolloutLogger:
    """Accumulates and persists one rollout's worth of debug data.

    Call :meth:`log_step` once per control step, then :meth:`save` when the
    rollout ends (e.g. in a ``finally`` block).
    """

    def __init__(
        self,
        log_root: Path = LOG_ROOT,
        run_name: Optional[str] = None,
        source: str = "infer",
        extra_meta: Optional[dict] = None,
    ):
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_root) / self.run_name
        self._t0 = time.perf_counter()

        #: Number of control steps executed open-loop per policy query (e.g.
        #: ``OpenPiBridge.execute_len``). Set by the caller if known; used only
        #: as a fallback default for ``chunk_start`` markers and recorded in
        #: ``meta.json`` so ``visualize_rollout.py`` can default its tick
        #: spacing to it.
        self.chunk_size: Optional[int] = None

        #: "infer" or "replay" — lets visualize_rollout.py / the run list
        #: distinguish live-policy rollouts from ground-truth replays of
        #: recorded data, so the two can be compared side by side.
        self.source = source
        self.extra_meta: dict = extra_meta or {}

        self._timestamps: list[float] = []
        # Robot state (actual joint positions) and predicted/commanded actions,
        # raiden [right(7), left(7)] layout, i.e. shape (14,) per step.
        self._states: list[np.ndarray] = []
        self._actions_pred: list[np.ndarray] = []
        self._actions_cmd: list[np.ndarray] = []
        # True on steps where the bridge fetched a fresh action chunk from the
        # policy (vs. executing an already-buffered action open-loop).
        self._chunk_starts: list[bool] = []
        # Full policy observation proprios, keyed by proprio name -> list of arrays.
        self._obs_proprios: dict[str, list[np.ndarray]] = {}
        # One list of records per image "group" (e.g. "chunk", "live", "dataset"),
        # each record: {"step", "timestamp", "cameras": {name: relpath}}. Images
        # are written to disk immediately (not held in memory) since a full
        # rollout's worth of frames would be too large for the npz.
        self._image_records: dict[str, list[dict]] = {}

    def log_step(
        self,
        state_14d: np.ndarray,
        action_pred_14d: np.ndarray,
        action_cmd_14d: np.ndarray,
        obs: Optional[Observation] = None,
        chunk_start: bool = False,
    ) -> None:
        """Record one control step.

        Args:
            state_14d: Actual robot joint state at the time of this step,
                raiden ``[right(7), left(7)]`` layout.
            action_pred_14d: Predicted/commanded action before any safety
                clipping (for ``rd infer``) or simply the replayed trajectory
                value (for ``rd replay``, where there is no clipping).
            action_cmd_14d: Action actually sent to the motors, after
                safety clipping (equal to ``action_pred_14d`` when there is
                no clipping step, e.g. during replay).
            obs: Raw observation passed to ``bridge.predict()``, if any
                (``rd infer`` only — omit for ``rd replay``).
            chunk_start: True if this step just pulled a fresh action chunk
                from the policy (i.e. the start of a new action-chunk
                execution window). Not applicable to ``rd replay``.
        """
        self._timestamps.append(time.perf_counter() - self._t0)
        self._states.append(np.asarray(state_14d, dtype=np.float32).copy())
        self._actions_pred.append(np.asarray(action_pred_14d, dtype=np.float32).copy())
        self._actions_cmd.append(np.asarray(action_cmd_14d, dtype=np.float32).copy())
        self._chunk_starts.append(bool(chunk_start))

        if obs is not None:
            for name, val in obs.proprios.items():
                self._obs_proprios.setdefault(name, []).append(np.asarray(val).copy())

    def log_images(self, step: int, images: dict[str, np.ndarray], group: str = "chunk") -> None:
        """Save one snapshot per camera, grouped and indexed for later plotting.

        Images are written as PNGs under ``<run_dir>/<group>_images/``
        immediately (not held in memory — a full rollout's worth of frames
        would be too large for the npz), and indexed in
        ``meta.json[f'{group}_images']`` for ``visualize_rollout.py``.

        Args:
            step: Control-step index (used to name files / order frames).
            images: ``{camera_name: (H, W, 3) uint8 RGB array}``.
            group: Logical source of these images — e.g. ``"chunk"`` (policy
                observation at an action-chunk boundary, ``rd infer``),
                ``"live"`` (real robot camera, ``rd replay``), or
                ``"dataset"`` (recorded episode frame, ``rd replay``).
        """
        if not images:
            return
        try:
            import imageio.v3 as iio
        except ImportError:
            print("[log] imageio not installed — skipping image save.")
            return

        img_dir = self.run_dir / f"{group}_images"
        img_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self._timestamps[-1] if self._timestamps else time.perf_counter() - self._t0
        record: dict = {"step": int(step), "timestamp": float(timestamp), "cameras": {}}
        for name, image in images.items():
            fname = f"{step:06d}_{name}.png"
            iio.imwrite(str(img_dir / fname), np.asarray(image))
            record["cameras"][name] = f"{group}_images/{fname}"
        self._image_records.setdefault(group, []).append(record)

    def log_chunk_images(self, step: int, cameras: list) -> None:
        """Backward-compatible wrapper: log a policy observation's cameras at
        an action-chunk boundary (``rd infer``). See :meth:`log_images`.

        Args:
            step: Control-step index.
            cameras: ``obs.cameras`` — list of objects with ``.name`` and
                ``.image`` ``(H, W, 3)`` uint8 RGB attributes (e.g.
                ``chiral.types.CameraInfo``).
        """
        images = {cam.name: np.asarray(cam.image) for cam in (cameras or [])}
        self.log_images(step, images, group="chunk")

    def save(self) -> Optional[Path]:
        """Write accumulated data to ``<run_dir>/rollout.npz`` + ``meta.json``.

        Returns the path to the saved ``.npz`` file, or ``None`` if no steps
        were logged.
        """
        if not self._timestamps:
            return None

        self.run_dir.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "timestamps": np.array(self._timestamps, dtype=np.float64),
            "state": np.stack(self._states),
            "action_pred": np.stack(self._actions_pred),
            "action_cmd": np.stack(self._actions_cmd),
            "chunk_start": np.array(self._chunk_starts, dtype=bool),
        }
        for name, vals in self._obs_proprios.items():
            try:
                arrays[f"obs__{name}"] = np.stack(vals)
            except ValueError:
                # Ragged (e.g. varying length) — skip rather than fail the save.
                continue

        out_path = self.run_dir / "rollout.npz"
        np.savez(out_path, **arrays)

        meta = {
            "run_name": self.run_name,
            "source": self.source,
            "num_steps": len(self._timestamps),
            "joint_names": JOINT_NAMES,
            "obs_proprio_names": sorted(self._obs_proprios.keys()),
            "chunk_size": self.chunk_size,
            "chunk_images": self._image_records.get("chunk", []),
            "live_images": self._image_records.get("live", []),
            "dataset_images": self._image_records.get("dataset", []),
            **self.extra_meta,
        }
        with open(self.run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[log] saved {len(self._timestamps)} steps to {out_path}")
        return out_path
