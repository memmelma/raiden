"""Leader teaching-handle IO — single source of truth for the eval/inference path.

Both physical buttons on each YAM teaching handle show up as bits in the
encoder's ``io_inputs`` (read via ``YAMLeaderRobot.get_encoder_io``). The
gripper trigger shows up as the encoder ``position``. Three independent call
sites consume these — the eval verdict poller (``raiden/inference.py``), the
intervention bridges (``raiden/inference_bridges/*``), and the teleop recorder
(``raiden/robot/controller.py``) — so the mapping lives here, in one leaf module
(numpy-only, no hardware imports), to keep them from drifting apart and colliding
on the same bit.

Eval/inference button mapping:
  * TOP (yellow) button   → verdict: left leader = success, right leader = failure
                            (ends the episode and returns the arms home)
  * BOTTOM (white) button → intervention toggle on either leader

IMPORTANT: which physical button maps to which ``io_inputs`` bit is
HARDWARE-SPECIFIC. Confirm it with ``scripts/probe_leader_buttons.py`` (press
each button, watch which index flips 0 -> 1). The values below are the
codebase's prior assumption and are **not yet verified** on the present
handles — once probed, this is the only place that needs editing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# !!! VERIFY WITH scripts/probe_leader_buttons.py BEFORE TRUSTING ON HARDWARE !!!
TOP_BIT: int = 0
"""io_inputs bit for the TOP (yellow) button — success/failure verdict."""

BOTTOM_BIT: int = 1
"""io_inputs bit for the BOTTOM (white) button — intervention toggle."""

# Gripper slots in the 14-DoF bimanual command [r_arm(6), r_grip(1), l_arm(6), l_grip(1)].
RIGHT_GRIPPER_IDX: int = 6
LEFT_GRIPPER_IDX: int = 13


def apply_leader_grippers(
    cmd: np.ndarray,
    right_grip: Optional[float],
    left_grip: Optional[float],
) -> np.ndarray:
    """Fill the gripper slots of a 14-DoF command from the leader triggers.

    Returns a copy of ``cmd`` (layout ``[r_arm(6), r_grip(1), l_arm(6),
    l_grip(1)]``) with ``cmd[6]`` / ``cmd[13]`` set to the teaching-handle
    trigger values, clipped to ``[0, 1]`` (0=open, 1=closed) to match the
    follower gripper convention used by teleop. A ``None`` trigger leaves that
    slot untouched, so the caller's prior value (e.g. the last policy command)
    stands when a leader isn't present.
    """
    out = np.asarray(cmd, dtype=np.float32).copy()
    if right_grip is not None:
        out[RIGHT_GRIPPER_IDX] = float(np.clip(right_grip, 0.0, 1.0))
    if left_grip is not None:
        out[LEFT_GRIPPER_IDX] = float(np.clip(left_grip, 0.0, 1.0))
    return out
