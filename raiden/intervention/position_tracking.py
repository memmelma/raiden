"""Position-tracking intervention controller.

Detects operator interventions by watching the joint-position deviation
between the leader arms (handheld by the operator) and the follower arms
(driven by the policy). Crossing an upper threshold engages intervention;
returning below a lower threshold (with debounce) releases it.

This module is pure Python (numpy only) with no hardware dependencies, so
it's unit-testable in isolation and can be reused outside the residual-RL
bridge.

Design notes:

* **Hysteresis**: engage threshold > release threshold so the controller
  doesn't chatter at the boundary.
* **Debounce**: a leader spike for one tick from sensor noise shouldn't
  trigger intervention. Require N consecutive ticks above/below.
* **Bimanual single state**: per the user's choice we run one state
  machine for the whole robot — either arm crossing the engage threshold
  triggers whole-robot intervention. Transitions are then recorded with a
  single `is_intervention: bool` rather than per-arm flags.
* **Joint subset**: only the 6 arm joints per arm participate; grippers are
  ignored (operator can hold a gripper at any open/closed state without
  meaning to intervene).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np


class InterventionState(str, Enum):
    IDLE = "IDLE"               # follower driven by policy
    INTERVENING = "INTERVENING"  # follower mirrors leader


@dataclass
class InterventionConfig:
    """Tunable thresholds for the position-tracking switch.

    All thresholds are in radians of max-per-joint absolute deviation. The
    debounce ticks operate at the inference loop's control rate (default
    30 Hz → 3 ticks ≈ 100 ms).
    """

    thresh_engage: float = 0.15
    """Engage intervention when max |leader - follower| over arm joints
    exceeds this, sustained for `debounce_ticks` consecutive ticks."""

    thresh_release: float = 0.03
    """Release intervention when max |leader - follower| drops below this,
    sustained for `debounce_ticks` consecutive ticks. Must be < thresh_engage
    (the controller asserts this on construction)."""

    debounce_ticks: int = 3
    """Number of consecutive ticks that the deviation must stay
    above/below the corresponding threshold before changing state. Prevents
    sensor-noise-induced toggling."""

    arm_joint_indices: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    """Which joint indices on each arm to compare. Default is the 6 arm
    joints; the gripper (index 6 if present) is intentionally excluded."""


@dataclass
class PositionTrackingInterventionController:
    """Bimanual single-state-machine intervention controller.

    Usage::

        ctrl = PositionTrackingInterventionController(InterventionConfig())
        for tick in loop:
            is_intervening, leader_cmd = ctrl.step(leader_pos, follower_pos)
            if is_intervening:
                send_to_motors(leader_cmd)
            else:
                send_to_motors(policy_action)
    """

    cfg: InterventionConfig = field(default_factory=InterventionConfig)
    state: InterventionState = field(default=InterventionState.IDLE)
    engage_streak: int = field(default=0)
    release_streak: int = field(default=0)

    def __post_init__(self):
        if self.cfg.thresh_release >= self.cfg.thresh_engage:
            raise ValueError(
                f"thresh_release ({self.cfg.thresh_release}) must be strictly "
                f"less than thresh_engage ({self.cfg.thresh_engage}) so the "
                f"controller has hysteresis."
            )
        if self.cfg.debounce_ticks < 1:
            raise ValueError(
                f"debounce_ticks must be >= 1; got {self.cfg.debounce_ticks}"
            )

    @property
    def is_intervening(self) -> bool:
        return self.state == InterventionState.INTERVENING

    def _max_arm_deviation(
        self, leader_pos: np.ndarray, follower_pos: np.ndarray
    ) -> float:
        """Compute the max absolute per-arm-joint deviation across both
        arms. Assumes 14-DoF bimanual layout
        ``[r_joints(6), r_grip(1), l_joints(6), l_grip(1)]`` and applies
        ``cfg.arm_joint_indices`` within each arm."""
        if leader_pos.shape != (14,):
            raise ValueError(
                f"leader_pos must be shape (14,), got {leader_pos.shape}"
            )
        if follower_pos.shape != (14,):
            raise ValueError(
                f"follower_pos must be shape (14,), got {follower_pos.shape}"
            )
        idx = np.asarray(self.cfg.arm_joint_indices, dtype=np.int64)
        right_dev = np.max(np.abs(leader_pos[idx] - follower_pos[idx]))
        # Left arm starts at offset 7 (after right arm's 6 + 1 gripper).
        left_dev = np.max(np.abs(leader_pos[7 + idx] - follower_pos[7 + idx]))
        return float(max(right_dev, left_dev))

    def step(
        self,
        leader_pos: np.ndarray,
        follower_pos: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """Advance the state machine one tick.

        Returns ``(is_intervening, leader_cmd_for_motors)``. The second
        element is always ``leader_pos`` (the caller can use it directly
        as the motor command when intervening); it's returned for caller
        convenience so callers don't keep a separate reference.
        """
        dev = self._max_arm_deviation(leader_pos, follower_pos)

        if self.state == InterventionState.IDLE:
            if dev > self.cfg.thresh_engage:
                self.engage_streak += 1
                if self.engage_streak >= self.cfg.debounce_ticks:
                    self.state = InterventionState.INTERVENING
                    self.engage_streak = 0
                    self.release_streak = 0
            else:
                self.engage_streak = 0
        else:  # INTERVENING
            if dev < self.cfg.thresh_release:
                self.release_streak += 1
                if self.release_streak >= self.cfg.debounce_ticks:
                    self.state = InterventionState.IDLE
                    self.engage_streak = 0
                    self.release_streak = 0
            else:
                self.release_streak = 0

        return (self.is_intervening, leader_pos)

    def reset(self) -> None:
        """Reset to IDLE with zero streak counters. Call this between
        episodes so a residual deviation at episode end doesn't carry
        intervention state into the next episode."""
        self.state = InterventionState.IDLE
        self.engage_streak = 0
        self.release_streak = 0
