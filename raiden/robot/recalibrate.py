"""Recalibrate DM motor firmware zero position(s) on a YAM arm.

Background: each motor's own firmware stores a zero reference (independent of
any software `motor_offset`). If that reference is off by some amount, the
whole arm's "commanded pos=0" home will consistently land at the wrong
physical angle -- invisible to get_joint_pos() (which always reads ~0 at
home by construction) but visible as a real physical pose difference.

See CALIBRATE.md at the repo root for the full diagnosis writeup and
recommended workflow (in particular: use `diag_fallen_pose.py`-style raw
position comparisons to identify which joint, if any, needs recalibrating
before running this).

This module intentionally talks directly to `DMSingleMotorCanInterface`
rather than going through `RobotController` / `get_yam_robot()`: recalibration
must act on individual motors without the full-arm homing/offset machinery
getting in the way.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType

# PD gains and motor types matching the standard YAM arm config (see
# i2rt/robots/get_robot.py _YAM_HW), so the ramp feels the same as normal
# operation. Index by joint (1-based motor id - 1).
YAM_ARM_KP = [80.0, 80.0, 80.0, 40.0, 10.0, 10.0]
YAM_ARM_KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]
YAM_ARM_MOTOR_TYPES = ["DM4340", "DM4340", "DM4340", "DM4310", "DM4310", "DM4310"]

# Friendly arm name -> CAN channel, matching RobotController's canonical names.
ARM_CHANNELS = {
    "right_follower": "can_follower_r",
    "left_follower": "can_follower_l",
    "right_leader": "can_leader_r",
    "left_leader": "can_leader_l",
}


def parse_joints(joints_arg: str) -> List[int]:
    return [int(j.strip()) for j in joints_arg.split(",") if j.strip()]


def parse_targets(targets_arg: Optional[str], joints: List[int]) -> Dict[int, Optional[float]]:
    """Return a dict {motor_id: target_or_None}. None means "stay in place"."""
    if targets_arg is None:
        return {j: None for j in joints}
    parts = [t.strip() for t in targets_arg.split(",")]
    if len(parts) != len(joints):
        raise ValueError(
            f"--targets has {len(parts)} value(s) but --joints has {len(joints)}; "
            "they must match 1:1 (in the same order)."
        )
    return {j: float(t) for j, t in zip(joints, parts)}


def parse_motor_types(motor_types_arg: Optional[str], joints: List[int]) -> Dict[int, str]:
    """Return a dict {motor_id: motor_type_str}."""
    if motor_types_arg is None:
        return {j: YAM_ARM_MOTOR_TYPES[j - 1] if 1 <= j <= 6 else "DM4340" for j in joints}
    parts = [t.strip() for t in motor_types_arg.split(",")]
    if len(parts) == 1:
        return {j: parts[0] for j in joints}
    if len(parts) != len(joints):
        raise ValueError(
            f"--motor-types has {len(parts)} value(s) but --joints has {len(joints)}; "
            "provide either one value (applied to all) or one per joint."
        )
    return {j: t for j, t in zip(joints, parts)}


def resolve_channel(channel_or_arm: str) -> str:
    """Accept either a friendly arm name (right_follower, left_leader, ...) or a raw CAN channel."""
    return ARM_CHANNELS.get(channel_or_arm, channel_or_arm)


@dataclass
class RecalibrationPlan:
    joints: List[int]
    start_pos: Dict[int, float]
    targets: Dict[int, float]
    motor_types: Dict[int, str]
    kp: Dict[int, float]
    kd: Dict[int, float]


def run_recalibrate(
    channel: str,
    joints: str = "1,2,3,4,5,6",
    targets: Optional[str] = None,
    motor_types: Optional[str] = None,
    ramp_time: float = 2.5,
    settle_time: float = 1.0,
    dry_run: bool = False,
) -> None:
    """Recalibrate the firmware zero of one or more DM motors on a YAM arm.

    For each joint (default: all 6 arm joints), smoothly ramps that motor from
    its current raw position to a target raw position (default: stay in
    place), then sends the DM motor's "save current position as zero"
    firmware command. This is a hardware calibration change -- not easily
    undone.
    """
    resolved_channel = resolve_channel(channel)
    joint_ids = parse_joints(joints)
    target_map = parse_targets(targets, joint_ids)
    motor_type_map = parse_motor_types(motor_types, joint_ids)

    print("=" * 80)
    print(f"  RECALIBRATE JOINTS {joint_ids} ON {resolved_channel}")
    print("=" * 80)

    iface = DMSingleMotorCanInterface(channel=resolved_channel, bustype="socketcan", control_mode=ControlMode.MIT)
    try:
        start_pos: Dict[int, float] = {}
        motor_type_objs: Dict[int, str] = {}
        for j in joint_ids:
            motor_type_objs[j] = getattr(MotorType, motor_type_map[j], motor_type_map[j])
            info = iface.motor_on(j, motor_type_objs[j])
            start_pos[j] = info.position
            if target_map[j] is None:
                target_map[j] = info.position  # stay in place

        kp = {j: (YAM_ARM_KP[j - 1] if 1 <= j <= 6 else 10.0) for j in joint_ids}
        kd = {j: (YAM_ARM_KD[j - 1] if 1 <= j <= 6 else 1.0) for j in joint_ids}

        print()
        for j in joint_ids:
            delta = target_map[j] - start_pos[j]
            print(
                f"  Joint {j} ({motor_type_map[j]}): current={start_pos[j]:+.4f} rad, "
                f"target={target_map[j]:+.4f} rad, delta={delta:+.4f} rad ({delta * 180 / 3.14159:+.1f} deg), "
                f"kp={kp[j]}, kd={kd[j]}"
            )
        print(f"\nRamp over {ramp_time}s, settle {settle_time}s")

        if dry_run:
            print("\n--dry-run set: not moving or saving zero. Exiting.")
            return

        # Ramp all joints together so no single motor is left alone mid-transition.
        steps = max(int(ramp_time * 50), 1)
        last_info = {}
        for i in range(steps + 1):
            alpha = i / steps
            for j in joint_ids:
                target_pos = (1 - alpha) * start_pos[j] + alpha * target_map[j]
                last_info[j] = iface.set_control(j, motor_type_objs[j], target_pos, 0.0, kp[j], kd[j], 0.0)
            time.sleep(ramp_time / steps)
        print("\nReached target(s):")
        for j in joint_ids:
            print(f"  Joint {j}: actual position {last_info[j].position:.4f} rad")

        print(f"\nHolding for {settle_time}s before saving zero...")
        hold_steps = max(int(settle_time * 50), 1)
        for _ in range(hold_steps):
            for j in joint_ids:
                last_info[j] = iface.set_control(j, motor_type_objs[j], target_map[j], 0.0, kp[j], kd[j], 0.0)
            time.sleep(settle_time / hold_steps)
        print("Settled position(s) right before save:")
        for j in joint_ids:
            print(f"  Joint {j}: {last_info[j].position:.4f} rad")

        print("\nSaving current position as zero (firmware command) for each joint...")
        for j in joint_ids:
            iface.save_zero_position(j)
            print(f"  Joint {j}: zero saved.")

        print("\nVerifying (zero torque read):")
        for j in joint_ids:
            final = iface.set_control(j, motor_type_objs[j], 0.0, 0.0, 0.0, 0.0, 0.0)
            print(f"  Joint {j}: {final.position:.4f} rad (should read close to 0)")
        print(
            "\n(Joints just recalibrated are now torque-free at pos=0 command -- "
            "gravity-affected joints may sag.)"
        )

    finally:
        iface.close()
