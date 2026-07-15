# Dangerous / Random Resets on Init

This document records the diagnosis and fix for an issue where running
multiple Raiden commands sequentially (`rd teleop`, `rd record`, `rd replay`,
…) caused **dangerous, random-looking resets** on robot initialization —
wrist spinning, arm swinging into leaders, joints doing full rotations —
even when the arm started near home. Ctrl-C shutdown was always fine.
Power-cycling the YAMs sometimes cleared it (randomly). The issue appeared
after weeks of stable operation with **no code changes**.

## Symptoms (what operators saw)

- Happens on **init/reset** of a new process (`rd teleop` / `rd record` / …),
  never on graceful Ctrl-C exit.
- Severity varies: wrist twitch → full wrist spin → violent self-collision.
- Worse when the arm is left in a **gravity-loaded pose** (e.g. raised up).
- Appears "random" across runs; power cycle sometimes helps.
- `rd recalibrate` and firmware-zero checks did **not** fix it (different
  class of bug from `CALIBRATE.md`).

## Root cause (summary)

On every follower init, the arm was commanded with **zero stiffness**
(`pos=0, kp=0, kd=0`) for a window ranging from tens of ms to **several
seconds**, so joints free-floated under gravity (plus a stale torque
snapshot). Visible severity depended on pose (gravity torque) and
slowly-drifting mechanical factors (friction, wear, temperature) — which is
why it looked random and could appear weeks later with no code change.

There were **two stacked windows** of the same failure mode:

### 1. Dominant: gripper auto-calibration free-floats the whole arm (~4s)

Follower grippers are `LINEAR_4310`, which always need auto-calibration
(`get_gripper_limits()` returns `None`). On every
`MotorChainRobot.__init__`, `detect_gripper_limits()` (`i2rt/robots/utils.py`)
runs:

- For up to `test_duration` (2s) × 2 directions ≈ **4 seconds**,
- it called `motor_chain.set_commands(torques=…)` with **default**
  `pos=0, kp=0, kd=0` for **every** joint — not just the gripper —
- while only actively torque-driving the gripper.

Arm joints therefore sat torque-only (frozen effort snapshot from t=0 of
calibration) with zero PD for the entire calibration. That alone explains
wrist spin, raised-arm violence, "only on init", and "started later with
no code changes".

### 2. Secondary: control thread started before a hold command

`get_yam_robot()` constructed `DMChainCanInterface` with the default
`start_thread=True`, so the 250Hz CAN sender immediately began streaming
its default `starting_command` (`pos=0, kp=0, kd=0`) **before**
`MotorChainRobot` finished setup and could call `command_joint_pos` to hold
the measured pose. `MotorChainRobot.update()` already had a lazy-start
branch (`if not self.motor_chain.start_thread_flag: set_commands();
start_thread()`), but it was **dead code** because the thread was already
running.

Ctrl-C never hit this path because it does not reconstruct the robot
object.

## What was ruled out (earlier investigation)

- **Wrap-offset / ±2π heuristic** in `get_yam_robot()`: diagnostic table
  showed no wrap/mismatch on runs that still spun the wrist. Still kept
  as a safety gate (see below).
- **Firmware zero / recalibrate**: already covered by `CALIBRATE.md`; did
  not address free-float init.
- Hardcoding joint offsets: deliberately avoided while hunting root cause.

## Fixes (in `third_party/i2rt`)

All changes live in the i2rt submodule:

| File | Change |
|------|--------|
| `i2rt/robots/utils.py` | `detect_gripper_limits()` takes `hold_pos` / `hold_kp` / `hold_kd` and PD-holds every **non-gripper** joint at the measured pose during calibration; gripper stays torque-only. |
| `i2rt/robots/motor_chain_robot.py` | Reorder `__init__`: build `kp`/`kd`, start motor thread **only after** loading a hold command, then run gripper cal with hold params; re-read joint state after gripper remapper is set. |
| `i2rt/robots/get_robot.py` | Real robot `DMChainCanInterface` uses `start_thread=False`; print init wrap-offset diagnostics; abort before enabling if homing would jump `>1.0` rad (bypass: `RAIDEN_SKIP_INIT_SAFETY_CHECK=1`); confirmatory second connection for initial position read. |
| `i2rt/motor_drivers/can_interface.py` | `drain_receive_queue()` to discard stale CAN backlog before handshakes. |
| `i2rt/motor_drivers/dm_driver.py` | Call drain before / during `_motor_on` so enable responses aren’t stale frames. |

### Safe init ordering (after fix)

1. Open motor chain with `start_thread=False` (motors enabled, no stream yet).
2. In `MotorChainRobot.__init__`: read current pos → `set_commands` hold
   with real `kp`/`kd` → **then** `start_thread()`.
3. Gripper calibration (if needed): same hold for arm joints, torque only
   on gripper.
4. Set remapper / final `command_joint_pos` / start gravity-comp server
   thread (already holding, never starts from zero stiffness).

## Safety nets kept in place

- **Init diagnostics** (printed, not buried in `LOGLEVEL=ERROR`): per-joint
  raw pos, whether wrap would apply, implied home jump.
- **Abort gate**: if any **arm** joint (not gripper) would jump `>1.0` rad
  on home, raise before enabling motion. Gripper is excluded — its home
  is not raw 0 and goes through `JointMapper`.
- **CAN drain + second-connection confirmatory read** of initial
  positions (stale-frame contamination was a secondary hypothesis).

## How to verify

1. Power on YAMs; leave arms near home.
2. Run repeatedly: `rd teleop` → exit → `rd record` → exit → `rd teleop` …
3. Deliberately leave a follower arm **raised**, then start another
   command: reset should still hold pose (no free-fall / wrist spin /
   self-hit).
4. Watch for `[can_…] --- init wrap-offset diagnostics ---` and any
   `WARN` / abort messages.

If init aborts with jump >1 rad and you are sure the pose is intentional:

```bash
RAIDEN_SKIP_INIT_SAFETY_CHECK=1 rd teleop
```

## Related docs

- `CALIBRATE.md` — firmware zero asymmetry (different bug; physical pose
  mismatch with matching logged joint 0).
- This file — free-float / zero-stiffness windows on **process init**.
