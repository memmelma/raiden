# YAM Arm Firmware Zero Calibration

This document records the diagnosis and fix for an issue where, on reset,
the right follower arm settled into a visibly higher physical pose than the
left follower arm (leader arms were unaffected and consistent). It also
documents how to re-run the same procedure in the future (e.g. if the issue
recurs, or on a different joint/arm).

## Background: two kinds of "zero"

There are two independent notions of "zero" for each arm joint:

1. **Software offset** (`motor_offset` in `get_yam_robot()` /
   `MotorChainRobot`): a per-session correction applied on top of the raw
   motor reading, computed at startup.
2. **Firmware zero**: a reference stored *inside* each DM motor's own
   firmware (set via the `save_zero_position` / "set current position as
   zero" command). This determines what raw position the motor reports as
   `0.0` before any software offset is applied.

Critically, `get_joint_pos()` always reads ~`0.0` at the commanded home
position *by construction* — home is defined as "joint position 0". This
means a firmware zero mismatch is **invisible** to normal joint-position
logging/diagnostics. It only shows up as a real, physical pose difference
you can see/feel on the hardware, not as a discrepancy in logged numbers.

This is why the initial hypothesis (a steady-state gravity-compensation /
gain asymmetry, fixable with a software home-position offset) was wrong: a
software offset was tried and confirmed **not** to fix the problem, because
the root cause was upstream of software entirely.

## Diagnosis procedure

The key insight: if you let an arm hang **torque-free** (motors enabled but
commanded with zero stiffness/effort) it falls under gravity to a pose
determined purely by its physical linkage and where gravity pulls it — this
pose is physically identical for both arms on a given joint (mirrored
left/right), *regardless of any firmware zero miscalibration*, for any joint
gravity actually acts on (shoulder/elbow/wrist-pitch; NOT the waist/yaw
joint, which gravity doesn't act on).

So: read the **raw** motor position (before any software offset correction)
for both arms while both are in this identical physical "fallen" pose. Raw
readings are a direct measure of each motor's firmware zero, so a large
raw right-vs-left difference on a given joint directly quantifies that
joint's firmware zero miscalibration, in radians.

This is implemented in `scripts/diag_fallen_pose.py`:

```bash
uv run scripts/diag_fallen_pose.py
```

It connects to both follower arms, reads raw (pre-offset) motor positions
while they're torque-free, and prints the raw right-vs-left delta per
joint, flagging any joint with `|delta| > 0.1 rad` as likely miscalibrated.

### What we found

Running this diagnostic with both arms resting in their natural fallen pose
showed a **~9.1° (0.158 rad) raw difference on joint 2** (shoulder/lift) of
the right follower arm (`can_follower_r`, motor id 2, a DM4340), with all
other joints matching to within noise (<0.3°). This exactly matches the
symptom: joint 2 controls arm lift, so a firmware zero that's off by 9° on
this joint makes the right arm's "home" pose sit visibly higher than the
left's.

## Fix: recalibrate the motor's firmware zero

`rd recalibrate` (implemented in `raiden/robot/recalibrate.py`, CLI wiring in
`raiden/cli.py`) fixes this by:

1. Connecting directly to the target motor(s) on their CAN channel (not the
   whole robot stack — other joints on the same channel are left alone
   unless explicitly included).
2. Smoothly ramping (under normal PD control, using the same gains as
   regular operation) from the motor's current raw position to a target raw
   position — either an explicit value you supply, or (by default) simply
   "stay where it currently is".
3. Holding briefly at the target to let it settle.
4. Sending the DM motor's `save_zero_position` firmware command, which
   permanently rebases that motor's firmware zero to wherever it physically
   is at that moment.
5. Reading back the position, which should now read ~`0.0`.

This is a **hardware calibration change** — it is not easily undone. Always
double check `--channel`, `--joints`, and `--targets` before running, and
prefer `--dry-run` first. By default `rd recalibrate` asks for interactive
`[y/N]` confirmation before touching hardware (unless `--dry-run` or `--yes`
is passed).

`--channel` accepts either a raw CAN channel (`can_follower_r`,
`can_follower_l`, `can_leader_r`, `can_leader_l`) or a friendly arm name
(`right_follower`, `left_follower`, `right_leader`, `left_leader`).

A standalone script wrapper (`scripts/recalibrate_joint_zero.py`, calling the
same underlying `raiden.robot.recalibrate.run_recalibrate`) is kept for
convenience, but `rd recalibrate` is the preferred entry point going
forward.

### What we actually ran

Left arm was used as the reference (assumed correctly calibrated, matching
factory/expected geometry). With the right arm resting in its natural
fallen pose (raw joint 2 position `-0.1574 rad`), the diagnostic showed the
left arm's equivalent physical pose corresponds to a raw reading of
`-0.1584 rad` on that motor. So the right arm's motor 2 needed to be told
"the position you're at right now, minus that tiny residual delta, is your
new zero":

```bash
rd recalibrate --channel can_follower_r --joints 2 --targets -0.1584
```

(The `--targets` value of `-0.1584` was derived from the raw right-vs-left
delta measured by `diag_fallen_pose.py`, applied to the right arm's
then-current raw reading, so that saving zero there makes the right arm's
zero match the left arm's physical zero.)

Verification with `diag_fallen_pose.py` afterwards showed joint 2's raw
right-vs-left delta dropped from `9.1°` to `0.1°`. A final `rd teleop` run
confirmed both arms now reset to matching heights.

### Re-running this procedure in the future

**If the arm is already resting in a known-good physical pose** (e.g. its
natural gravity-fallen pose, or any pose you've visually confirmed matches
the other arm), you can zero all 6 joints in place with no `--targets`
needed:

```bash
rd recalibrate --channel right_follower
```

(add `--joints 1,4,5` etc. to restrict to specific joints)

**If you need to correct a specific measured miscalibration** (arm not
currently in a good pose, but you know the raw delta from a diagnostic
like `diag_fallen_pose.py`), pass explicit target(s):

```bash
rd recalibrate --channel right_follower --joints 2 --targets -0.1584
```

Multiple joints can be recalibrated in one invocation; `--joints` and
`--targets` (if given) must have the same number of comma-separated
entries, matched by position:

```bash
rd recalibrate --channel left_follower --joints 1,2,3 --targets 0.01,-0.02,0.0
```

Pass `--yes` to skip the confirmation prompt (e.g. for scripted use), and
`--dry-run` to preview without touching hardware.

Always sanity-check with `--dry-run` first, and re-run
`scripts/diag_fallen_pose.py` (and/or `rd teleop`) afterwards to confirm the
fix.

## Related hardening (CAN handshake)

While investigating this issue, a separate (previously-fixed) class of bug
was also hardened: stale CAN frames left over from the `clean_error()` /
`motor_on()` handshake during robot startup could occasionally be
misinterpreted as valid initial position feedback, corrupting the absolute
position baseline for a session. This was fixed in
`third_party/i2rt/i2rt/motor_drivers/dm_driver.py` and
`third_party/i2rt/i2rt/motor_drivers/can_interface.py` by adding a
`drain_receive_queue()` helper (purges all queued CAN frames) used before
`motor_on()`/`clean_error()` handshakes, plus a confirmatory second-read
check on initial position. This is a distinct issue from the firmware zero
miscalibration described above, but was ruled out/addressed as part of the
same investigation. It does not require any recalibration and only affects
software robustness at startup.
