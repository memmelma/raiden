"""End-effector (20-D) policy bridge for openpi YAM models.

:class:`OpenPiBridge` serves **14-D joint** actions and reorders them from
openpi's left-first ``[L(7), R(7)]`` to raiden's right-first ``[R(7), L(7)]``.
That arm-block swap is WRONG for an end-effector action head, whose 20-D layout
is *field-interleaved*::

    [l_xyz(3), r_xyz(3), l_rot6d(6), r_rot6d(6), l_grip(1), r_grip(1)]

So this bridge passes the action through in openpi order — exactly what
``RaidenPolicyServer._ee_pose_to_joint_cmd`` expects (it parses ``action[0:3]``
as ``l_xyz`` etc. and IK-solves). The poses are RELATIVE to the EE pose at
chunk-query time (shardify's ``*_relative`` encoding); the inference loop
reconstructs absolute IK targets by composing each action with the anchor it
captures when :attr:`just_refilled` flips True (once per chunk).

Everything else — websocket client, request building, image/state encoding,
chunk refill — is inherited from :class:`OpenPiBridge`. Leader interventions are
not supported here (the 14-D leader command path doesn't apply to a 20-D EE
action); use the joint bridge if you need interventions.

Usage::

    rd infer \\
        --bridge raiden.inference_bridges.openpi_ee_bridge:OpenPiEEBridge \\
        --ckpt-path unused \\
        --action-type ee_pose \\
        --bridge-kwargs host=localhost port=8000 prompt='...'
"""
from __future__ import annotations

import numpy as np

from raiden.inference_bridges.openpi_bridge import OpenPiBridge


class OpenPiEEBridge(OpenPiBridge):
    """20-D end-effector variant of :class:`OpenPiBridge`.

    Reuses the parent's connection, request builder, and chunk refill; only
    :meth:`predict` differs: the action is returned unswapped (the EE layout is
    field-interleaved, not arm-blocked), and :attr:`just_refilled` signals the
    inference loop to re-capture the per-chunk relative-pose anchor.
    """

    #: True only on the step where a fresh chunk was just fetched. The loop reads
    #: this to re-capture the relative-pose anchor once per chunk.
    just_refilled: bool = False

    def predict(self, obs) -> np.ndarray:
        assert self._client is not None, "load() must run before predict()"
        self.just_refilled = False
        if not self._chunk_queue:
            self._refill_chunk(obs)
            self.just_refilled = True
        # 20-D EE action in openpi order [l_xyz, r_xyz, l_rot6d, r_rot6d,
        # l_grip, r_grip] — passed through unswapped; _ee_pose_to_joint_cmd
        # parses this layout directly.
        action = np.asarray(self._chunk_queue.popleft(), dtype=np.float32)
        if self.just_refilled and action.shape[0] >= 20:
            # Dump the first (absolute) action of each chunk so the raw EE-pose
            # scale is visible: l_xyz/r_xyz are meters in each arm's base frame.
            print(
                f"[OpenPiEEBridge] raw action[0]  "
                f"l_xyz={action[0:3].round(4)} r_xyz={action[3:6].round(4)}  "
                f"|l_xyz|={float(np.linalg.norm(action[0:3])):.3f}m "
                f"|r_xyz|={float(np.linalg.norm(action[3:6])):.3f}m  "
                f"l_rot6d={action[6:12].round(3)} r_rot6d={action[12:18].round(3)}  "
                f"grip=[{action[18]:.2f}, {action[19]:.2f}]"
            )
        self._last_action = action
        return action
