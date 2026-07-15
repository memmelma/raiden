"""Concrete ModelBridge implementations for raiden inference.

Each bridge owns its own model-specific logic — preprocessing, inference,
postprocessing — and plugs into ``raiden.inference.RaidenInferenceLoop``
via the abstract ``raiden.inference.ModelBridge`` contract.

Available bridges:

- :class:`OpenPiBridge` — closed-loop eval against an openpi-style WebSocket
  policy server, with leader-follower position-tracking interventions.
- :class:`OpenPiEEBridge` — the same server, but for 20-D end-effector action
  heads (no arm swap; emits relative EE poses for on-the-fly IK).

Note: the residual-RL ``ResidualBridge`` from raiden_oldv1 is intentionally
not ported here — it depends on the ``alder`` (hil-serl) package, which is
outside the intervention-core scope.
"""

from raiden.inference_bridges.openpi_bridge import OpenPiBridge
from raiden.inference_bridges.openpi_ee_bridge import OpenPiEEBridge

__all__ = ["OpenPiBridge", "OpenPiEEBridge"]
