"""AntiPaSTO-ASVD: diagonal-covariance sibling of antipasto_corda.

Same frozen-basis bounded gain, but orients the SVD by the DIAGONAL of the input
second moment (per-channel activation scale) instead of the full covariance:

    M = diag(E[x_i^2])      vs   CorDA's full   C = E[x x^T]

This is Activation-aware SVD (Yuan+ 2023, arXiv:2312.05821): SVD(W diag(s)) with s a
per-channel scale. It is NOT a sub-basis of CorDA -- diag(C)^{1/2} and C^{1/2} are
different oblique rotations, so the top-r directions differ and either can win on a task.
ASVD is the cheap arm: O(d_in) moment, no d_in x d_in matrix, no eigh. The head-to-head
with antipasto_corda isolates whether the off-diagonal of C earns its init cost here.

Reuses antipasto_corda's buffers (U, S, P, g), plain-SVD init, gain forward, and the
shared `_covariance_orient` (only the diag flag differs), so there is one copy of the
math to keep in sync.

Refs: antipasto_corda.py (full-covariance sibling), ASVD arXiv:2312.05821.
"""
from dataclasses import dataclass

from ..variant import register
from ..config import register_config
from .antipasto_corda import AntiPaSTOCorDA, AntiPaSTOCorDAConfig, _covariance_orient


@register_config
@dataclass
class AntiPaSTOASVDConfig(AntiPaSTOCorDAConfig):
    variant: str = "antipasto_asvd"


@register
class AntiPaSTOASVD:
    name = "antipasto_asvd"
    param_specs = staticmethod(AntiPaSTOCorDA.param_specs)
    init = staticmethod(AntiPaSTOCorDA.init)
    forward = staticmethod(AntiPaSTOCorDA.forward)

    @staticmethod
    def group_init(model, targets, cfg, calibration_data) -> None:
        """ASVD: re-orient by the diagonal of the input second moment (per-channel)."""
        _covariance_orient(model, targets, cfg, calibration_data, diag=True)
