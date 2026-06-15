"""AntiPaSTO-DPLR: diagonal-plus-low-rank core in the frozen SVD basis.

antipasto's core is diagonal (a per-direction gain); it rescales each singular
direction but cannot mix one into another. The arrowhead tried a dense b x b block
on the top-b directions, but a dense block is the wrong shape (b^2 params, mixes only
the top-b) and -- sitting on the S-scaled coords -- its perturbation is amplified by
the largest singular values, so it destabilizes. The fix is LoRA's lesson: a low-rank
core. Put a trainable rank-k core inside the frozen U/Vh basis, ADDED to the gain:

    W = U diag(S) Vh + W_res                          # frozen top-r SVD
    learn: g (r,)                                      # diagonal gain
           A (k,r), B (r,k)                            # low-rank mixing core, B=0 at init
    S_eff = S * (1 + ELU(coeff * g))
    y = x @ W_res.T + ( (Vh x) * S_eff  +  coeff * B (A (Vh x)) ) @ U.T

so the trainable core is  C = diag(S_eff) + coeff * B A  acting in S-space, and
DeltaW = U C Vh. The diagonal part scales directions; the low-rank part B A mixes them
across the whole top-r subspace for 2*r*k params (k=LoRA's rank), not b^2.

Why the low-rank part is ADDED, not multiplied into diag(S): an additive core
U (BA) Vh is independent of S, so a unit step in BA moves W by O(1), not O(S). That is
exactly the S-amplification edge that made the dense arrowhead block blow up at the
gain's learning rate -- gone by construction.

Identity at init: B=0 -> BA=0, g=0 -> 1+ELU(0)=1, so C=diag(S) and DeltaW = U diag(S) Vh.
coeff=0 -> identity too (runtime off). The basis (U, Vh) stays frozen and interpretable;
only the gain and the rank-k core move.

Refs: antipasto.py (diagonal sibling), lora.py (the low-rank core), antipasto_corda.py
(oriented basis -- composes with this core).
"""
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import nn, Tensor as T

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config

CalibrationBatch = dict | tuple | list | T
CalibrationData = Iterable[CalibrationBatch]


@register_config
@dataclass
class AntiPaSTODPLRConfig(AdapterConfig):
    variant: str = "antipasto_dplr"
    r: int = 256
    # Rank of the low-rank mixing core (LoRA's r, but inside the frozen subspace).
    # Params = r (gain) + 2*r*lora_rank. k=0 degenerates to plain antipasto.
    lora_rank: int = 8
    suppress_only: bool = False  # clamp the gain g<=0 (attenuate only); core unaffected.
    coeff: float = 1.0           # runtime knob: 0=identity, scales gain and core.
    act_pool: Literal["rms", "mean_abs"] = "rms"  # group_init selection, see antipasto.


@register
class AntiPaSTODPLR:
    name = "antipasto_dplr"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        r, k = cfg.r, cfg.lora_rank
        if not 0 < k <= r:
            raise ValueError(f"antipasto_dplr needs 0 < lora_rank({k}) <= r({r}).")
        return dict(
            lora_U=ParamSpec((d_out, r), init="zeros", trainable=False, as_buffer=True),
            lora_S=ParamSpec((r,), init="zeros", trainable=False, as_buffer=True),
            lora_Vh=ParamSpec((r, d_in), init="zeros", trainable=False, as_buffer=True),
            # Diagonal gain (== antipasto). init 0 -> 1+ELU(0)=1 -> identity.
            lora_g=ParamSpec((r,), init="zeros"),
            # Low-rank core B@A in the frozen subspace. A down (r->k), B up (k->r).
            # B=0 at init -> core=0 -> identity (LoRA convention).
            lora_A=ParamSpec((k, r), init="kaiming"),
            lora_B=ParamSpec((r, k), init="zeros"),
        )

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        if type(layer) is not nn.Linear:
            raise TypeError("AntiPaSTODPLR mutates layer.weight into W_res; nn.Linear only.")
        with torch.no_grad():
            W = layer.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = cfg.r
            Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
            layer.lora_U.copy_(Ur.to(layer.lora_U.dtype))
            layer.lora_S.copy_(Sr.to(layer.lora_S.dtype))
            layer.lora_Vh.copy_(Vhr.to(layer.lora_Vh.dtype))
            W_res = (W - (Ur * Sr) @ Vhr).to(layer.weight.dtype)
            layer.weight.data.copy_(W_res)

    @staticmethod
    def group_init(model: nn.Module, targets, cfg, calibration_data: CalibrationData | None) -> None:
        """Wanda-style re-selection of the top-r directions, identical to antipasto.
        Runs before training while g and B are still zero, so the core contributes
        nothing and re-selecting the basis is a no-op on the adapter output."""
        if calibration_data is None:
            return

        layers = {name: layer for name, layer, _ in targets}
        captured: dict[str, list[T]] = {n: [] for n in layers}

        def make_hook(name):
            def _h(module, args, kwargs):
                x = args[0].detach()
                captured[name].append(rearrange(x, "... d -> (...) d").to(torch.float32).cpu())
            return _h

        handles = [layers[n].register_forward_pre_hook(make_hook(n), with_kwargs=True) for n in layers]
        try:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                for batch in calibration_data:
                    if isinstance(batch, dict):
                        model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        model(*batch)
                    else:
                        model(batch)
            if was_training:
                model.train()
        finally:
            for h in handles:
                h.remove()

        r, pool = cfg.r, cfg.act_pool
        for name, layer in layers.items():
            X = torch.cat(captured[name], dim=0)
            if X.shape[0] < r:
                raise RuntimeError(f"AntiPaSTODPLR at {name}: {X.shape[0]} tokens, need >= r={r}")
            # Rebuild the FULL W exactly (W_res + stored top-r), then re-select top-r.
            W_res = layer.weight.data.float()
            W_orig = W_res + (layer.lora_U.float() * layer.lora_S.float()) @ layer.lora_Vh.float()
            U_full, S_full, Vh_full = torch.linalg.svd(W_orig, full_matrices=False)
            proj = X.to(Vh_full) @ Vh_full.T
            act_mag = proj.pow(2).mean(0).sqrt() if pool == "rms" else proj.abs().mean(0)
            idx = (S_full * act_mag).argsort(descending=True)[:r].sort().values
            Ur, Sr, Vhr = U_full[:, idx], S_full[idx], Vh_full[idx]
            W_res_new = (W_orig - (Ur * Sr) @ Vhr).to(layer.weight.dtype)
            with torch.no_grad():
                layer.lora_U.copy_(Ur.to(layer.lora_U))
                layer.lora_S.copy_(Sr.to(layer.lora_S))
                layer.lora_Vh.copy_(Vhr.to(layer.lora_Vh))
                layer.weight.data.copy_(W_res_new)

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        U = layer.lora_U.to(x.dtype)                          # (d_out, r)
        S = layer.lora_S.to(x.dtype)                          # (r,)
        Vh = layer.lora_Vh.to(x.dtype)                        # (r, d_in)
        g = layer.lora_g.to(x.dtype)                          # (r,)
        A = layer.lora_A.to(x.dtype)                          # (k, r)
        B = layer.lora_B.to(x.dtype)                          # (r, k)
        coeff = float(cfg.coeff)

        if cfg.suppress_only:
            g = torch.clamp(g, max=0.0)

        p = x @ Vh.T                                          # (..., r) = Vh x (unscaled)
        S_eff = S * (1.0 + F.elu(coeff * g))                  # diagonal gain (see antipasto.py)
        # Diagonal part scales each direction; low-rank part B@A mixes across the
        # subspace. Additive (not * diag(S)), so the core is S-independent: a unit
        # step in B@A moves W by O(1), not O(S) -- no S-amplification edge.
        h = p * S_eff + coeff * (p @ A.T) @ B.T               # (..., r)
        return y + h @ U.T
