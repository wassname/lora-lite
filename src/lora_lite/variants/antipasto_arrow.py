"""AntiPaSTO-Arrow: a STRUCTURED fixed-basis core, the cheap way to add cross-
direction mixing that plain antipasto (a diagonal gain) cannot express.

antipasto's core is diagonal: S_eff = S * (1 + ELU(coeff*g)) reweights each frozen
singular direction independently. It can turn a direction up or down but it can never
let direction i's input drive direction j's output. Yet the behaviour you steer is a
combination Sigma c_i v_i that generically lies OFF any single axis (the same argument
that motivates antipasto_corda), so a diagonal core can only ever approximate it.

The obvious fix -- a full dense r x r core M, DeltaW = U M Vh -- restores all mixing but
costs r^2 params (r=256 -> 65536, a rank-8 LoRA's worth) and an r x r matmul per forward.
antipasto.py's own header flags this trap: "a dense r x r core is r^2 params ... add a
*fixed-basis* core U M Vh rather than rotating". This file is that core, made cheap by
making it STRUCTURED instead of dense -- an arrowhead, not an r x r.

Arrowhead structure (dense top-block + diagonal tail):

    core C (r x r, acting on the S-scaled coords) =

        [ B (b x b dense) |        0          ]      B couples the top-b directions
        [        0        |  diag(1+ELU(c*g))  ]      tail = exactly antipasto's gain

    DeltaW = U @ C @ diag(S) @ Vh

The top b singular directions (largest S = where PiSSA says the action lives) get a full
b x b interaction block B = I_b + coeff*M; the remaining r-b stay on the cheap bounded
diagonal gain. Cost is b^2 + (r-b) params and one b x b matmul per forward -- for b=8,r=256
that is 312 params and a 64-MAC corner, versus 65536 for dense r x r and versus the
rotation variant's per-forward Cayley solve (measured 72ms vs 36ms). So: cross-direction
mixing where it matters, at diagonal-core cost.

(We call it "arrowhead" after the shape -- a dense head with a diagonal shaft. A true
numerical-LA arrowhead also carries a hub row+column coupling the block to the tail; that
would add 2(r-b) params and is a one-line extension if the top-b span turns out too small.
Not added until measured to be needed.)

Identity at init: M=0 -> B=I_b, g=0 -> 1+ELU(0)=1, so C=I and DeltaW = U diag(S) Vh exactly
(up to the one-time SVD-residual rounding). coeff=0 -> C=I too (runtime off). The block is
the linear-amplification regime of antipasto's design (a matmul, constant-gradient, no exp
self-amplification); it is stable like 1+ELU's upper branch, not strictly bounded -- if you
need the tail's structural can't-blow-up guarantee on the top directions too, use
antipasto_ablate instead.

Refs: antipasto.py (diagonal sibling), antipasto_corda.py (the off-axis argument).
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
class AntiPaSTOArrowConfig(AdapterConfig):
    variant: str = "antipasto_arrow"
    r: int = 256
    # Size of the dense interaction block on the top-b singular directions. The ONLY
    # quadratic cost (b^2 params); keep small. b=1 degenerates to antipasto.
    block: int = 8
    suppress_only: bool = False  # clamp the tail g<=0 (attenuate only); block unaffected.
    #   Tail guarantee holds for coeff>=0; coeff<0 inverts the product and re-amplifies.
    coeff: float = 1.0           # runtime knob: 0=identity, scales both block and tail
    act_pool: Literal["rms", "mean_abs"] = "rms"  # group_init selection, see antipasto


@register
class AntiPaSTOArrow:
    name = "antipasto_arrow"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        r, b = cfg.r, cfg.block
        if not 1 <= b < r:
            raise ValueError(f"antipasto_arrow needs 1 <= block({b}) < r({r}).")
        return dict(
            lora_U=ParamSpec((d_out, r), init="zeros", trainable=False, as_buffer=True),
            lora_S=ParamSpec((r,), init="zeros", trainable=False, as_buffer=True),
            lora_Vh=ParamSpec((r, d_in), init="zeros", trainable=False, as_buffer=True),
            # Dense b x b interaction on the top-b directions. init 0 -> B=I -> identity.
            lora_M=ParamSpec((b, b), init="zeros"),
            # Diagonal bounded gain on the remaining r-b directions (== antipasto's g).
            lora_g=ParamSpec((r - b,), init="zeros"),
        )

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        if type(layer) is not nn.Linear:
            raise TypeError("AntiPaSTOArrow mutates layer.weight into W_res; nn.Linear only.")
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
        """Wanda-style data-driven dimension selection, identical to antipasto: re-pick
        the top-r directions by S[i] * pool|X @ Vh[i]|. Runs before training (g, M at
        their zero init), so re-selecting the basis is a harmless no-op on the core."""
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
                raise RuntimeError(f"AntiPaSTOArrow at {name}: {X.shape[0]} tokens, need >= r={r}")
            W_res = layer.weight.data.float()
            W_orig = W_res + (layer.lora_U.float() * layer.lora_S.float()) @ layer.lora_Vh.float()
            U_full, S_full, Vh_full = torch.linalg.svd(W_orig, full_matrices=False)
            proj = X.to(Vh_full) @ Vh_full.T
            act_mag = proj.pow(2).mean(0).sqrt() if pool == "rms" else proj.abs().mean(0)
            # Select top-r by score, then re-sort ascending by SVD index. Since svd()
            # returns S descending, the first b stored dirs (the block's cS[..., :b]) are
            # the b LARGEST-S among the selected r -- not the b highest-score. Matches the
            # block's "largest S = where the action lives" intent, but a high-S dir dropped
            # by score-selection won't be in the block.
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
        M = layer.lora_M.to(x.dtype)                          # (b, b)
        g = layer.lora_g.to(x.dtype)                          # (r-b,)
        coeff, b = float(cfg.coeff), cfg.block

        cS = (x @ Vh.T) * S                                   # (..., r) = diag(S) Vh x

        # Top-b: dense block B = I_b + coeff*M couples the top singular directions.
        eye = torch.eye(b, dtype=x.dtype, device=x.device)
        top = cS[..., :b] @ (eye + coeff * M).T               # (..., b)
        # Tail: antipasto's bounded diagonal gain (see antipasto.py for the 1+ELU why).
        if cfg.suppress_only:
            g = torch.clamp(g, max=0.0)
        tail = cS[..., b:] * (1.0 + F.elu(coeff * g))         # (..., r-b)

        h = torch.cat([top, tail], dim=-1)                    # (..., r)
        return y + h @ U.T
