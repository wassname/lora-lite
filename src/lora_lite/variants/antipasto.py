"""AntiPaSTO: learnable bounded reweighting of frozen SVD singular values.

wassname 2026  https://arxiv.org/abs/2601.07473

    W = U diag(S) Vh + W_res           # top-r SVD; W_res = W - U_r S_r Vh_r, frozen
    learn: g (r,)                      # per-direction gain
    S_eff = S * (1 + ELU(coeff * g))   # exp(z) for z<0 (bounded), 1+z for z>0
    y = x @ W_res.T + ((x @ Vh.T) * S_eff) @ U.T

    suppress_only: clamp g<=0 -> S_eff in (0, S], attenuation only.
    coeff:         runtime scale; 0 = identity, <0 swaps amplify/suppress.

Identity at g=0 or coeff=0: 1+ELU(0)=1, so S_eff=S (up to the bf16 SVD round-trip).
The basis (U, Vh) is frozen, so the singular directions stay interpretable and only
the gain is learned. See forward() for why 1+ELU over linear/exp/tanh.

Refs:
  - paper: https://github.com/wassname/AntiPaSTO
  - sibling (whitened, mean-diff): steering-lite/.../sspace.py
  - selection: Wanda (Sun+ 2023, arXiv:2306.11695), ASVD (Yuan+ 2023, arXiv:2312.05821)
  - top-r SVD init: PiSSA (Meng+ 2024, arXiv:2404.02948)
"""
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from einops import rearrange
from jaxtyping import Float
from torch import nn, Tensor as T

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config

CalibrationBatch = dict | tuple | list | T
CalibrationData = Iterable[CalibrationBatch]


@register_config
@dataclass
class AntiPaSTOConfig(AdapterConfig):
    variant: str = "antipasto"
    # Only r + r trainable scalars, so r can be large.
    r: int = 256
    # Per-direction reweighting is S_eff = S * (1 + ELU(coeff * g)). See forward()
    # for the why; identity at g=0 or coeff=0, positive always, no free bound knob.
    suppress_only: bool = False  # clamp g<=0 -> factor in (0,1]: attenuation only.
    #   Guarantee holds for coeff>=0; coeff<0 inverts the product and re-amplifies.
    # Runtime steering scale. 0 = identity. <0 inverts (swaps amplify/suppress).
    coeff: float = 1.0
    # group_init Wanda-style pooling of |X @ Vh[i]|: 'rms' is outlier-sensitive
    # (ASVD intuition), 'mean_abs' is the original outlier-robust pooling.
    act_pool: Literal["rms", "mean_abs"] = "rms"


@register
class AntiPaSTO:
    name = "antipasto"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        r = cfg.r
        return dict(
            # Frozen top-r SVD captured at init.
            lora_U=ParamSpec((d_out, r), init="zeros", trainable=False, as_buffer=True),
            lora_S=ParamSpec((r,), init="zeros", trainable=False, as_buffer=True),
            lora_Vh=ParamSpec((r, d_in), init="zeros", trainable=False, as_buffer=True),
            # Trainable per-direction log-scale. init 0 -> 1+ELU(0)=1 -> identity.
            lora_g=ParamSpec((r,), init="zeros"),
        )

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        if type(layer) is not nn.Linear:
            raise TypeError(
                "AntiPaSTO mutates layer.weight into W_res (like PiSSA), so v1 "
                "only supports plain nn.Linear, not bnb 4/8-bit."
            )
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
            # group_init() refines the dimension selection if calibration_data is given.

    @staticmethod
    def group_init(model: nn.Module, targets, cfg, calibration_data: CalibrationData | None) -> None:
        """Data-driven re-selection of which top-r singular directions to keep.

            init():       top-r by S alone (PiSSA-style)
            group_init(): top-r by score[i] = S[i] * pool|X @ Vh[i]|   (Wanda/ASVD)
            pool = 'rms' (outlier-sensitive) | 'mean_abs' (outlier-robust)

        This re-RANKS W's own singular vectors by activation; it does NOT re-orient
        the basis (that is CorDA -> antipasto_corda.py). So the kept directions are
        still plain weight-SVD directions, just a better subset. None -> keep init().
        """
        if calibration_data is None:
            return

        layers = {name: layer for name, layer, _ in targets}
        captured: dict[str, list[T]] = {n: [] for n in layers}

        def make_hook(name):
            def _h(module, args, kwargs):
                x = args[0].detach()
                captured[name].append(rearrange(x, "... d -> (...) d").to(torch.float32).cpu())
            return _h

        handles = [
            layers[n].register_forward_pre_hook(make_hook(n), with_kwargs=True)
            for n in layers
        ]
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

        r = cfg.r
        pool = cfg.act_pool
        for name, layer in layers.items():
            X = torch.cat(captured[name], dim=0)          # (N, d_in)
            if X.shape[0] < r:
                raise RuntimeError(
                    f"AntiPaSTO at {name}: only {X.shape[0]} calibration tokens, need >= r={r}"
                )

            # Rebuild the FULL W: init() stored the exact top-r it subtracted, so
            # W_res + U_r S_r Vh_r == W (full rank, not a cropped matrix). The SVD
            # below therefore re-selects from W's whole spectrum, not a truncation.
            W_res = layer.weight.data.float()
            U_old = layer.lora_U.float()
            S_old = layer.lora_S.float()
            Vh_old = layer.lora_Vh.float()
            W_orig = W_res + (U_old * S_old.unsqueeze(0)) @ Vh_old

            U_full, S_full, Vh_full = torch.linalg.svd(W_orig, full_matrices=False)
            proj = X.to(Vh_full) @ Vh_full.T              # (N, r) input in S-coords (X CPU -> GPU here)
            if pool == "rms":
                act_mag = proj.pow(2).mean(dim=0).sqrt()  # outlier-sensitive
            else:
                act_mag = proj.abs().mean(dim=0)          # outlier-robust (original)
            scores = S_full * act_mag
            idx = scores.argsort(descending=True)[:r]     # top-r by joint importance
            idx = idx.sort().values                       # stable ordering

            Ur, Sr, Vhr = U_full[:, idx], S_full[idx], Vh_full[idx]
            W_res_new = (W_orig - (Ur * Sr.unsqueeze(0)) @ Vhr).to(layer.weight.dtype)

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
        coeff = float(cfg.coeff)

        if cfg.suppress_only:
            g = torch.clamp(g, max=0.0)                       # factor in (0,1]: attenuation only

        # S_eff = S * (1 + ELU(z)),  z = coeff*g,  1+ELU(z) = exp(z) for z<=0 else 1+z.
        # Why 1+ELU and not the obvious alternatives:
        #   linear S*(1+z)  : z<-1 -> S_eff<0, a sign flip that drives incoherence.
        #   exp    S*exp(z) : unbounded, gradient self-amplifies (amplification blows up).
        #   tanh   bounded  : arbitrary bound knob, saturation kills the gradient.
        # 1+ELU uses each in its safe regime: exp where it is bounded in (0,1]
        # (attenuation), linear where exp would diverge (amplification). >0 always.
        S_eff = S * (1.0 + torch.nn.functional.elu(coeff * g))

        h = (x @ Vh.T) * S_eff                               # input in S-coords, reweighted
        return y + h @ U.T
