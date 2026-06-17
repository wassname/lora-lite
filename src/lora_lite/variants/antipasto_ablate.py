"""AntiPaSTO-Ablate: trainable directional ablation in the weight-SVD output basis.

A contractive sibling of antipasto.py: instead of reweighting the singular gains it
projects out a learned direction in the output (U-side) singular basis.

    W = U diag(S) Vh + W_res
    learn:  c (r, k) ablation directions,  alpha (k,) strengths in [0, 1]
    Chat  = orthonormal(c)                            # k unit dirs in S-space
    h     = (x @ Vh.T) * S                            # output S-coords = diag(S) Vh x
    h    <- h - coeff * (h @ Chat) * alpha @ Chat.T   # project the span out
    y     = x @ W_res.T + h @ U.T

The core (I - alpha Chat Chat^T) is a contraction: eigenvalues 1-alpha along Chat,
1 elsewhere, all in [0, 1]. It cannot amplify, so it cannot blow up -- the instability
the multiplicative gain bounds away is structurally absent (and a contraction is the
natural core to recurse). This is the trainable form of directional ablation (Arditi+
2024): target residual writers (down_proj, o_proj) for the surgical regime, not all
Linears.

Runtime: coeff is the per-call knob. coeff=0 -> identity; (0, 1] -> ablate; <0 adds the
direction back (the side that can grow, so bound coeff there).

Refs: antipasto.py (gain sibling), directional ablation Arditi+ 2024 arXiv:2406.11717.
"""
from dataclasses import dataclass
from typing import Iterable

import torch
from einops import rearrange
from jaxtyping import Float
from torch import nn, Tensor as T

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config

CalibrationBatch = dict | tuple | list | T
CalibrationData = Iterable[CalibrationBatch]

ε = 1e-6


@register_config
@dataclass
class AntiPaSTOAblateConfig(AdapterConfig):
    variant: str = "antipasto_ablate"
    r: int = 256          # top-r SVD captured (or |dS|-selected via group_init)
    k: int = 1            # number of ablation directions (rank of the projection)
    init_alpha: float = 0.05  # small >0 so c gets gradient at step 0
    coeff: float = 1.0    # runtime: 0=identity, (0,1]=ablate, <0=amplify (bound this side)
    # CorDA-orient the basis from input covariance (group_init, needs calibration_data).
    # The ablation is OUTPUT-side and CorDA's U stays orthonormal, so this is a clean
    # contraction; the win is at low r -- the data-oriented top-r captures the behavior
    # output direction that plain-SVD top-r drops (measured 1.00 vs 0.65 at r=16).
    cov_orient: bool = False
    cov_eps: float = 1e-3


@register
class AntiPaSTOAblate:
    name = "antipasto_ablate"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        r, k = cfg.r, cfg.k
        return dict(
            lora_U=ParamSpec((d_out, r), init="zeros", trainable=False, as_buffer=True),
            lora_S=ParamSpec((r,), init="zeros", trainable=False, as_buffer=True),
            lora_Vh=ParamSpec((r, d_in), init="zeros", trainable=False, as_buffer=True),
            # Trainable: k ablation directions in S-space, and their strengths.
            lora_c=ParamSpec((r, k), init=lambda t: t.normal_(0, 1.0 / max(r, 1) ** 0.5)),
            lora_alpha=ParamSpec((k,), init=lambda t: t.fill_(float(cfg.init_alpha))),
        )

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        if type(layer) is not nn.Linear:
            raise TypeError("AntiPaSTOAblate mutates layer.weight into W_res; nn.Linear only.")
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
            # lora_c starts random here; group_init warm-starts it from the S-space output
            # variance when calibration_data is given (see group_init), else it trains from noise.

    @staticmethod
    def group_init(model: nn.Module, targets, cfg, calibration_data: CalibrationData | None) -> None:
        """Warm-start each lora_c from calibration activations (and, if cov_orient,
        re-orient the frozen SVD by input covariance C=E[x xᵀ] first, CorDA-style).

        lora_c is seeded to the top-k principal axes of the S-space OUTPUT coords
        h = diag(S) Vh x over the calibration set: the highest-energy output directions,
        where the loss-gradient on the ablation strength is largest, so lora_c starts in a
        high-gradient region instead of a near-orthogonal random one. NOTE this is the data
        VARIANCE direction, not a contrastive behavior direction -- this benchmark is SFT
        with no pos/neg split. For steering with contrastive pairs, seed lora_c from
        mean(h|pos) - mean(h|neg) instead (cf. steering-lite sspace extract).

        Σ xxᵀ (d_in², heavy for down_proj) is only accumulated to orient; the warm-start
        alone (cov_orient=False) needs just the cheap r×r second moment Σ hhᵀ."""
        if calibration_data is None:
            return
        orient = bool(getattr(cfg, "cov_orient", False))
        layers = {name: layer for name, layer, _ in targets}
        gram: dict[str, T] = {}   # Σ xxᵀ (d_in²), only when orienting
        mom: dict[str, T] = {}    # Σ hhᵀ (r²), when not orienting (basis is fixed at init)
        cnt: dict[str, int] = {n: 0 for n in layers}

        def make_hook(name):
            layer = layers[name]
            def _h(module, args, kwargs):
                x = rearrange(args[0].detach(), "... d -> (...) d").to(torch.float32).cpu()
                if orient:
                    g = x.T @ x
                    gram[name] = g if name not in gram else gram[name] + g
                else:
                    h = (x @ layer.lora_Vh.float().cpu().T) * layer.lora_S.float().cpu()
                    m = h.T @ h
                    mom[name] = m if name not in mom else mom[name] + m
                cnt[name] += x.shape[0]
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

        r, k, eps = cfg.r, cfg.k, float(cfg.cov_eps)
        for name, layer in layers.items():
            if cnt[name] < r:
                raise RuntimeError(f"AntiPaSTOAblate at {name}: {cnt[name]} tokens, need >= r={r}")
            if orient:
                W_res = layer.weight.data.float().cpu()
                U_old, S_old, Vh_old = (layer.lora_U.float().cpu(),
                                        layer.lora_S.float().cpu(),
                                        layer.lora_Vh.float().cpu())
                W_orig = W_res + (U_old * S_old) @ Vh_old

                C = gram[name] / cnt[name]
                lam, Q = torch.linalg.eigh(C)
                lam = lam.clamp_min(0) + eps
                Chalf    = (Q * lam.sqrt())  @ Q.T
                Cinvhalf = (Q * lam.rsqrt()) @ Q.T
                Ut, St, Vht = torch.linalg.svd(W_orig @ Chalf, full_matrices=False)
                Ur = Ut[:, :r]                          # orthonormal output basis (ablation acts here)
                Sr = St[:r]
                Pr = Vht[:r] @ Cinvhalf                 # oblique input projector (input-side only)
                W_res_new = W_orig - (Ur * Sr) @ Pr
                with torch.no_grad():
                    layer.lora_U.copy_(Ur.to(layer.lora_U))
                    layer.lora_S.copy_(Sr.to(layer.lora_S))
                    layer.lora_Vh.copy_(Pr.to(layer.lora_Vh))    # store P in the Vh slot
                    layer.weight.data.copy_(W_res_new.to(layer.weight))
                # output S-space second moment in the (now oriented) basis: diag(S) P Σxxᵀ Pᵀ diag(S)
                SP = Sr[:, None] * Pr
                M = SP @ gram[name] @ SP.T
            else:
                M = mom[name]                                    # (r, r) Σ hhᵀ in the init basis

            c0 = torch.linalg.eigh(M).eigenvectors[:, -k:]       # top-k principal dirs (orthonormal)
            with torch.no_grad():
                layer.lora_c.copy_(c0.to(layer.lora_c))

    @staticmethod
    def _orthonormal(c: T) -> T:
        """(r, k) -> (r, k) with orthonormal columns. k=1 is a plain normalize."""
        if c.shape[-1] == 1:
            return c / (c.norm(dim=0, keepdim=True) + ε)
        # geqrf has no bf16/fp16 kernel (CPU or CUDA); do the QR in fp32, cast back.
        q, _ = torch.linalg.qr(c.float())   # reduced QR; columns orthonormal
        return q.to(c.dtype)

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        U = layer.lora_U.to(x.dtype)                  # (d_out, r)
        S = layer.lora_S.to(x.dtype)                  # (r,)
        Vh = layer.lora_Vh.to(x.dtype)                # (r, d_in)
        Chat = AntiPaSTOAblate._orthonormal(layer.lora_c.to(x.dtype))   # (r, k)
        alpha = layer.lora_alpha.to(x.dtype).clamp(0.0, 1.0)            # (k,)
        coeff = float(cfg.coeff)

        h = (x @ Vh.T) * S                            # (..., r) output S-coords
        proj = h @ Chat                               # (..., k) component along each dir
        # contractive removal: h <- h - coeff * Sum_j alpha_j (h . chat_j) chat_j
        h = h - coeff * (proj * alpha) @ Chat.T       # (..., r)
        return y + h @ U.T
