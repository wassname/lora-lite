"""AntiPaSTO-CorDA: steer in a covariance-ORIENTED basis, not the weight-gain basis.

The complaint that motivates this: plain SVD sorts directions by weight gain ||W v||
on an *isotropic* input. The behaviour you steer lives where the *data* has energy.
Those orderings disagree, so the behaviour smears off the top singular axes and a
top-r crop in the weight basis throws it away. CorDA (Yang+ 2024, arXiv:2406.05223)
re-orients the decomposition by the input covariance C = E[x x^T], so the top
directions are the ones with the most energy *on real activations*.

Decomposition (verified: full-rank reconstruction ~1e-5, and on anisotropic data the
top-r data-truncation error drops ~27x vs plain SVD):

    C = E[x x^T] (+ eps I)              # input second moment on calibration data
    C^{1/2}, C^{-1/2}  via eigh(C)
    W~ = W C^{1/2};  SVD(W~) = U S V~h
    P  = V~h C^{-1/2}                   # (r, d_in) OBLIQUE input projector
    W  = U diag(S) P    (exactly)       # so y = x W_res^T + ((x P^T) * S_eff) U^T

S here are the singular values of W weighted by input std, so top-r is the optimal
rank-r in the input-weighted norm E||(W - W_r) x||^2 -- the directions that actually
move the output on your data.

Connection to the shared/differing-basis problem: C is built from pos AND neg inputs
pooled, so P spans the *shared* activation structure (the common encoder) that
chosen-minus-rejected cancels by construction. A trainable gain on this basis can
therefore reach shared structure that contrastive dS extraction is blind to.

Core: rotation-free. S_eff = S * (1 + ELU(coeff * g)). This is exp(coeff*g) on the
attenuation side (g<0, bounded, no blow-up) and 1+coeff*g on the amplification side
(g>0, where exp would diverge). g=0 -> identity. coeff is the runtime knob (0=off).

Basis note: P is OBLIQUE (rows not orthonormal -- C^{-1/2} skews them). That is fine
for gain reweighting (we scale oblique coordinates), and also fine for OUTPUT-side
directional ablation: the obliqueness is input-side only, while ablation acts in the
U/output space where U stays orthonormal. antipasto_ablate has a cov_orient flag that
reuses this basis -- at low r it captures the behavior output direction that plain-SVD
top-r drops (measured 1.00 vs 0.65 at r=16).

Falls back to plain SVD (== antipasto, rotation-free) if no calibration_data.
"""
from dataclasses import dataclass
from typing import Iterable

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
class AntiPaSTOCorDAConfig(AdapterConfig):
    variant: str = "antipasto_corda"
    r: int = 256
    cov_eps: float = 1e-3        # damping on C eigenvalues; guards C^{-1/2} on rare dirs
    coeff: float = 1.0           # runtime steer knob: 0=identity, scales trained g
    suppress_only: bool = False  # clamp g<=0 (attenuate only) -- for coeff>=0;
    #   coeff<0 inverts the product (coeff*g>=0) and re-amplifies.


def _gain(S: T, g: T, coeff: float, suppress_only: bool) -> T:
    """S_eff = S * (1 + ELU(coeff*g)); exp-bounded attenuation, linear amplification."""
    if suppress_only:
        g = g.clamp(max=0.0)
    return S * (1.0 + F.elu(coeff * g))


@register
class AntiPaSTOCorDA:
    name = "antipasto_corda"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        r = cfg.r
        return dict(
            lora_U=ParamSpec((d_out, r), init="zeros", trainable=False, as_buffer=True),
            lora_S=ParamSpec((r,), init="zeros", trainable=False, as_buffer=True),
            # P replaces Vh: oblique covariance-oriented input projector.
            lora_P=ParamSpec((r, d_in), init="zeros", trainable=False, as_buffer=True),
            # Trainable per-direction log-scale. init 0 -> 1+ELU(0)=1 -> exact identity.
            # No sign-symmetry hack needed (1+ELU is sign-preserving, basis frozen),
            # matching antipasto.py.
            lora_g=ParamSpec((r,), init="zeros"),
        )

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        """Plain-SVD fallback so the adapter is valid before group_init. group_init
        refines P/U/S to the covariance-oriented basis when calibration_data is given."""
        if type(layer) is not nn.Linear:
            raise TypeError("AntiPaSTOCorDA mutates layer.weight into W_res; nn.Linear only.")
        with torch.no_grad():
            W = layer.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = cfg.r
            Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
            layer.lora_U.copy_(Ur.to(layer.lora_U.dtype))
            layer.lora_S.copy_(Sr.to(layer.lora_S.dtype))
            layer.lora_P.copy_(Vhr.to(layer.lora_P.dtype))   # P := Vh until oriented
            W_res = (W - (Ur * Sr) @ Vhr).to(layer.weight.dtype)
            layer.weight.data.copy_(W_res)

    @staticmethod
    def group_init(model: nn.Module, targets, cfg, calibration_data: CalibrationData | None) -> None:
        """Re-orient each target's SVD by its input covariance C = E[x x^T].

        Without calibration_data the plain-SVD init from init() is kept (so this
        degrades to antipasto, rotation-free).

        Called by attach() BEFORE any training, so the trainable g is still at its
        zero init when the basis changes -- re-orienting zero gains is a no-op, no
        re-indexing needed. Do not call group_init after training has updated g."""
        if calibration_data is None:
            return

        layers = {name: layer for name, layer, _ in targets}
        # accumulate C = sum x x^T on CPU. Peak GPU cost would otherwise be
        # sum_targets d_in^2 fp32 held at once; for down_proj (d_in=intermediate,
        # e.g. 14336) that is ~0.8 GB *per layer* and OOMs. CPU accumulation bounds
        # GPU use to the live activation; the eigh/SVD below run on CPU (one-time).
        # Diagonal C is NOT a usable shortcut: it misses cross-channel correlation,
        # which is where the orientation gain lives (measured ~= plain SVD).
        # If down_proj's d_in^2 is too big even on CPU/RAM, exclude it from CorDA
        # (leave it on plain antipasto) or use a low-rank C (top-k eig of subsampled
        # inputs) -- not implemented here.
        cov: dict[str, T] = {}
        cnt: dict[str, int] = {n: 0 for n in layers}

        def make_hook(name):
            def _h(module, args, kwargs):
                x = rearrange(args[0].detach(), "... d -> (...) d").to(torch.float32).cpu()
                g = x.T @ x                                  # (d_in, d_in) on CPU
                cov[name] = g if name not in cov else cov[name] + g
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

        r, eps = cfg.r, float(cfg.cov_eps)
        for name, layer in layers.items():
            if cnt[name] < r:
                raise RuntimeError(f"AntiPaSTOCorDA at {name}: {cnt[name]} tokens, need >= r={r}")
            # decomposition on CPU (where C lives); copy results back to device buffers.
            W_res = layer.weight.data.float().cpu()
            U_old, S_old, P_old = (layer.lora_U.float().cpu(),
                                   layer.lora_S.float().cpu(),
                                   layer.lora_P.float().cpu())
            W_orig = W_res + (U_old * S_old) @ P_old

            C = cov[name] / cnt[name]                        # (d_in, d_in) CPU
            lam, Q = torch.linalg.eigh(C)
            lam = lam.clamp_min(0) + eps
            Chalf    = (Q * lam.sqrt())  @ Q.T               # C^{1/2}
            Cinvhalf = (Q * lam.rsqrt()) @ Q.T               # C^{-1/2}

            Ut, St, Vht = torch.linalg.svd(W_orig @ Chalf, full_matrices=False)
            Ur = Ut[:, :r]                                   # (d_out, r)
            Sr = St[:r]                                      # (r,)
            Pr = (Vht[:r] @ Cinvhalf)                        # (r, d_in) oblique projector
            W_res_new = (W_orig - (Ur * Sr) @ Pr)

            with torch.no_grad():
                layer.lora_U.copy_(Ur.to(layer.lora_U))
                layer.lora_S.copy_(Sr.to(layer.lora_S))
                layer.lora_P.copy_(Pr.to(layer.lora_P))
                layer.weight.data.copy_(W_res_new.to(layer.weight))

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        U = layer.lora_U.to(x.dtype)                         # (d_out, r)
        S = layer.lora_S.to(x.dtype)                         # (r,)
        P = layer.lora_P.to(x.dtype)                         # (r, d_in) oblique
        g = layer.lora_g.to(x.dtype)                         # (r,)
        S_eff = _gain(S, g, float(cfg.coeff), bool(cfg.suppress_only))
        h = (x @ P.T) * S_eff                                # (..., r)
        return y + h @ U.T
