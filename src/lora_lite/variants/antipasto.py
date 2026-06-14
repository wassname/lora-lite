"""AntiPaSTO: SVD steering with learnable, bounded singular-value reweighting.

wassname 2026  https://arxiv.org/abs/2601.07473

    W = U diag(S) Vh + W_res         (top-r SVD; W_res = W - U_r S_r Vh_r)
    learn: g (r,)                    per-singular-direction gain log/lin-scale
    S_eff = S * (1 + ELU(coeff * g))    exp(.) for g<=0, 1+. for g>0
        suppress_only:  clamp g<=0   -> factor in (0,1], attenuation only
    y = x @ W_res.T + ((x @ Vh.T) * S_eff) @ U.T

Identity at g=0 (or coeff=0): 1+ELU(0)=1 exactly, so S_eff = S and the output is
x @ W^T up to the one-time SVD-residual rounding. No additive sign-symmetry hack
needed: the basis is frozen, so the direction sign is fixed and exp/(1+.) is
sign-preserving. The 1+ELU shape is chosen over linear (sign-flips at g<-1), exp
(amplification blows up), and tanh (arbitrary bound) -- see forward() for why.

Changes vs the rotation version this replaces:
  - Rotation dropped. Rotating Vh/U leaves the interpretable singular basis (the
    SVD-direction / Conjecture property), which is the entire point of steering in
    S-space, and the Cayley solve was numerically finicky. The basis is now frozen;
    the only learned object is the per-direction gain. If you later want
    cross-direction mixing, add a *fixed-basis* core U M Vh (M trainable, U/Vh frozen)
    rather than rotating -- that keeps the directions interpretable. It is also far
    cheaper than PiSSA: a dense r x r core is r^2 params (~= a rank-8 LoRA at r=256),
    versus PiSSA's free A,B at r*(d_in+d_out), which drifts off the SVD basis.
  - Additive delta_s -> bounded multiplicative S * (1 + ELU(coeff*g)). Multiplicative
    is "scaled by S" (uniform *relative* control over an orders-of-magnitude spectrum),
    stays positive (no S_eff<0 sign-flip -> no incoherence from that path), and the
    1+ELU shape stops the exp blowup. The 4e-4 sign-symmetry hack is gone.
  - suppress_only = clamp g<=0 -> factor in (0,1]: attenuation only, structurally
    cannot blow up. Matches the eval-awareness use case (turn a direction down).
  - coeff: runtime steering scalar (0 = identity, <0 inverts). The per-call alpha
    the rotation version lacked.
  - group_init activation pooling is configurable: 'rms' weights outliers (ASVD
    intuition), 'mean_abs' is the original outlier-robust pooling.

Refs:
  - paper: https://github.com/wassname/AntiPaSTO
  - sibling (whitened, rotation-free, mean-diff): steering-lite/.../sspace.py
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
    suppress_only: bool = False  # clamp g<=0 -> factor in (0,1]: attenuation only
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
        """Wanda-style, data-driven dimension selection within the weight SVD.

        init() picks the top-r singular dimensions by S alone (PiSSA-style).
        group_init() re-selects by score[i] = S[i] * pool|X @ Vh[i]|: dimensions
        that are both large in W AND active on real inputs. pool = 'rms' (outlier-
        sensitive, the ASVD intuition that activation outliers carry signal) or
        'mean_abs' (the original, outlier-robust). If calibration_data is None the
        weight-SVD init from init() is kept.
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

            # Recover W_orig: init() wrote W_res into layer.weight and stored top-r.
            W_res = layer.weight.data.float()
            U_old = layer.lora_U.float()
            S_old = layer.lora_S.float()
            Vh_old = layer.lora_Vh.float()
            W_orig = W_res + (U_old * S_old.unsqueeze(0)) @ Vh_old

            U_full, S_full, Vh_full = torch.linalg.svd(W_orig, full_matrices=False)
            proj = X.to(Vh_full) @ Vh_full.T              # (N, k) input in S-coords (X captured on CPU)
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

        # Per-direction reweighting: S_eff = S * (1 + ELU(coeff * g)).
        #   1 + ELU(z) = exp(z) for z<=0,  1+z for z>0.
        # Why this and not the obvious ones (all of which we tried):
        #   linear  S*(1+z)        : constant gradient (stable), but z<-1 -> S_eff<0,
        #                            a sign flip that drives incoherence. Unstable in
        #                            the negatives.
        #   exp     S*exp(z)       : positive, but unbounded and the gradient self-
        #                            amplifies (d/dz exp = exp), so amplification blows up.
        #   tanh    S*exp(c*tanh z): bounded, but c is an arbitrary free knob with no
        #                            principled value, and saturation kills the gradient.
        #   1+ELU                  : uses each in its safe regime -- exp only where it is
        #                            bounded in (0,1] (attenuation, cannot go negative),
        #                            linear where exp would diverge (amplification, const
        #                            gradient). C1 at z=0 (both -> 1, slope 1); >0 always.
        # coeff=0 or g=0 -> S_eff = S (identity). coeff<0 swaps amplify/suppress.
        S_eff = S * (1.0 + torch.nn.functional.elu(coeff * g))

        h = (x @ Vh.T) * S_eff                               # input in S-coords, reweighted
        return y + h @ U.T
