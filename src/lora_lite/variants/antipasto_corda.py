"""AntiPaSTO-CorDA: reweight in a covariance-oriented basis, not the weight basis.

Plain SVD sorts directions by weight gain ||W v|| on isotropic input. The behaviour
you steer lives where the DATA has energy, off the top weight-singular axes. CorDA
(Yang+ 2024, arXiv:2406.05223) re-orients the SVD by the input covariance, so the top-r
directions move the output most on real activations.

    C = E[x x^T] (+ eps I)             # input second moment on calibration data
    C^{1/2}, C^{-1/2} via eigh(C)
    U S Vht = SVD(W C^{1/2})           # top-r is Eckart-Young best under x ~ N(0,C)
    P = Vht C^{-1/2}                   # (r, d_in) oblique input projector
    W = W_res + U_r diag(S_r) P_r      # exact (residual carries the dropped tail)
    S_eff = S * (1 + ELU(coeff*g))     # same bounded gain as antipasto
    y = x @ W_res.T + ((x @ P.T) * S_eff) @ U.T

Identity at g=0 or coeff=0: S_eff=S. P is oblique (rows not orthonormal -- C^{-1/2}
skews them); fine for gain reweighting since U stays orthonormal. Requires
calibration_data (group_init raises otherwise).

Refs: antipasto.py (gain + selection sibling), CorDA arXiv:2406.05223.
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


def _covariance_orient(model, targets, cfg, calibration_data, *, diag: bool) -> None:
    """Re-orient each target's SVD by its input second moment, then rewrite the frozen
    buffers (U, S, P) and residual weight in that basis. Shared by CorDA and ASVD:

        diag=False -> CorDA: full C = E[x x^T] (cross-channel covariance, via eigh)
        diag=True  -> ASVD:  diag(C) = E[x_i^2] only (per-channel scale, O(d_in), no eigh)

    The off-diagonal of C is the sole difference. g=0 stays exact identity either way --
    the reconstruction (W_res + U_r S_r P_r = W_orig) is lossless. Accumulated on CPU: a
    full C is d_in^2 fp32 per target and would crowd the GPU; the diagonal is a d_in vector.
    Call at attach-time, before training touches g (re-orienting g=0 is a no-op).
    """
    if calibration_data is None:
        raise ValueError("covariance orientation requires calibration_data; got None.")

    layers = {name: layer for name, layer, _ in targets}
    moment: dict[str, T] = {}                          # (d_in,d_in) full, or (d_in,) diagonal
    cnt: dict[str, int] = {n: 0 for n in layers}
    keep: dict[str, T] = {}                            # non-pad mask of the in-flight batch

    def make_hook(name):
        def _h(module, args, kwargs):
            x = rearrange(args[0].detach(), "... d -> (...) d").to(torch.float32).cpu()
            if "mask" in keep:
                x = x[keep["mask"]]                     # drop padding positions (see loop below)
            m = x.pow(2).sum(0) if diag else x.T @ x
            moment[name] = m if name not in moment else moment[name] + m
            cnt[name] += x.shape[0]                     # real (non-pad) tokens accumulated
        return _h

    handles = [layers[n].register_forward_pre_hook(make_hook(n), with_kwargs=True) for n in layers]
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                # Padding activations are not task-representative; mask them out of the moment
                # so the oriented basis reflects real tokens (CorDA/SVD-LLM official code does
                # the same). The mask is per-token, shared across all target layers in a batch.
                keep.pop("mask", None)
                if isinstance(batch, dict):
                    if "attention_mask" in batch:
                        keep["mask"] = rearrange(batch["attention_mask"], "... -> (...)").bool().cpu()
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
            raise RuntimeError(f"covariance orient at {name}: {cnt[name]} tokens, need >= r={r}")
        # decomposition on CPU (where the moment lives); results copied back to device buffers.
        W_res = layer.weight.data.float().cpu()
        U_old, S_old, P_old = (layer.lora_U.float().cpu(),
                               layer.lora_S.float().cpu(),
                               layer.lora_P.float().cpu())
        W_orig = W_res + (U_old * S_old) @ P_old

        if diag:
            c = (moment[name] / cnt[name]).clamp_min(0) + eps   # (d_in,) per-channel scale
            Ut, St, Vht = torch.linalg.svd(W_orig * c.sqrt(), full_matrices=False)  # @ diag(c^1/2)
            Pr = Vht[:r] * c.rsqrt()                            # @ diag(c^-1/2): oblique projector
        else:
            C = moment[name] / cnt[name]                        # (d_in,d_in)
            lam, Q = torch.linalg.eigh(C)
            lam = lam.clamp_min(0) + eps
            Mhalf    = (Q * lam.sqrt())  @ Q.T                   # C^{1/2}
            Minvhalf = (Q * lam.rsqrt()) @ Q.T                   # C^{-1/2}
            Ut, St, Vht = torch.linalg.svd(W_orig @ Mhalf, full_matrices=False)
            Pr = Vht[:r] @ Minvhalf                             # (r, d_in) oblique projector
        # Quantize the frozen buffers to their stored dtype FIRST, then form the residual
        # against those exact (bf16) values. The forward reconstructs from the bf16 buffers,
        # so W_res + U_r S_r P_r = W_orig to one residual-rounding -- without this, the
        # residual is built from fp32 U/S/P and the forward also eats the U/S/P quantization
        # mismatch, so g=0 drifts further from identity.
        Ur = Ut[:, :r].to(layer.lora_U.dtype)
        Sr = St[:r].to(layer.lora_S.dtype)
        Pr = Pr.to(layer.lora_P.dtype)
        W_res_new = W_orig - (Ur.float() * Sr.float()) @ Pr.float()

        with torch.no_grad():
            layer.lora_U.copy_(Ur)
            layer.lora_S.copy_(Sr)
            layer.lora_P.copy_(Pr)
            layer.weight.data.copy_(W_res_new.to(layer.weight))


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
        """CorDA: re-orient by the full input covariance C = E[x x^T] (cross-channel)."""
        _covariance_orient(model, targets, cfg, calibration_data, diag=False)

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
