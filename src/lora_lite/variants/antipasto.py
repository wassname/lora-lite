"""AntiPaSTO: SVD steering with learnable singular-value deltas + block-diagonal Cayley rotation.

wassname 2026  https://arxiv.org/abs/2601.07473

    W = U diag(S) Vh + W_res        (top-r SVD; W_res = W - U_r S_r Vh_r)
    learn: delta_s (r,), rot_T (n_blocks, bs(bs-1)/2)
    R = block_diag(Cayley(skew(rot_T)));  Vh_eff = R @ Vh (or U_eff = U @ R.T)
    y = x @ W_res.T + ((x @ Vh_eff.T) * (S + delta_s)) @ U_eff.T

Identity at t=0: rot_T~0 -> R≈I, delta_s~0 -> y ≈ x @ W^T (fp32 SVD round-trip). near_zero init breaks bf16 symmetry without meaningfully breaking identity (~1e-4 noise around zero).

Scope cut vs antipasto3: this is a fine-tuning adapter, not the full runtime
steering interface. There is no per-call alpha, so it does not expose the
bidirectional R(+alpha) / R(-alpha) inference symmetry. The V-basis path uses the
opposite chirality to antipasto3's default U-basis path, so checkpoints are not
portable without a sign/basis convention.

Refs:
  - paper: https://github.com/wassname/AntiPaSTO
  - lite port of: https://github.com/wassname/antipasto3
    (offline: docs/refs/antipasto3_svd_adapter.py)
"""
import math
from dataclasses import dataclass
from typing import Literal

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import nn, Tensor as T

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config


@register_config
@dataclass
class AntiPaSTOConfig(AdapterConfig):
    variant: str = "antipasto"
    # Block size for the block-diagonal Cayley rotation. r must be divisible by it.
    block_size: int = 4
    # Cayley map saturation: bounds rotation angle to ~max_rotation_angle radians.
    max_rotation_angle: float = 0.5
    # Which singular basis to rotate: 'V' (input) or 'U' (output).
    rotate_basis: Literal["V", "U"] = "V"


def _cayley(skew: torch.Tensor) -> torch.Tensor:
    """R = (I - X)^-1 (I + X) for X = skew/2; preserves orthogonality."""
    bs = skew.shape[-1]
    eye = torch.eye(bs, dtype=skew.dtype, device=skew.device).expand_as(skew)
    X = skew / 2
    return torch.linalg.solve(eye - X, eye + X)


def _build_rotation(rot_T: torch.Tensor, bs: int, max_angle: float) -> torch.Tensor:
    """rot_T: (n_blocks, bs*(bs-1)/2) -> R: (n_blocks, bs, bs) Cayley rotation."""
    n_blocks, _ = rot_T.shape
    rows, cols = torch.triu_indices(bs, bs, offset=1, device=rot_T.device).unbind(0)
    A = torch.zeros(n_blocks, bs, bs, dtype=rot_T.dtype, device=rot_T.device)
    A[:, rows, cols] = rot_T
    A = 0.5 * (A - A.transpose(-1, -2))
    a_limit = 2.0 * math.tan(max_angle / 2.0)
    A = a_limit * torch.tanh(A / a_limit)
    return _cayley(A)


@register
class AntiPaSTO:
    name = "antipasto"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        r = cfg.r
        bs = int(cfg.block_size)
        if r % bs != 0:
            raise ValueError(f"AntiPaSTO requires r={r} divisible by block_size={bs}")
        n_blocks = r // bs
        n_triu = bs * (bs - 1) // 2
        return dict(
            # Frozen SVD components captured at init.
            lora_U=ParamSpec((d_out, r), init="zeros", trainable=False, as_buffer=True),
            lora_S=ParamSpec((r,), init="zeros", trainable=False, as_buffer=True),
            lora_Vh=ParamSpec((r, d_in), init="zeros", trainable=False, as_buffer=True),
            # Trainable: per-singular-value delta + block-diagonal Cayley rotation.
            lora_delta_s=ParamSpec((r,), init="near_zero"),
            lora_rot_T=ParamSpec((n_blocks, n_triu), init="near_zero"),
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

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        bs = int(cfg.block_size)
        max_angle = float(cfg.max_rotation_angle)
        rotate_basis = cfg.rotate_basis

        U = layer.lora_U.to(x.dtype)                          # (d_out, r)
        S = layer.lora_S.to(x.dtype)                          # (r,)
        Vh = layer.lora_Vh.to(x.dtype)                        # (r, d_in)

        R_blocks = _build_rotation(layer.lora_rot_T.float(), bs, max_angle).to(x.dtype)
        n_blocks = R_blocks.shape[0]                          # R_blocks: (n, bs, bs)

        # Apply block-diagonal R per-block via einsum, never materializing (r,r).
        if rotate_basis == "V":
            # Vh_eff = R @ Vh, viewed block-wise on the r-axis.
            Vh_blocks = rearrange(Vh, "(n a) i -> n a i", n=n_blocks)
            Vh_rot = einsum(R_blocks, Vh_blocks, "n a b, n b i -> n a i")
            Vh_eff = rearrange(Vh_rot, "n a i -> (n a) i")
            U_eff = U
        elif rotate_basis == "U":
            # U_eff = U @ R.T, viewed block-wise on the r-axis.
            U_blocks = rearrange(U, "d (n b) -> d n b", n=n_blocks)
            U_rot = einsum(U_blocks, R_blocks, "d n b, n c b -> d n c")
            U_eff = rearrange(U_rot, "d n c -> d (n c)")
            Vh_eff = Vh
        else:
            raise ValueError(f"rotate_basis must be 'U' or 'V', got {rotate_basis!r}")

        S_eff = S + layer.lora_delta_s.to(x.dtype)            # (r,)
        h = einsum(x, Vh_eff, "... i, r i -> ... r")          # x @ Vh_eff.T
        h = h * S_eff                                         # diag(S_eff)
        delta = einsum(h, U_eff, "... r, o r -> ... o")       # @ U_eff.T
        return y + delta
