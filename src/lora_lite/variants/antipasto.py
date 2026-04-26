"""AntiPaSTO: SVD steering with learnable singular-value deltas + block-diagonal Cayley rotation.

Paper: https://arxiv.org/pdf/2601.07473  (wassname, AntiPaSTO -- SVD-based PEFT)
Repo:  https://github.com/wassname/AntiPaSTO
Lite port of the AntiPaSTO3 SVD adapter from
  https://github.com/wassname/antipasto3 (offline: docs/refs/antipasto3_svd_adapter.py)

Decomposition (PyTorch nn.Linear convention, weight (d_out, d_in)):

    W = U diag(S) Vh + W_res     (top-r SVD; W_res = W - U_r diag(S_r) Vh_r)

We freeze U, S, Vh, W_res and learn:
  - delta_s : (r,)                    -- additive delta to singular values
  - rot_T   : (n_blocks, bs(bs-1)/2)  -- upper-triangle of skew matrix per block

Forward (matches base layer convention exactly at t=0):

    R          = block_diag(Cayley(skew(rot_T)))            # (r, r) effective
    Vh_rot     = R @ Vh                                     # rotates input basis
    S_eff      = S + delta_s                                # learnable spectrum
    delta_y    = ((x @ Vh_rot.T) * S_eff) @ U.T             # rank-r path
    base_y     = x @ W_res.T                                # frozen residual
    y_total    = base_y + delta_y                           # == original output at t=0

At init: rot_T = 0 -> R = I -> Vh_rot = Vh, delta_s = 0 -> S_eff = S, so
delta_y reconstructs the truncated SVD term and y_total == x @ W^T to numerical
precision (fp32 SVD round-tripped to cfg.dtype).

WHICH BASIS IS ROTATED:
  By default we rotate Vh (the INPUT singular basis). This is what AntiPaSTO3
  calls `rotate_V=True` in adapter terms (V == Vh.T columns). To rotate U
  (output basis) instead, pass variant_kwargs={'rotate_basis': 'U'}.
  Rotating both is not implemented (one rotation is enough to span the
  identifiable steering directions; two is degenerate).

REQUIRES even rank divisible by `block_size` (default 4). r=8, bs=4 -> 2 blocks.
"""
import math

import torch
from einops import einsum
from jaxtyping import Float
from torch import nn, Tensor as T

from ..variant import register, ParamSpec


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


def _block_diag(blocks: torch.Tensor) -> torch.Tensor:
    """(n_blocks, bs, bs) -> (n_blocks*bs, n_blocks*bs) block-diagonal."""
    n, bs, _ = blocks.shape
    out = blocks.new_zeros(n * bs, n * bs)
    for i in range(n):
        out[i * bs : (i + 1) * bs, i * bs : (i + 1) * bs] = blocks[i]
    return out


@register
class AntiPaSTO:
    name = "antipasto"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        r = cfg.r
        bs = int(cfg.variant_kwargs.get("block_size", 4))
        if r % bs != 0:
            raise ValueError(f"AntiPaSTO requires r={r} divisible by block_size={bs}")
        n_blocks = r // bs
        n_triu = bs * (bs - 1) // 2
        return {
            # Frozen SVD components captured at init (buffers travel with state_dict).
            "lora_U": ParamSpec((d_out, r), init="zeros", trainable=False, as_buffer=True),
            "lora_S": ParamSpec((r,), init="zeros", trainable=False, as_buffer=True),
            "lora_Vh": ParamSpec((r, d_in), init="zeros", trainable=False, as_buffer=True),
            # Trainable: per-singular-value delta + block-diagonal Cayley rotation.
            "lora_delta_s": ParamSpec((r,), init="zeros", trainable=True),
            "lora_rot_T": ParamSpec((n_blocks, n_triu), init="zeros", trainable=True),
        }

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
            # W_res is the residual after rank-r truncation. Forward adds back
            # the truncated path so total == W exactly at init (mod dtype).
            W_res = (W - (Ur * Sr) @ Vhr).to(layer.weight.dtype)
            layer.weight.data.copy_(W_res)

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        bs = int(cfg.variant_kwargs.get("block_size", 4))
        max_angle = float(cfg.variant_kwargs.get("max_rotation_angle", 0.5))
        rotate_basis = cfg.variant_kwargs.get("rotate_basis", "V")

        U = layer.lora_U.to(x.dtype)                          # (d_out, r)
        S = layer.lora_S.to(x.dtype)                          # (r,)
        Vh = layer.lora_Vh.to(x.dtype)                        # (r, d_in)

        R_blocks = _build_rotation(layer.lora_rot_T.float(), bs, max_angle)
        R = _block_diag(R_blocks).to(x.dtype)                 # (r, r)

        if rotate_basis == "V":
            Vh_eff = R @ Vh                                   # rotate INPUT basis
            U_eff = U
        elif rotate_basis == "U":
            Vh_eff = Vh
            U_eff = U @ R.T                                   # rotate OUTPUT basis
        else:
            raise ValueError(f"rotate_basis must be 'U' or 'V', got {rotate_basis!r}")

        S_eff = S + layer.lora_delta_s.to(x.dtype)            # (r,)
        h = einsum(x, Vh_eff, "... i, r i -> ... r")          # x @ Vh_eff.T
        h = h * S_eff                                         # diag(S_eff)
        delta = einsum(h, U_eff, "... r, o r -> ... o")       # @ U_eff.T
        return y + delta
