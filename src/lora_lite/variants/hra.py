"""HRA: Householder Reflection Adaptation. Yuan et al. 2024  https://arxiv.org/abs/2405.17484

Paper formulation (Sec. 3): adapt each frozen weight as

    W' = W R,    R = prod_{i=1..r} H_i,    H_i = I - 2 u_i u_i^T / ||u_i||^2

so the layer output becomes  y' = W' x = W (R x).  R is in INPUT space (d_in x d_in).

We implement this via a `forward_input` pre-hook that returns `R x`, then the
frozen base layer (including bnb 4/8-bit Linear) computes `W (R x)` itself.

Identity at t=0 (PEFT-style symmetric init, requires even r):
  Rows are kaiming-init in pairs: U[0]=U[1], U[2]=U[3], ...  Adjacent pairs of
  Householder reflections with identical vectors cancel exactly
  (H_i H_i = I), so R = I at init -> y' = y to bit-precision.
  After the first gradient step the paired rows diverge and the chain becomes a
  general orthogonal matrix; gradient flows into U from step 0 (no dead-grad).
  Odd r is rejected (matches peft warning behaviour).

OMITTED: paper also adds an orthogonality regularizer (Eq. 6 / Sec. 3.3),
a loss-side term. Add it in your training loop if you want regularized HRA.

Reference implementations (for review/cross-check):
  - HRA paper authors (DaShenZi721/HRA), llama variant of OFT layer with HRA:
    https://github.com/DaShenZi721/HRA/blob/master/llama/peft/oft/layer_GS_HRA.py
    (offline: docs/refs/orig_hra_layer.py)
  - peft HRA layer, reset_hra_parameters (lines 100-108):
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/hra/layer.py
    (offline: docs/refs/peft_hra_layer.py)
"""
import torch
from einops import einsum
from jaxtyping import Float
from torch import nn, Tensor as T
from dataclasses import dataclass

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config


@register_config
@dataclass
class HRAConfig(AdapterConfig):
    variant: str = "hra"


@register
class HRA:
    name = "hra"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        if cfg.r % 2 != 0:
            raise ValueError(
                f"HRA symmetric init requires even r; got r={cfg.r}. "
                "Pick an even rank or use a different variant."
            )
        return {
            # Householder vectors stacked as rows (one vector per rank slot)
            # init done in init() to enforce paired rows -> R = I at t=0.
            "lora_U": ParamSpec((cfg.r, d_in), init="zeros", trainable=True),
        }

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        # Symmetric init per peft (docs/refs/peft_hra_layer.py:101-108):
        #   half = kaiming(r//2, d_in); U = repeat_interleave(half, 2, dim=0)
        # Adjacent pairs (H_2k H_2k+1) cancel since H^2 = I, so R = I exactly,
        # while gradient still flows into U from step 0.
        with torch.no_grad():
            r, d_in = layer.lora_U.shape
            half = torch.empty(r // 2, d_in, dtype=layer.lora_U.dtype, device=layer.lora_U.device)
            nn.init.kaiming_uniform_(half, a=5 ** 0.5)
            layer.lora_U.copy_(torch.repeat_interleave(half, 2, dim=0))
        return

    @staticmethod
    def forward_input(
        layer: nn.Module,
        x: Float[T, '*B i'],
    ) -> Float[T, '*B i']:
        """Apply x -> x R^T where R = H_0 H_1 ... H_{r-1}, H_i = I - 2 u_i u_i^T / ||u_i||^2.

        peft applies `W @ R` so y = F.linear(x, W@R) = x @ R^T @ W^T. Our pre-hook
        produces `x @ R^T = x @ H_{r-1} ... H_0`, then the base layer computes
        `(x R^T) @ W^T = (x R^T W^T)`, matching peft (docs/refs/peft_hra_layer.py:225-264).

        Iterate i = r-1 down to 0: each step right-multiplies x by H_i, building
        x H_{r-1} H_{r-2} ... H_0 = x R^T.  At symmetric init H_{2k} H_{2k+1} = I
        regardless of order, so identity-at-t=0 holds either way; the order only
        matters once paired rows diverge.
        """
        U = layer.lora_U                                     # (r, d_in)
        Rx = x
        for i in range(U.shape[0] - 1, -1, -1):
            u = U[i]                                         # (d_in,)
            sq = (u * u).sum().clamp_min(1e-12)
            coeff = einsum(Rx, u, "... i, i -> ...") * (2.0 / sq)
            Rx = Rx - coeff.unsqueeze(-1) * u
        return Rx
