"""HRA: Householder Reflection Adaptation. Yuan et al. 2024  https://arxiv.org/abs/2405.17484

    W' = W R,   R = prod_{i=1..r} H_i,   H_i = I - 2 u_i u_i^T / ||u_i||^2

R is in input space (d_in x d_in); applied via a `forward_input` pre-hook so the
frozen base layer (bnb 4/8-bit OK) computes W (R x).

Identity at t=0: peft-style symmetric init -- U pairs (U[0]=U[1], ...) so adjacent
H_i H_i = I cancel, R = I exactly. Requires even r. Paired rows diverge after step 1.

Note: paper's orthogonality regularizer (Eq. 6) is loss-side; add it in your loop.

Refs:
  - paper code: https://github.com/DaShenZi721/HRA/blob/master/llama/peft/oft/layer_GS_HRA.py
    (offline: docs/refs/orig_hra_layer.py)
  - peft:       https://github.com/huggingface/peft/blob/main/src/peft/tuners/hra/layer.py
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
        # Symmetric init: kaiming(r//2, d_in) repeat-interleaved -> R = I, grad alive.
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
        """x -> x R^T = x H_{r-1} ... H_0. Iterate i = r-1 down to 0 to match peft."""
        U = layer.lora_U                                     # (r, d_in)
        Rx = x
        for i in range(U.shape[0] - 1, -1, -1):
            u = U[i]                                         # (d_in,)
            sq = (u * u).sum().clamp_min(1e-12)
            coeff = einsum(Rx, u, "... i, i -> ...") * (2.0 / sq)
            Rx = Rx - coeff.unsqueeze(-1) * u
        return Rx
