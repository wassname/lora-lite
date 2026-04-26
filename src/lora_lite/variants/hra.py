"""HRA: Householder Reflection Adaptation. Yuan et al. 2024  https://arxiv.org/abs/2405.17484

Output-side formulation with an identity-init gate:

    y' = (1 - alpha) * y + alpha * R y         (so y' = y when alpha = 0)
    R  = prod_{i=1..r} H_i,    H_i = I - 2 u_i u_i^T / ||u_i||^2

`lora_gate` is initialized to 0 so y' = y at t=0. `lora_U` is initialized
kaiming so ||u_i||^2 is well-defined (no 0/0). Gradients flow into both U and
the gate even at init.

Hook-only, no weight access -> works on bnb 4/8-bit layers.
"""
import torch
from einops import einsum
from torch import nn

from ..variant import register, ParamSpec


@register
class HRA:
    name = "hra"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        return {
            # one Householder vector per rank slot in R^{d_out}
            "lora_U": ParamSpec((cfg.r, d_out), init="kaiming", trainable=True),
            # identity gate; 0 -> y' = y exactly
            "lora_gate": ParamSpec((), init="zeros", trainable=True),
        }

    @staticmethod
    def init(layer: nn.Linear, cfg) -> None:
        return

    @staticmethod
    def forward(layer: nn.Linear, x, y):
        U = layer.lora_U                                     # (r, d_out)
        Ry = y
        for i in range(U.shape[0]):
            u = U[i]
            sq = (u * u).sum().clamp_min(1e-12)
            coeff = einsum(Ry, u, "... o, o -> ...") * (2.0 / sq)
            Ry = Ry - coeff.unsqueeze(-1) * u
        return y + layer.lora_gate * (Ry - y)
