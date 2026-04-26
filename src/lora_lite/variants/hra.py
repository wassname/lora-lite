"""HRA: Householder Reflection Adaptation. Yuan et al. 2024  https://arxiv.org/abs/2405.17484

Paper formulation (Sec. 3): adapt each frozen weight as

    W' = W R,    R = prod_{i=1..r} H_i,    H_i = I - 2 u_i u_i^T / ||u_i||^2

so the layer output becomes  y' = W' x = W (R x).  R is in INPUT space (d_in x d_in).

We implement this via a `forward_input` pre-hook that returns `R x`, then the
frozen base layer (including bnb 4/8-bit Linear) computes `W (R x)` itself.

Identity at t=0: `lora_gate` is initialized to 0 and gates each Householder
vector, so the effective u_i starts at 0 -> H_i = I -> R = I -> y' = y.
At training time the gate scales the active reflection direction.

OMITTED: paper also adds an orthogonality regularizer
    lambda * sum_i (u_i^T u_j)^2          (Eq. 6 / Sec. 3.3)
which is a loss term, not a forward-pass change. Add it in your training loop if
you want the regularized HRA variant.
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
            # one Householder vector per rank slot in INPUT space R^{d_in}
            "lora_U": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
            # identity gate; 0 -> R = I exactly
            "lora_gate": ParamSpec((), init="zeros", trainable=True),
        }

    @staticmethod
    def init(layer: nn.Linear, cfg) -> None:
        return

    @staticmethod
    def forward_input(layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        """Apply x + gate * (Rx - x). gate=0 -> identity; nonzero -> full Householder chain."""
        U = layer.lora_U                                     # (r, d_in)
        Rx = x
        for i in range(U.shape[0]):
            u = U[i]                                         # (d_in,)
            sq = (u * u).sum().clamp_min(1e-12)
            coeff = einsum(Rx, u, "... i, i -> ...") * (2.0 / sq)
            Rx = Rx - coeff.unsqueeze(-1) * u
        return x + layer.lora_gate * (Rx - x)
