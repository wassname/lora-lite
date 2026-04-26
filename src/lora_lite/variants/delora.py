"""DeLoRA: column-normalised A, B, scaled by lambda/r. Bini et al. 2025  arXiv:2503.18225.

NOTE on identity at t=0: paper uses kaiming for both A and B with a learned lambda
init at 0 (or small) so the effective delta starts near zero. We honour that:
default lambda0 == 0 gives bit-identity; user can override via variant_kwargs.
"""
import torch
import torch.nn.functional as F
from einops import einsum
from torch import nn

from ..variant import register, ParamSpec


@register
class DeLoRA:
    name = "delora"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        lam0 = float(cfg.variant_kwargs.get("lambda0", 0.0))
        return {
            "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
            "lora_B": ParamSpec((d_out, cfg.r), init="kaiming", trainable=True),
            "lora_lambda": ParamSpec(
                (), init=lambda t: t.fill_(lam0), trainable=True
            ),
        }

    @staticmethod
    def init(layer: nn.Linear, cfg) -> None:
        return

    @staticmethod
    def forward(layer: nn.Linear, x, y):
        cfg = layer._lora_cfg
        # rows of A unit, cols of B unit (per paper)
        A = F.normalize(layer.lora_A, dim=1)              # (r, d_in)
        B = F.normalize(layer.lora_B, dim=0)              # (d_out, r)
        scale = layer.lora_lambda / cfg.r
        h = einsum(x, A, "... i, r i -> ... r")
        delta = einsum(h, B, "... r, o r -> ... o")
        return y + scale * delta
