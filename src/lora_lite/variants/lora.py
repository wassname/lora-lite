"""Vanilla LoRA. Reference variant. y = Wx + (alpha/r) * B @ A @ x."""
from einops import einsum
from torch import nn
import torch

from ..variant import register, ParamSpec


@register
class LoRA:
    name = "lora"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        return {
            "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
            "lora_B": ParamSpec((d_out, cfg.r), init="zeros",  trainable=True),
        }

    @staticmethod
    def init(layer: nn.Linear, cfg) -> None:
        # B is zeros => delta=0 at t=0; identity invariant holds.
        return

    @staticmethod
    def forward(layer: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = layer._lora_cfg
        scale = cfg.alpha / cfg.r
        h = einsum(x, layer.lora_A, "... i, r i -> ... r")
        delta = einsum(h, layer.lora_B, "... r, o r -> ... o")
        return y + scale * delta
