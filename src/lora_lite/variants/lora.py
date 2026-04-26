"""Vanilla LoRA. Hu et al. 2021  https://arxiv.org/abs/2106.09685

    h = W x + (alpha/r) B A x

Identity at t=0 from B=0. Faithful to the paper.

Reference implementations (for review/cross-check):
  - peft Linear.update_layer + lora_A/B init, forward:
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
    (see docs/refs/peft_lora_layer.py for offline copy)
"""
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
