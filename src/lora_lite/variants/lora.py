"""Vanilla LoRA. Hu et al. 2021  https://arxiv.org/abs/2106.09685

    h = W x + (alpha/r) B A x

Identity at t=0 from B=0.

Refs:
  - peft: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
    (offline: docs/refs/peft_lora_layer.py)
"""
from einops import einsum
from jaxtyping import Float
from torch import nn, Tensor as T
import torch
from dataclasses import dataclass

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config


@register_config
@dataclass
class LoRAConfig(AdapterConfig):
    variant: str = "lora"


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
    def init(layer: nn.Module, cfg) -> None:
        # B is zeros => delta=0 at t=0; identity invariant holds.
        return

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        scale = cfg.alpha / cfg.r
        h = einsum(x, layer.lora_A, "... i, r i -> ... r")
        delta = einsum(h, layer.lora_B, "... r, o r -> ... o")
        return y + scale * delta
