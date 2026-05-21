"""Vanilla LoRA. Hu et al. 2021  https://arxiv.org/abs/2106.09685

    h = W x + (alpha/r) B A x

Identity at t=0 from B=0.

Refs:
  - peft: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
    (offline: docs/refs/peft_lora_layer.py)
"""
from jaxtyping import Float
from torch import nn, Tensor as T
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
        return dict(
            lora_A=ParamSpec((cfg.r, d_in), init="kaiming"),
            lora_B=ParamSpec((d_out, cfg.r), init="zeros"),
        )

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        return

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        scale = cfg.alpha / cfg.r
        xA = x.to(layer.lora_A.dtype)
        h = xA @ layer.lora_A.T
        delta = h @ layer.lora_B.T
        return y + (scale * delta).to(y.dtype)
