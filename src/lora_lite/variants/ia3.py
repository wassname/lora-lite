"""IA3 elementwise gating. Liu et al. 2022  https://arxiv.org/abs/2205.05638

* `ia3`    -- output-side: y_new = y * g, g shape (d_out,). For k_proj, v_proj.
* `ia3_ff` -- input-side:  y_new = base(x * g), g shape (d_in,). For down_proj/fc2.

Identity at t=0: g=1.

Example (paper's Llama/Qwen block needs both passes):

    cfg_attn = IA3Config(   target_names=(r"\\.k_proj$", r"\\.v_proj$"))
    cfg_ffn  = IA3FFConfig( target_names=(r"\\.down_proj$",))

Refs:
  - peft: https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/layer.py
    (offline: docs/refs/peft_ia3_layer.py)
"""
import torch
from jaxtyping import Float
from torch import nn, Tensor as T
from dataclasses import dataclass

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config


@register_config
@dataclass
class IA3Config(AdapterConfig):
    variant: str = "ia3"


@register_config
@dataclass
class IA3FFConfig(AdapterConfig):
    variant: str = "ia3_ff"


@register
class IA3:
    name = "ia3"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        return dict(lora_g=ParamSpec((d_out,), init="ones"))

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        return

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        return y * layer.lora_g


@register
class IA3FF:
    name = "ia3_ff"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        return dict(lora_g=ParamSpec((d_in,), init="ones"))

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        return

    @staticmethod
    def forward_input(
        layer: nn.Module,
        x: Float[T, '*B i'],
    ) -> Float[T, '*B i']:
        return x * layer.lora_g