"""IA3-style elementwise gating. Liu et al. 2022  https://arxiv.org/abs/2205.05638

Two registered variants, matching the paper's two regimes:

* `ia3`     -- OUTPUT-side gating, parameter shape (d_out,).
              y_new = y * g.  Use for attention projections (k_proj, v_proj).

* `ia3_ff`  -- INPUT-side gating, parameter shape (d_in,).
              y_new = base_layer(x * g).  Use for FFN-down layers (down_proj,
              fc2). Equivalent to the paper's "gate the FFN intermediate (post-
              activation)" position because down_proj's input IS that
              intermediate hidden state.

In both cases g is initialized to 1 -> identity at t=0.

To match the paper exactly on a Llama/Qwen-style block requires TWO attach
passes (one per variant), since each variant uses one hook type:

    cfg_attn = IA3Config(   target_names=(r"\\.k_proj$", r"\\.v_proj$"))
    cfg_ffn  = IA3FFConfig( target_names=(r"\\.down_proj$",))

Reference implementation:
  - peft IA3 layer (is_feedforward toggles input-vs-output gating, see
    docs/refs/peft_ia3_layer.py:177-188 forward and :214 update_layer):
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/layer.py
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
        return {"lora_g": ParamSpec((d_out,), init="ones", trainable=True)}

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
        return {"lora_g": ParamSpec((d_in,), init="ones", trainable=True)}

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        return

    @staticmethod
    def forward_input(
        layer: nn.Module,
        x: Float[T, '*B i'],
    ) -> Float[T, '*B i']:
        return x * layer.lora_g