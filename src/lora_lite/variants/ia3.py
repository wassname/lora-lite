"""IA3-style output gating. Liu et al. 2022  https://arxiv.org/abs/2205.05638

    y_new = y * g,    g initialized to 1  (identity at t=0)

DEVIATION FROM PAPER:
    The original IA3 gates only three positions per transformer block:
        l_k * (k_proj output),  l_v * (v_proj output),  l_ff * (FFN intermediate after activation)
    This implementation gates ANY linear layer the targeting system selects.
    To match the paper exactly on a typical Llama/Qwen-style block, attach with:

        cfg = LoraLiteConfig(
            variant="ia3",
            target_names=(r"\\.k_proj$", r"\\.v_proj$", r"\\.up_proj$"),
            target_roles=(),
        )

    `up_proj` is the closest stand-in for "FFN intermediate" in gated-MLP blocks
    (Llama uses gate * up; gating the up branch is the IA3-spirit choice).

Reference implementations (for review/cross-check):
  - peft IA3 layer (uses ia3_l elementwise scaling, fan_in_fan_out aware):
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/layer.py
    (offline: docs/refs/peft_ia3_layer.py)
"""
import torch
from torch import nn

from ..variant import register, ParamSpec


@register
class IA3:
    name = "ia3"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        return {"lora_g": ParamSpec((d_out,), init="ones", trainable=True)}

    @staticmethod
    def init(layer: nn.Linear, cfg) -> None:
        return

    @staticmethod
    def forward(layer: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y * layer.lora_g