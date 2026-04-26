"""IA3-style output gating. y_new = y * g, with g initialized to ones."""
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