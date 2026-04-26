"""DoRA: weight-decomposed LoRA. Liu et al. 2024  https://arxiv.org/abs/2402.09353

W' = m * V / ||V||_c   where V = W + (alpha/r) B A   (||.||_c = per-output-row L2 norm)

At t=0:  B=0 -> V=W -> y_new = (m_init / ||W||_c) (Wx + 0) = Wx   when m_init = ||W||_c.

Limitation: requires materializing the dense weight to compute ||V||_c. v1 supports
plain nn.Linear only; bnb 4/8-bit layers raise loudly.

DEVIATION (numerical):
  - We differentiate through ||V||_c every forward. The paper's sec. 4.3 suggests
    a 'cost-saving' variant that detaches ||V|| in backward (gradient only flows
    through V); we do NOT do that. Real impact: slower step, slightly different
    gradient direction. Faithful to the eq.5 forward, not the optimized one.

Reference implementations (for review/cross-check):
  - peft DoRA (separate file under lora/):
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py
    (offline: docs/refs/peft_lora_dora.py)
"""
import torch
from einops import einsum
from jaxtyping import Float
from torch import nn, Tensor as T
from dataclasses import dataclass

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config


@register_config
@dataclass
class DoRAConfig(AdapterConfig):
    variant: str = "dora"


@register
class DoRA:
    name = "dora"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        return {
            "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
            "lora_B": ParamSpec((d_out, cfg.r), init="zeros",  trainable=True),
            # m is filled from ||W||_c during init(); shape (d_out,)
            "lora_m": ParamSpec((d_out,), init="zeros", trainable=True),
        }

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        if type(layer) is not nn.Linear:
            raise TypeError(
                "DoRA needs ||W||_c, so v1 only supports plain nn.Linear. "
                "For bnb layers, dequantize first or use LoRA/IA3."
            )
        with torch.no_grad():
            W = layer.weight.data.float()                      # (d_out, d_in)
            col_norm = W.norm(dim=1).to(layer.lora_m.dtype)    # (d_out,)
            layer.lora_m.data.copy_(col_norm)

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        scale = cfg.alpha / cfg.r
        # V = W + scale * B @ A
        BA = einsum(layer.lora_B, layer.lora_A, "o r, r i -> o i")
        V = layer.weight + scale * BA                          # (d_out, d_in)
        v_norm = V.norm(dim=1).clamp_min(1e-12)                # (d_out,)
        # Bias passes through UNSCALED -- only Wx + scale*BAx is normalized.
        # Matches peft DoRA forward (docs/refs/peft_lora_dora.py:157-161).
        bias = getattr(layer, "bias", None)
        wx = y if bias is None else (y - bias)
        h = einsum(x, layer.lora_A, "... i, r i -> ... r")
        delta = einsum(h, layer.lora_B, "... r, o r -> ... o")
        combined = wx + scale * delta
        out = (layer.lora_m / v_norm) * combined
        return out if bias is None else out + bias
