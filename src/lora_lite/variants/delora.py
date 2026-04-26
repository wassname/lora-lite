"""DeLoRA: per-channel weight-norm scaling, per-rank A/B normalization.

Bini et al. 2025 (ICLR'25)  https://arxiv.org/abs/2503.18225

    W' = W + (lambda * ||W||_F / r) B Xi A,   Xi_{i,i} = 1 / (||b_i|| ||a_i||)

Per peft upstream: ||W|| is per-input-channel (not scalar Frobenius), and
per-rank norms divide inside the diag (not via F.normalize on A,B) so
gradients flow through un-normalized parameters.

Identity at t=0 comes from B=0, so any lambda0 keeps delta=0. Keep lambda0
nonzero for training: lambda0=0 makes every DeLoRA gradient zero on step 0.

Refs:
  - paper code: https://github.com/ExplainableML/DeLoRA/blob/main/peft/src/peft/tuners/delora.py
    (offline: docs/refs/orig_delora.py)
  - peft:       https://github.com/huggingface/peft/blob/main/src/peft/tuners/delora/layer.py
    (offline: docs/refs/peft_delora_layer.py)
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
class DeLoRAConfig(AdapterConfig):
    variant: str = "delora"
    # peft/paper default. B=0 preserves t=0 identity; lambda0=0 would make the
    # whole adapter dead on step 0 because delta and all gradients scale by lambda.
    lambda0: float = 15.0


@register
class DeLoRA:
    name = "delora"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        lam0 = float(cfg.lambda0)
        return {
            "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
            "lora_B": ParamSpec((d_out, cfg.r), init="zeros",   trainable=True),
            "lora_lambda": ParamSpec(
                (), init=lambda t: t.fill_(lam0), trainable=True
            ),
            # ||W||_2 per input channel; frozen buffer captured at init.
            "lora_wnorm": ParamSpec((d_in,), init="ones", trainable=False, as_buffer=True),
        }

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        # Reads weight pre-quant -- OK for nn.Linear and bnb 8bit (fp16 until 1st fwd).
        # bnb Linear4bit gives garbage; use lora/ia3/hra for those.
        with torch.no_grad():
            W = layer.weight.data.float()
            wnorm = W.norm(dim=0).detach().to(layer.lora_wnorm.dtype)
            layer.lora_wnorm.copy_(wnorm)
        return

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        A = layer.lora_A                                   # (r, d_in)
        B = layer.lora_B                                   # (d_out, r)
        x_scaled = x * layer.lora_wnorm                    # (..., d_in)
        h = einsum(x_scaled, A, "... i, r i -> ... r")
        An = torch.clamp(A.norm(dim=1), min=1e-4)          # (r,)
        Bn = torch.clamp(B.norm(dim=0), min=1e-4)          # (r,)
        scale = (layer.lora_lambda / cfg.r) / (An * Bn)    # (r,)
        h = h * scale
        delta = einsum(h, B, "... r, o r -> ... o")
        return y + delta
