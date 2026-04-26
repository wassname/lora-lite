"""DeLoRA: per-input-channel weight-norm scaling, per-rank A/B normalization.

Bini et al. 2025 (ICLR'25)  https://arxiv.org/abs/2503.18225

Paper Eq. 8:    W' = W + (lambda * ||W||_F / r) B Xi A
where Xi_{i,i} = 1 / (||b_i|| ||a_i||) makes each rank-1 component unit-norm.

Implementation follows the peft upstream (which the DeLoRA authors maintain),
which differs from the paper notation in two ways that are equivalent at the
forward level but matter for gradients/numerics:
  1. ||W|| is captured PER INPUT CHANNEL (shape (d_in,)), not as a scalar
     Frobenius norm. Used to scale `x` element-wise on the input dim.
     See docs/refs/peft_delora_layer.py:150 (init) and :250 (forward).
  2. Per-rank normalization applied via division (1/||A_i||*||B^j||) inside
     the diagonal scaling, instead of as F.normalize on A,B themselves.
     This keeps the gradient flowing through the un-normalized parameters.

Identity at t=0: lambda0=0 -> delta is exactly zero (bit-identity).

KNOWN GRADIENT ISSUE (flagged by external review 2026-04-26):
  With lambda0=0 the *forward* is identity but `A,B` get zero gradient on step 0
  (delta is proportional to lambda). Only `lora_lambda` moves first step.
  The paper's true initialization (frozen-copy trick, Eq. 9) achieves both
  identity AND non-zero A/B gradients; we do NOT implement it here.

Reference implementations:
  - DeLoRA paper authors (ExplainableML/DeLoRA) -- their fork of peft:
    https://github.com/ExplainableML/DeLoRA/blob/main/peft/src/peft/tuners/delora.py
    (offline: docs/refs/orig_delora.py)
  - peft DeLoRA (upstreamed):
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/delora/layer.py
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
    # Initial scale for the per-layer learnable lambda. peft default is 15.0;
    # we default to 0.0 (identity at t=0 even before B is zero-initialized).
    lambda0: float = 0.0


@register
class DeLoRA:
    name = "delora"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        lam0 = float(cfg.lambda0)
        return {
            # peft DeLoRA default: A=kaiming, B=zeros (docs/refs/peft_delora_layer.py:138-140).
            # Identity at t=0 from B=0 -> delta=0 regardless of lambda. With B=0 the
            # delta is a function of B alone on step 0; gradient flows into B (nonzero)
            # and into A only after B becomes nonzero (step 2+). Matches peft.
            "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
            "lora_B": ParamSpec((d_out, cfg.r), init="zeros",   trainable=True),
            "lora_lambda": ParamSpec(
                (), init=lambda t: t.fill_(lam0), trainable=True
            ),
            # ||W||_2 per input channel (shape (d_in,)); frozen buffer captured at init
            # per peft DeLoRA (docs/refs/peft_delora_layer.py:150).
            "lora_wnorm": ParamSpec((d_in,), init="ones", trainable=False, as_buffer=True),
        }

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        # DeLoRA needs ||W||_2 per input column. Plain nn.Linear: just read weight.
        # bnb Linear8bitLt: weight is fp16 until first forward (then int8 + SCB),
        # so capturing here works; quality is correct only because we read pre-quant.
        # bnb Linear4bit / fully quantized layers: would give garbage. Use lora/ia3/hra
        # for those.
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
        # peft delora forward (docs/refs/peft_delora_layer.py:248-260):
        # h = (x * w_norm) @ A.T;  scale per-rank = (lambda/r) / (||A_i|| * ||B^j||);
        # delta = (h * scale) @ B.T
        x_scaled = x * layer.lora_wnorm                    # (..., d_in)
        h = einsum(x_scaled, A, "... i, r i -> ... r")
        An = torch.clamp(A.norm(dim=1), min=1e-4)          # (r,)
        Bn = torch.clamp(B.norm(dim=0), min=1e-4)          # (r,)
        scale = (layer.lora_lambda / cfg.r) / (An * Bn)    # (r,)
        h = h * scale
        delta = einsum(h, B, "... r, o r -> ... o")
        return y + delta
