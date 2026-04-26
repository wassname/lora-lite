"""DeLoRA: column-normalised A, B, scaled by lambda * ||W||_F / r.

Bini et al. 2025 (ICLR'25)  https://arxiv.org/abs/2503.18225

Paper Eq. 8:    W' = W + (lambda * ||W||_F / r) B Xi A
where Xi_{i,i} = 1 / (||b_i|| ||a_i||) makes each rank-1 component unit-norm.
This is equivalent to row-normalising A and column-normalising B (each column of
B and row of A has unit norm), so each rank-1 outer product b_i a_i^T has unit
spectral norm -> the whole low-rank update is bounded.

Identity at t=0: paper uses kaiming init for both A and B with `lambda` initialised
to 0 (or small) so the effective delta starts near zero. We honour that:
default lambda0 == 0 gives bit-identity; user can override via variant_kwargs.

KNOWN GRADIENT ISSUE (flagged by external review 2026-04-26):
  With lambda0=0 the *forward* is identity but `A,B` get zero gradient on step 0
  (delta = lambda * ... -> d_output/d_A is proportional to lambda). Only
  `lora_lambda` moves first step. With lambda0>0, A,B train but identity is broken.
  Paper's true initialization (frozen-copy trick, see Eq. 9) achieves both;
  we do NOT implement that here.

The frozen ||W||_F factor is captured once at init() into a buffer `lora_wnorm`.

Reference implementations (for review/cross-check):
  - DeLoRA paper authors (ExplainableML/DeLoRA) -- their fork of peft:
    https://github.com/ExplainableML/DeLoRA/blob/main/peft/src/peft/tuners/delora.py
    (offline: docs/refs/orig_delora.py)
  - peft DeLoRA (upstreamed):
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/delora/layer.py
    (offline: docs/refs/peft_delora_layer.py)
"""
import torch
import torch.nn.functional as F
from einops import einsum
from torch import nn

from ..variant import register, ParamSpec


@register
class DeLoRA:
    name = "delora"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        lam0 = float(cfg.variant_kwargs.get("lambda0", 0.0))
        return {
            "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
            "lora_B": ParamSpec((d_out, cfg.r), init="kaiming", trainable=True),
            "lora_lambda": ParamSpec(
                (), init=lambda t: t.fill_(lam0), trainable=True
            ),
            # ||W||_F captured at init; frozen scalar buffer (no grad)
            "lora_wnorm": ParamSpec((), init="zeros", trainable=False),
        }

    @staticmethod
    def init(layer: nn.Linear, cfg) -> None:
        # Reading layer.weight only works for plain Linear; for bnb layers this
        # dequantizes via .float() round-trip if available, or fails cleanly.
        with torch.no_grad():
            W = layer.weight.data.float()
            layer.lora_wnorm.data.fill_(W.norm().item())
        return

    @staticmethod
    def forward(layer: nn.Linear, x, y):
        cfg = layer._lora_cfg
        # rows of A unit, cols of B unit (per paper, equivalent to Xi)
        A = F.normalize(layer.lora_A, dim=1)              # (r, d_in)
        B = F.normalize(layer.lora_B, dim=0)              # (d_out, r)
        scale = layer.lora_lambda * layer.lora_wnorm / cfg.r
        h = einsum(x, A, "... i, r i -> ... r")
        delta = einsum(h, B, "... r, o r -> ... o")
        return y + scale * delta
