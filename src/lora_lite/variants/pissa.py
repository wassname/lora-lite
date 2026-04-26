"""PiSSA: top-r SVD of W into A,B; replace W with W_res = W - B@A.

Meng et al. 2024  https://arxiv.org/abs/2404.02948

    W = U S Vh        (truncated to top-r)
    B = U sqrt(S),  A = sqrt(S) Vh,   W_res = W - B A

Identity at t=0: W_res + B@A == W (bf16 round-trip, not bit-exact).
Pass alpha=r for paper-faithful scale=1.

Refs:
  - paper: https://github.com/MuLabPKU/PiSSA/blob/main/utils/init_pissa.py
    (offline: docs/refs/orig_pissa_init.py)
  - peft:  https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
    (offline: docs/refs/peft_lora_layer.py, see pissa_init path)
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
class PiSSAConfig(AdapterConfig):
    variant: str = "pissa"


@register
class PiSSA:
    name = "pissa"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        return {
            "lora_A": ParamSpec((cfg.r, d_in), init="zeros", trainable=True),
            "lora_B": ParamSpec((d_out, cfg.r), init="zeros", trainable=True),
        }

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        if type(layer) is not nn.Linear:
            raise TypeError(
                "PiSSA mutates layer.weight into W_res, so v1 only supports plain nn.Linear. "
                "For bnb 4/8-bit, use LoRA/DeLoRA or implement explicit dequantize/requantize."
            )
        W = layer.weight.data.float()                       # (d_out, d_in)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = cfg.r
        Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
        sqrtS = Sr.sqrt()
        # B @ A = Ur diag(Sr) Vhr;  pick B = Ur sqrt(Sr),  A = sqrt(Sr) * Vhr
        B = (Ur * sqrtS).to(cfg.dtype)
        A = (sqrtS[:, None] * Vhr).to(cfg.dtype)
        layer.lora_B.data.copy_(B)
        layer.lora_A.data.copy_(A)
        # fp32 subtraction so W_res stays accurate.
        BA = (B.float() @ A.float())
        scale = cfg.alpha / cfg.r
        layer.weight.data.copy_((W - scale * BA).to(layer.weight.dtype))

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
