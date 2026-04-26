"""PiSSA: top-r SVD of W into A,B; replace W with W_res = W - B@A.

Meng et al. 2024  https://arxiv.org/abs/2404.02948
W_eff(t=0) = W_res + B@A = W (numerically; bf16 round-trip not bit-exact).
"""
import torch
from einops import einsum
from torch import nn

from ..variant import register, ParamSpec


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
    def init(layer: nn.Linear, cfg) -> None:
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
        # Compute BA in fp32 for the subtraction so W_res is accurate.
        BA = (B.float() @ A.float())
        # NOTE: PiSSA uses scale=1 (alpha==r) implicitly via init. We let the user pick;
        # for fidelity at t=0, the convention is alpha==r so scale==1. Document in README.
        scale = cfg.alpha / cfg.r
        layer.weight.data.copy_((W - scale * BA).to(layer.weight.dtype))

    @staticmethod
    def forward(layer: nn.Linear, x, y):
        cfg = layer._lora_cfg
        scale = cfg.alpha / cfg.r
        h = einsum(x, layer.lora_A, "... i, r i -> ... r")
        delta = einsum(h, layer.lora_B, "... r, o r -> ... o")
        return y + scale * delta
