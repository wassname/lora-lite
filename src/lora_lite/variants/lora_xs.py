"""LoRA-XS: freeze W's top-r SVD as A,B; train only a small r x r matrix R between them.

Bałazy et al. 2024  https://arxiv.org/abs/2405.17604

    W = U S Vh                          (truncated to top-r)
    A = diag(Sr) Vhr   (r, d_in)        frozen  -- singular values folded into A (ref)
    B = Ur             (d_out, r)       frozen
    R                  (r, r)           trainable, ~0 at init
    h = W x + (alpha/r) B R A x

Unlike PiSSA, W is NOT cropped: B@A reconstructs the top-r but stays *added on top* of
the full W, and R (init normal(0, 1e-5)) starts the adapter at ~identity. So the only
trainable tensor is r*r (e.g. r=32 -> 1024 params/layer), hence "extremely small".

The reference folds all singular values into A and leaves B as the raw left singular
vectors. So R sits between B = Ur (orthonormal) and A = diag(Sr) Vhr (orthonormal rows
*scaled* by the singular values, so row norms = Sr, not unit) -- the asymmetry is the
reference's, not a bug. Their LLaMA math-tuning config sets lora_alpha = r (scale = 1.0)
and lr ~ 4e-3 (scripts/run_math_tuning.sh).

Refs:
  - paper repo: https://github.com/MohammadrezaBanaei/LoRA-XS
    (utils/initialization_utils.py: init_module_weights(R, sigma=1e-5), A/B requires_grad=False;
     utils/latent_utils.py forward_latent: result += scaling * lora_B(R(lora_A(x))))
"""
import torch
from jaxtyping import Float
from torch import nn, Tensor as T
from dataclasses import dataclass

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config


@register_config
@dataclass
class LoRAXSConfig(AdapterConfig):
    variant: str = "lora_xs"


@register
class LoRAXS:
    name = "lora_xs"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        return dict(
            # Frozen top-r SVD factors of W (filled in init()); W itself stays intact.
            lora_A=ParamSpec((cfg.r, d_in), init="zeros", trainable=False, as_buffer=True),
            lora_B=ParamSpec((d_out, cfg.r), init="zeros", trainable=False, as_buffer=True),
            # The only trainable tensor: r x r core, near-zero so the adapter ~ identity at t=0
            # (ref uses normal(0, 1e-5); matches the repo's near_zero philosophy).
            lora_R=ParamSpec((cfg.r, cfg.r), init=lambda t: t.normal_(0, 1e-5)),
        )

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        if type(layer) is not nn.Linear:
            raise TypeError(
                "LoRA-XS needs the dense SVD of layer.weight, so v1 only supports plain "
                "nn.Linear, not bnb 4/8-bit."
            )
        W = layer.weight.data.float()                       # (d_out, d_in)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = cfg.r
        Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
        # A = diag(Sr) Vhr, B = Ur  ->  B@A = Ur diag(Sr) Vhr = top-r(W). W is left intact.
        A = (Sr[:, None] * Vhr).to(cfg.dtype)
        B = Ur.to(cfg.dtype)
        layer.lora_A.copy_(A)
        layer.lora_B.copy_(B)

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        scale = cfg.alpha / cfg.r
        A = layer.lora_A                                    # (r, d_in), frozen
        B = layer.lora_B                                    # (d_out, r), frozen
        R = layer.lora_R                                    # (r, r), trainable
        xA = x.to(A.dtype)
        h = xA @ A.T                                        # (*B, r)
        h = h @ R.T                                         # (*B, r)   <- the learned core
        delta = h @ B.T                                     # (*B, d_out)
        return y + (scale * delta).to(y.dtype)
