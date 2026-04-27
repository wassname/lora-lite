"""Variant protocol + registry. Variants own (x, layer.weight, layer.lora_*) -> y_new."""
from dataclasses import dataclass
from typing import Callable, Protocol, Any
import torch
from torch import nn

from .config import AdapterConfig


@dataclass
class ParamSpec:
    shape: tuple[int, ...]
    init: str | Callable[[torch.Tensor], None] = "near_zero"  # 'zeros'|'near_zero'|'kaiming'|'ones'|callable(t)
    trainable: bool = True
    as_buffer: bool = False  # if True, register_buffer instead of register_parameter

    def _empty(self, dtype: torch.dtype, device) -> torch.Tensor:
        t = torch.empty(self.shape, dtype=dtype, device=device)
        if callable(self.init):
            self.init(t)
        elif self.init == "zeros":
            t.zero_()
        elif self.init == "near_zero":
            # avoid exact-zero dead zone; N(0, 1e-4) is small enough to be
            # ~identity but nonzero so gradients always have somewhere to go
            t.normal_(0, 1e-4)
        elif self.init == "near_one":
            # avoid exact-one dead zone; 1 + N(0, 1e-4)
            t.fill_(1.0).add_(torch.randn_like(t).mul_(1e-4))
        elif self.init == "ones":
            t.fill_(1.0)
        elif self.init == "kaiming":
            # match nn.Linear default: kaiming_uniform_(a=sqrt(5))
            nn.init.kaiming_uniform_(t, a=5 ** 0.5) if t.ndim >= 2 else t.normal_(0, 0.02)
        else:
            raise ValueError(f"unknown init: {self.init}")
        return t

    def make(self, dtype: torch.dtype, device) -> nn.Parameter:
        # legacy entry: returns a Parameter (used for trainable adapter params)
        if self.as_buffer:
            raise RuntimeError("as_buffer spec must be installed via register_buffer; see adapter.attach")
        return nn.Parameter(self._empty(dtype, device), requires_grad=self.trainable)

    def make_tensor(self, dtype: torch.dtype, device) -> torch.Tensor:
        # returns a raw tensor for buffer registration
        return self._empty(dtype, device)


class Variant(Protocol):
    name: str

    @staticmethod
    def param_specs(d_in: int, d_out: int, cfg: AdapterConfig) -> dict[str, ParamSpec]: ...

    @staticmethod
    def init(layer: nn.Linear, cfg: AdapterConfig) -> None: ...

    @staticmethod
    def forward(layer: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the layer's NEW output (not just delta).
        Additive variants: `return y + delta`.
        Replacing variants: ignore `y`, return new value."""
        ...


REGISTRY: dict[str, type] = {}


def register(cls):
    if not getattr(cls, "name", None):
        raise ValueError(f"variant {cls} missing .name")
    REGISTRY[cls.name] = cls
    return cls
