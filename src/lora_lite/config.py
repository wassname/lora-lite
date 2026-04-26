from dataclasses import dataclass, field, asdict
from typing import Any
import torch


@dataclass
class LoraLiteConfig:
    variant: str = "lora"
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0  # currently ignored; variants may use cfg.variant_kwargs
    dtype: torch.dtype = torch.bfloat16

    # targeting
    target_roles: tuple[str, ...] = ("reader", "writer")
    target_names: tuple[str, ...] = ()
    exclude_names: tuple[str, ...] = ("lm_head", "embed_tokens")
    layers: tuple[int, ...] | None = None

    # variant-specific bag (e.g. lambda0 for DeLoRA)
    variant_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dtype"] = str(self.dtype).removeprefix("torch.")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LoraLiteConfig":
        d = dict(d)
        if isinstance(d.get("dtype"), str):
            d["dtype"] = getattr(torch, d["dtype"])
        if isinstance(d.get("layers"), list):
            d["layers"] = tuple(d["layers"])
        for k in ("target_roles", "target_names", "exclude_names"):
            if isinstance(d.get(k), list):
                d[k] = tuple(d[k])
        return cls(**d)
