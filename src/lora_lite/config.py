from dataclasses import dataclass, field, asdict
from typing import Any, Literal
import torch

Role = Literal["reader", "writer", "inner"]


@dataclass
class LoraLiteConfig:
    variant: str = "lora"
    r: int = 8
    alpha: float | int = 16.0
    dtype: torch.dtype = torch.bfloat16

    # targeting
    target_roles: tuple[Role, ...] = ("reader", "writer")
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
        # to_dict always serializes dtype as str; torch.save preserves tuples.
        # If you build the dict by hand, pass the right types -- fail loud otherwise.
        d = dict(d)
        d["dtype"] = getattr(torch, d["dtype"])
        return cls(**d)

