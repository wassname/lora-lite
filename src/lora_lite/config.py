"""AdapterConfig: per-variant typed dataclass.

Replaces the older `LoraLiteConfig` + `variant_kwargs` dict. Each variant
ships its own subclass under `variants/*.py` (e.g. `DeLoRAConfig`), adding
strongly-typed fields so users discover the knobs via IDE / dataclass
introspection instead of stringly-typed dict lookups.

Wire-up:
  - `AdapterConfig` holds the universal fields (variant name, rank, alpha,
    dtype, targeting filters).
  - Subclasses override the `variant` default and add new fields.
  - `register_config(cls)` adds the subclass to `_CONFIG_REGISTRY` so
    `from_dict` can route to the right type at load time.

Save format:
  to_dict() emits a flat dict including `variant`; from_dict() uses that
  field to look up the right subclass.
"""
from dataclasses import dataclass, asdict
from typing import Literal
import torch

Role = Literal["reader", "writer", "inner"]


@dataclass
class AdapterConfig:
    """Base config. Subclass per variant; do not instantiate directly."""
    # variant name (subclass overrides default)
    variant: str = "?"

    # rank-style hyperparams shared across most variants
    r: int = 8
    alpha: float | int = 16.0
    dtype: torch.dtype = torch.bfloat16

    # targeting
    target_roles: tuple[Role, ...] = ("reader", "writer")
    target_names: tuple[str, ...] = ()
    exclude_names: tuple[str, ...] = ("lm_head", "embed_tokens")
    layers: tuple[int, ...] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dtype"] = str(self.dtype).removeprefix("torch.")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AdapterConfig":
        d = dict(d)
        name = d["variant"]
        sub = _CONFIG_REGISTRY[name]
        d["dtype"] = getattr(torch, d["dtype"])
        return sub(**d)


# Registry of variant_name -> config subclass. Populated by `register_config`
# decorators in each `variants/*.py` module at import time.
_CONFIG_REGISTRY: dict[str, type[AdapterConfig]] = {}


def register_config(cls: type[AdapterConfig]) -> type[AdapterConfig]:
    """Decorator: register `cls` under its `variant` default value."""
    name = cls.__dataclass_fields__["variant"].default
    if name in _CONFIG_REGISTRY:
        raise ValueError(f"config for variant {name!r} already registered")
    _CONFIG_REGISTRY[name] = cls
    return cls

