import os as _os

# Optional runtime shape/dtype checking via jaxtyping + beartype.
# Set BEARTYPE=1 for smoke tests / debugging; off by default for zero overhead.
if _os.environ.get("BEARTYPE"):
    from beartype.claw import beartype_this_package as _bt
    _bt()

from .config import AdapterConfig
from .adapter import attach, detach, save, load
from .variant import REGISTRY, register, ParamSpec, Variant
from . import variants  # noqa: F401  triggers variant + config registration

# Expose per-variant config classes for ergonomic typed construction.
from .variants.lora import LoRAConfig
from .variants.pissa import PiSSAConfig
from .variants.delora import DeLoRAConfig
from .variants.ia3 import IA3Config, IA3FFConfig
from .variants.dora import DoRAConfig
from .variants.hra import HRAConfig
from .variants.eva import EVAConfig
from .variants.antipasto import AntiPaSTOConfig

__all__ = [
    "AdapterConfig",
    "LoRAConfig",
    "PiSSAConfig",
    "DeLoRAConfig",
    "IA3Config",
    "IA3FFConfig",
    "DoRAConfig",
    "HRAConfig",
    "EVAConfig",
    "AntiPaSTOConfig",
    "attach",
    "detach",
    "save",
    "load",
    "REGISTRY",
    "register",
    "ParamSpec",
    "Variant",
]
