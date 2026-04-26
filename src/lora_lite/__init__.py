import os as _os

# Optional runtime shape/dtype checking via jaxtyping + beartype.
# Set BEARTYPE=1 for smoke tests / debugging; off by default for zero overhead.
if _os.environ.get("BEARTYPE"):
    from beartype.claw import beartype_this_package as _bt
    _bt()

from .config import LoraLiteConfig
from .adapter import attach, detach, save, load
from .variant import REGISTRY, register, ParamSpec, Variant
from . import variants  # noqa: F401  triggers variant registration

__all__ = [
    "LoraLiteConfig",
    "attach",
    "detach",
    "save",
    "load",
    "REGISTRY",
    "register",
    "ParamSpec",
    "Variant",
]
