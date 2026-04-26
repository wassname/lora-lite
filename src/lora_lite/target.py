"""Find linear-like targets by shape (reader/writer/inner) + name regex.

Structural matching is deliberate: bnb Linear4bit/8bitLt are not nn.Linear, but
they expose in_features/out_features/weight and their forward already handles
dequantization.
"""
import re
from torch import nn


def is_linear_like(m: nn.Module) -> bool:
    return (
        hasattr(m, "in_features")
        and hasattr(m, "out_features")
        and hasattr(m, "weight")
        and callable(m)
    )


def _layer_idx(name: str) -> int | None:
    m = re.search(r"\.layers?\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _classify(m: nn.Module, d_model: int, name: str) -> str:
    di, do = m.in_features, m.out_features
    if di == d_model and do != d_model:
        return "reader"
    if do == d_model and di != d_model:
        return "writer"
    if di == d_model and do == d_model:
        return "writer" if any(s in name for s in ("o_proj", "out_proj")) else "reader"
    return "inner"


def find_targets(model: nn.Module, cfg) -> list[tuple[str, nn.Module, str]]:
    # discover d_model: prefer config.hidden_size, else infer from largest Linear in_features
    d_model = getattr(getattr(model, "config", None), "hidden_size", None)
    if d_model is None:
        dims = [m.in_features for m in model.modules() if is_linear_like(m)]
        d_model = max(dims) if dims else 0

    out = []
    for name, m in model.named_modules():
        if not is_linear_like(m):
            continue
        if any(re.search(p, name) for p in cfg.exclude_names):
            continue
        if cfg.layers is not None:
            li = _layer_idx(name)
            if li is None or li not in cfg.layers:
                continue
        role = _classify(m, d_model, name)
        if cfg.target_roles and role not in cfg.target_roles:
            continue
        if cfg.target_names and not any(re.search(p, name) for p in cfg.target_names):
            continue
        out.append((name, m, role))
    return out
