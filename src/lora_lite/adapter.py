"""attach / detach / save / load. The whole runtime."""
from __future__ import annotations
import json
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from .config import AdapterConfig
from .variant import REGISTRY
from .target import find_targets


_ATTACHED_ATTR = "_lora_lite_attached"


def _hook(layer, args, y):
    (x,) = args
    cfg: AdapterConfig = layer._lora_cfg
    x_cast = x.to(cfg.dtype)
    out = layer._lora_variant.forward(layer, x_cast, y)
    return out.to(y.dtype)


def _pre_hook(layer, args):
    (x,) = args
    cfg: AdapterConfig = layer._lora_cfg
    x_cast = x.to(cfg.dtype)
    x_new = layer._lora_variant.forward_input(layer, x_cast)
    return (x_new.to(x.dtype),)


def attach(model: nn.Module, cfg: AdapterConfig, calibration_data=None, *, _skip_group_init: bool = False) -> list[RemovableHandle]:
    if cfg.variant not in REGISTRY:
        raise KeyError(f"unknown variant {cfg.variant!r}; registered: {list(REGISTRY)}")
    variant = REGISTRY[cfg.variant]
    targets = find_targets(model, cfg)
    if not targets:
        raise RuntimeError("no target layers matched cfg")

    # freeze base
    for p in model.parameters():
        p.requires_grad_(False)

    handles: list[RemovableHandle] = []
    attached_names: list[str] = []
    attached_targets = []
    for name, layer, role in targets:
        d_in, d_out = layer.in_features, layer.out_features
        for pname, spec in variant.param_specs(d_in, d_out, cfg).items():
            if hasattr(layer, pname):
                raise RuntimeError(f"{name} already has attribute {pname}; detach first")
            if spec.as_buffer:
                t = spec.make_tensor(cfg.dtype, layer.weight.device)
                layer.register_buffer(pname, t, persistent=True)
            else:
                p = spec.make(cfg.dtype, layer.weight.device)
                layer.register_parameter(pname, p)
        layer._lora_cfg = cfg
        layer._lora_variant = variant
        layer._lora_role = role
        variant.init(layer, cfg)
        attached_names.append(name)
        attached_targets.append((name, layer, role))

    group_init = getattr(variant, "group_init", None)
    if group_init is not None and not _skip_group_init:
        group_init(model, attached_targets, cfg, calibration_data)

    for _, layer, _ in attached_targets:
        if hasattr(layer._lora_variant, "forward_input"):
            handles.append(layer.register_forward_pre_hook(_pre_hook))
        else:
            handles.append(layer.register_forward_hook(_hook))

    setattr(model, _ATTACHED_ATTR, {"cfg": cfg, "targets": attached_names, "handles": handles})
    return handles


def detach(model: nn.Module) -> None:
    state = getattr(model, _ATTACHED_ATTR, None)
    if state is None:
        return
    for h in state["handles"]:
        h.remove()
    # remove params + scratch attrs
    for name, layer in model.named_modules():
        if not hasattr(layer, "_lora_variant"):
            continue
        variant = layer._lora_variant
        for pname in variant.param_specs(layer.in_features, layer.out_features, layer._lora_cfg):
            if pname in layer._parameters:
                del layer._parameters[pname]
            elif pname in layer._buffers:
                del layer._buffers[pname]
        for attr in ("_lora_cfg", "_lora_variant", "_lora_role"):
            if hasattr(layer, attr):
                delattr(layer, attr)
    delattr(model, _ATTACHED_ATTR)


def _base_weight_fingerprint(model: nn.Module) -> dict[str, str]:
    """Per-target fingerprint of the (post-init) base weights so PiSSA-style
    variants that mutate `layer.weight` can fail loud on base mismatch.
    Uses a cheap fp32 sum-of-squares + shape signature; not cryptographic.
    """
    state = getattr(model, _ATTACHED_ATTR, None)
    if state is None:
        return {}
    fp = {}
    for name, layer in model.named_modules():
        if not hasattr(layer, "_lora_variant"):
            continue
        if name not in state["targets"]:
            continue
        w = layer.weight.detach().to(torch.float32, copy=False)
        fp[name] = f"{tuple(w.shape)}|{float((w * w).sum()):.6e}"
    return fp


def save(model: nn.Module, path: str) -> None:
    state = getattr(model, _ATTACHED_ATTR, None)
    if state is None:
        raise RuntimeError("no adapter attached; call attach() first")
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    metadata = {
        "cfg": json.dumps(state["cfg"].to_dict()),
        "base_fp": json.dumps(_base_weight_fingerprint(model)),
    }
    from safetensors.torch import save_file
    save_file(sd, path, metadata=metadata)


def load(model: nn.Module, path: str) -> list[RemovableHandle]:
    from safetensors.torch import load_file, safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    sd = load_file(path, device="cpu")
    cfg = AdapterConfig.from_dict(json.loads(metadata["cfg"]))
    handles = attach(model, cfg, _skip_group_init=True)  # creates empty params; data-driven inits restored from state_dict
    missing, unexpected = model.load_state_dict(sd, strict=False)
    expected_lora = {k for k in model.state_dict() if "lora_" in k}
    missing_lora = sorted(expected_lora.intersection(missing))
    if missing_lora:
        raise RuntimeError(f"missing lora keys in checkpoint: {missing_lora}")
    unexpected_lora = [k for k in unexpected if "lora_" in k]
    if unexpected_lora:
        raise RuntimeError(f"unexpected lora keys in checkpoint: {unexpected_lora}")
    saved_fp = json.loads(metadata.get("base_fp", "{}"))
    if saved_fp:
        cur_fp = _base_weight_fingerprint(model)
        diffs = [k for k in saved_fp if saved_fp[k] != cur_fp.get(k)]
        if diffs:
            raise RuntimeError(
                f"base weight fingerprint mismatch on {len(diffs)} layer(s) "
                f"(e.g. {diffs[0]}). For PiSSA the saved adapter assumes the same "
                "base; reload onto the original model or re-run init."
            )
    return handles
