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

    # Register the adapter hooks BEFORE group_init. init() crops each weight to W_res,
    # so without the hooks the calibration forward inside group_init would run through a
    # model missing every target's top-r. At g=0 (and B=0) the hooks reconstruct the
    # cropped component exactly, so calibration sees the true full W.
    for _, layer, _ in attached_targets:
        if hasattr(layer._lora_variant, "forward_input"):
            handles.append(layer.register_forward_pre_hook(_pre_hook))
        else:
            handles.append(layer.register_forward_hook(_hook))

    group_init = getattr(variant, "group_init", None)
    ran_data_init = group_init is not None and not _skip_group_init and calibration_data is not None
    if group_init is not None and not _skip_group_init:
        group_init(model, attached_targets, cfg, calibration_data)

    # A data-driven group_init (CorDA orient, Wanda re-select) rewrites the frozen
    # base residual W_res into a form init() cannot reproduce at load time (it only
    # knows the plain top-r crop). So those residuals are part of the saved adapter.
    base_weight_keys = [f"{n}.weight" for n in attached_names] if ran_data_init else []
    setattr(model, _ATTACHED_ATTR,
            {"cfg": cfg, "targets": attached_names, "handles": handles,
             "base_weight_keys": base_weight_keys})
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
        # Undo the PiSSA-style crop: init() set weight = W - U_r S_r (Vh|P)_r, so add the
        # frozen top-r back to recover the original W (the trained gain/core are dropped).
        # Keyed on the shared SVD-gain buffer convention (antipasto family); variants
        # without lora_U leave weight untouched (e.g. LoRA never cropped it).
        if hasattr(layer, "lora_U"):
            proj = layer.lora_P if hasattr(layer, "lora_P") else layer.lora_Vh
            with torch.no_grad():
                layer.weight.data += ((layer.lora_U * layer.lora_S) @ proj).to(layer.weight.dtype)
        for pname in variant.param_specs(layer.in_features, layer.out_features, layer._lora_cfg):
            if pname in layer._parameters:
                del layer._parameters[pname]
            elif pname in layer._buffers:
                del layer._buffers[pname]
        for attr in ("_lora_cfg", "_lora_variant", "_lora_role"):
            if hasattr(layer, attr):
                delattr(layer, attr)
    delattr(model, _ATTACHED_ATTR)


def save(model: nn.Module, path: str) -> None:
    state = getattr(model, _ATTACHED_ATTR, None)
    if state is None:
        raise RuntimeError("no adapter attached; call attach() first")
    full_sd = model.state_dict()
    sd = {k: v.detach().cpu() for k, v in full_sd.items() if "lora_" in k}
    # data-driven variants also persist their rewritten base residuals (see attach()).
    base_weight_keys = state.get("base_weight_keys", [])
    for wk in base_weight_keys:
        sd[wk] = full_sd[wk].detach().cpu()
    metadata = {"cfg": json.dumps(state["cfg"].to_dict()),
                "base_weight_keys": json.dumps(base_weight_keys)}
    from safetensors.torch import save_file
    save_file(sd, path, metadata=metadata)


def load(model: nn.Module, path: str) -> list[RemovableHandle]:
    from safetensors.torch import load_file, safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    sd = load_file(path, device="cpu")
    cfg = AdapterConfig.from_dict(json.loads(metadata["cfg"]))
    # Base residuals a data-driven group_init rewrote: must be in the checkpoint and
    # are restored by load_state_dict below (init()'s plain crop would be wrong).
    base_weight_keys = json.loads(metadata.get("base_weight_keys", "[]"))
    missing_base = [wk for wk in base_weight_keys if wk not in sd]
    if missing_base:
        raise RuntimeError(f"checkpoint declares but omits base residuals: {missing_base}")
    handles = attach(model, cfg, _skip_group_init=True)  # creates empty params; data-driven inits restored from state_dict
    missing, unexpected = model.load_state_dict(sd, strict=False)
    expected_lora = {k for k in model.state_dict() if "lora_" in k}
    missing_lora = sorted(expected_lora.intersection(missing))
    if missing_lora:
        raise RuntimeError(f"missing lora keys in checkpoint: {missing_lora}")
    unexpected_lora = [k for k in unexpected if "lora_" in k]
    if unexpected_lora:
        raise RuntimeError(f"unexpected lora keys in checkpoint: {unexpected_lora}")
    # Carry the residual keys onto the attach state so a later save() re-persists them.
    getattr(model, _ATTACHED_ATTR)["base_weight_keys"] = base_weight_keys
    return handles
