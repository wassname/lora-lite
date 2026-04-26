# lora-lite

A hackable, single-file-per-variant LoRA library built on PyTorch forward hooks.

The goal is not to be PEFT-compatible. The goal is to make adapter ideas easy to read, edit, test, and throw away.

## Install

```bash
pip install -e .
```

## Quickstart

```python
import torch, lora_lite as ll

model = MyTransformer()  # any nn.Module containing linear-like children
cfg = ll.LoraLiteConfig(variant="lora", r=8, alpha=16, dtype=torch.bfloat16)
ll.attach(model, cfg)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
# train...

ll.save(model, "adapter.pt")
ll.detach(model)
ll.load(model, "adapter.pt")
```

Inspect a live adapter tensor directly:

```python
A = model.layers[5].self_attn.q_proj.lora_A
```

## Core idea

Each variant owns the adapter math. The runtime only finds target layers, attaches `lora_*` parameters, registers hooks, and saves full-path adapter keys.

```python
def attach(model, cfg):
    targets ← find_linear_like_modules(model, cfg)
    freeze(model.parameters())
    for name, layer in targets:
        layer.lora_* ← variant.param_specs(layer, cfg)
        hook(layer, lambda x, y: variant.forward(layer, x, y))

def save(model, path):
    torch.save({"cfg": cfg, "state": state_dict_keys_containing("lora_")}, path)
```

Minimal by design:

- One file per variant.
- No module replacement, merge/unmerge, mixed-adapter batches, or PEFT config soup.
- LoRA/DeLoRA hooks work with `nn.Linear` and bnb-style `Linear{4bit,8bitLt}` modules that expose `in_features`, `out_features`, and `weight`.
- PiSSA is fp-only because it mutates `weight` into `W_res`; quantized PiSSA should fail loudly until dequantize/requantize is explicit.

Currently shipped variants:

| Variant | Class | File |
|---|---|---|
| LoRA | A (additive) | [src/lora_lite/variants/lora.py](src/lora_lite/variants/lora.py) |
| PiSSA ([Meng+ 2024](https://arxiv.org/abs/2404.02948)) | A + B (special init mutates W) | [src/lora_lite/variants/pissa.py](src/lora_lite/variants/pissa.py) |
| DeLoRA ([Bini+ 2025](https://arxiv.org/abs/2503.18225)) | A (additive, normalised) | [src/lora_lite/variants/delora.py](src/lora_lite/variants/delora.py) |

See [docs/spec/20260426_lora_lite_plan.md](docs/spec/20260426_lora_lite_plan.md) for goals, status, and the current design plan. The original broader design was stress-tested against the [adapters_as_hypotheses](https://github.com/wassname/adapters_as_hypotheses) catalog (~26/27 variants covered with 3 small API tweaks).

## Targeting

By default we target linear-like modules (`in_features`, `out_features`, `weight`) whose shape matches a "reader" (`d_in == d_model`) or "writer" (`d_out == d_model`) role, excluding `lm_head` and `embed_tokens`. This structural test is what lets bnb Linear4bit/8bitLt modules be targeted without a backend-specific class. Knobs on `LoraLiteConfig`:

- `target_roles`: subset of `("reader", "writer", "inner")`. `()` = all.
- `target_names`: regex includes (must match if non-empty).
- `exclude_names`: regex excludes (default skips `lm_head`, `embed_tokens`).
- `layers`: tuple of layer indices, or `None` for all (matches `.layers.<idx>.` in module name).

## Variant API

A variant is a class with a `name` and three statics:

```python
@register
class MyVariant:
    name = "myvariant"

    @staticmethod
    def param_specs(d_in, d_out, cfg) -> dict[str, ParamSpec]:
        return {"lora_A": ParamSpec((cfg.r, d_in), init="kaiming"), ...}

    @staticmethod
    def init(layer, cfg) -> None:
        # Optional. Run after params are created. May read/mutate layer.weight.
        ...

    @staticmethod
    def forward(layer, x, y) -> Tensor:
        # Return the layer's NEW output (additive: `return y + delta`).
        ...
```

Adapter params attached as `layer.lora_*` get full-path keys in `state_dict()` automatically (e.g. `model.layers.5.self_attn.q_proj.lora_A`).

## Data-calibrated init

PiSSA, DeLoRA, and LoRA only use `layer.weight` for init -- no calibration data needed.

For variants that DO need data (e.g. AntiPaSTO, LoRA-GA, activation-aware SVD), keep dataloaders out of `cfg` so adapter checkpoints stay serializable. Use:

```python
ll.attach(model, cfg, calibration_data=calib)
```

where `calib` is an iterable of whole-model inputs, e.g. `Iterable[dict[str, Tensor]]` for HF models or `Iterable[Tensor]` of token ids. Activation-aware variants implement:

```python
@staticmethod
def group_init(model, targets, cfg, calibration_data): ...
```

`targets` is `list[(name, layer, role)]`. The variant adds temporary hooks, runs `model(batch)` over `calibration_data`, removes the hooks, then writes `lora_*` params. Per-layer `init(layer, cfg)` stays weight-only.

Sketch:

```python
@register
class ActSVD:
    name = "actsvd"
    @staticmethod
    def param_specs(d_in, d_out, cfg): ...
    @staticmethod
    def group_init(model, targets, cfg, calibration_data):
        bufs = {name: [] for name, _, _ in targets}
        hooks = [
            layer.register_forward_pre_hook(
                lambda m, args, name=name: bufs[name].append(args[0].detach().float())
            )
            for name, layer, _ in targets
        ]
        try:
            with torch.no_grad():
                for batch in calibration_data:
                    model(**batch) if isinstance(batch, dict) else model(batch)
        finally:
            for h in hooks:
                h.remove()
        # For each target: X = torch.cat(bufs[name], dim=0); do SVD; write A/B.
```

## Smoke test

```bash
just check       # pytest + smoke + package build
just test
just smoke
just qwen-queue  # queued Qwen/Qwen3-0.6B proof via pueue
```

`just test` verifies, for each of `lora`, `pissa`, `delora`:

1. Identity at t=0: `max|y_adapter - y_base|` within float tolerance.
2. Adapter hooks are live: perturbing only `lora_*` changes outputs.
3. Save/load round-trip preserves full-path adapter keys and tensors.
4. Missing or unexpected `lora_*` checkpoint keys fail loudly.
5. Only `lora_*` parameters are trainable and base parameters get no gradients.
6. A 20-step tiny regression training probe gets finite nonzero adapter gradients and >5% loss drop.

`just qwen-probe` is the real-model proof. It loads `Qwen/Qwen3-0.6B` fresh per variant, attaches only layer-0 `q_proj`/`v_proj`, trains one fixed LM batch, saves adapters, reloads into a fresh base model, and checks logits match. Last verified on 2026-04-26:

| variant | targets | trainable | identity err | perturb delta | loss0 | lossN | drop % | grad norm | adapter delta | reload err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 2 | 20,480 | 0 | 0.3750 | 5.250 | 3.131 | 40.36 | 1.432 | 4.262 | 0 |
| PiSSA | 2 | 20,480 | 0.3125 | 0.7500 | 5.250 | 3.629 | 30.88 | 6.124 | 4.381 | 0 |
| DeLoRA | 2 | 20,482 | 0.3750 | 0.4062 | 5.246 | 5.166 | 1.537 | 0.04778 | 8.196 | 0 |

This is an interface/training proof, not a benchmark: exact Qwen target names, hook activity, lora-only gradients, loss decrease, adapter tensor save/load, and reload equivalence on a 0.6B HF model.

CI runs `just check` on GitHub. The larger Qwen proof stays in `pueue` because it needs the shared GPU.

## What's NOT in v1

| Feature | Why dropped |
|---|---|
| merge/unmerge | reload base if you want vanilla |
| 4/8-bit-aware merge | DoRA on bnb supported in forward only (drop merge path) |
| Embedding / Conv adapters | trivial extension; add when needed |
| `adapter_names=` mixed batch forward | rare; add when needed |
| Multiple named adapters per layer | one variant per `attach()` |
| HF `PeftConfig` / hub upload | `torch.save({cfg, state})` is enough |
| AdaLoRA-style rank scheduling | needs `Variant.on_step(step)` -- punt |
| ReFT-style position interventions | sibling submodule (different hook site) |

## Status

v0.0.1: LoRA + PiSSA + DeLoRA + minimal functional tests + Qwen proof. Next likely variants are IA3 or DoRA because they fit the current hook contract with little new machinery. OFT/ROAD/AntiPaSTO-style methods are more interesting, but should wait until the simple hook family is boring.

## Citation

```bibtex
@misc{wassname2026loralite,
  title = {LoRA-Lite: A Hackable Adapter Library for Research},
  author = {Michael J. Clark},
  year = {2026},
  url = {https://github.com/wassname/lora-lite/}
}
```