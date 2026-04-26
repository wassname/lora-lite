# Developer guide

This is the implementation note for people adding adapter variants. The README is only for prospective users.

## Design principles

- Variants own adapter math.
- The runtime owns targeting, parameter attachment, hooks, and save/load.
- Adapter parameters live directly on target layers as `lora_*` parameters.
- Save/load uses normal full-path `state_dict()` keys filtered by `"lora_"`.
- Fail loudly on unsupported weight semantics. No silent quantized PiSSA or merge fallback.

## Variant contract

A variant is a registered class with a small static interface:

```python
@register
class MyVariant:
    name = "myvariant"

    @staticmethod
    def param_specs(d_in, d_out, cfg) -> dict[str, ParamSpec]:
        return {"lora_A": ParamSpec((cfg.r, d_in), init="kaiming")}

    @staticmethod
    def init(layer, cfg) -> None:
        ...

    @staticmethod
    def forward(layer, x, y):
        return y_new
```

Pseudocode for the runtime:

```python
def attach(model, cfg):
    targets ← find_linear_like_modules(model, cfg)
    freeze(model.parameters())
    for name, layer in targets:
        layer.lora_* ← variant.param_specs(layer, cfg)
        variant.init(layer, cfg)
        hook(layer, lambda x, y: variant.forward(layer, x, y))

def save(model, path):
    torch.save({"cfg": cfg, "state": state_dict_keys_containing("lora_")}, path)
```

## Data-calibrated init

LoRA, PiSSA, DeLoRA, and IA3 only use `layer.weight` or identity constants for init.

Variants that need data, e.g. AntiPaSTO, LoRA-GA, or activation-aware SVD, should keep dataloaders out of `cfg` so adapter checkpoints stay serializable:

```python
ll.attach(model, cfg, calibration_data=calib)
```

Activation-aware variants implement `group_init(model, targets, cfg, calibration_data)`. The variant may add temporary hooks, run calibration batches, remove hooks, then write `lora_*` params. `load()` should not require calibration data.

## Current limitations

| Feature | Current choice |
|---|---|
| merge/unmerge | reload the base model if vanilla weights are needed |
| multiple named adapters | one variant per `attach()` |
| mixed-adapter batches | out of scope until needed |
| quantized PiSSA | fail-fast; explicit dequantize/requantize required |
| AdaLoRA rank scheduling | needs a future `Variant.on_step(step)` hook |
| ReFT-style interventions | likely a sibling module or different hook site |

## Adapter roadmap

| Variant | Fit to current runtime | Next invariant |
|---|---|---|
| IA3 | Done. Output gate `y * g`, identity at `g=1`. | Qwen proof in latest probe. |
| DoRA | Done for fp layers. Reads dense `weight` to compute `||V||_c`; quantized layers fail fast. | Qwen proof in latest probe. |
| HRA | Done. Output-side Householder with identity gate; hook-only -> works on bnb. | Qwen proof in latest probe. |
| SSVD / PiSSA-family | Fits weight-SVD init path. | reconstruction/identity invariant plus train proof. |
| OFT / ROAD | Block-diagonal rotations; weight-transform semantics need clearer hook-only formulation. | pseudocode first, then rotation/non-dead-code invariant. |
| S-steer / AntiPaSTO | Should use `group_init` and activation evidence. | calibration consumed, hooks removed, load works without calibration. |
