# lora-lite

Hackable PyTorch adapters for LoRA-family and small PEFT experiments.

`lora-lite` uses forward hooks instead of module replacement. Adapter parameters are plain `nn.Parameter`s on the target layer, e.g. `model.layers[5].self_attn.q_proj.lora_A`.

## Install

```bash
pip install -e git+https://github.com/wassname/lora-lite.git#egg=lora-lite
```

## Quickstart

```python
import torch, lora_lite as ll

model = MyTransformer()
cfg = ll.LoraLiteConfig(variant="lora", r=8, alpha=16, dtype=torch.bfloat16)
ll.attach(model, cfg)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
# train...

ll.save(model, "adapter.pt")
ll.detach(model)
ll.load(model, "adapter.pt")
```

## Does it work?

```bash
just check       # pytest + smoke + package build + metadata check
just bnb-smoke   # required CUDA bitsandbytes 4bit/8bit smoke
just qwen-probe  # Qwen/Qwen3-0.6B train/save-load probe
```

See [docs/spec/20260426_lora_lite_plan.md](docs/spec/20260426_lora_lite_plan.md) for verification history and exact results.

## Variants

| Variant | Support | Notes |
|---|---|---|
| LoRA | yes | additive low-rank adapter |
| PiSSA | yes, fp only | mutates `weight` into `W_res`; quantized PiSSA intentionally fails |
| DeLoRA | yes | normalized additive adapter with learned scalar |
| IA3 | yes | output gate initialized to ones |
| DoRA | no | next small candidate |
| SSVD / OFT / HRA / ROAD | no | planned after the hook-only invariant is clear |
| S-steer / AntiPaSTO | no | should use data-calibrated `group_init`, not plain LoRA tests |

## Targeting

By default, `lora-lite` targets linear-like modules with `in_features`, `out_features`, and `weight`, excluding `lm_head` and `embed_tokens`.

Useful `LoraLiteConfig` fields:

- `target_roles`: subset of `("reader", "writer", "inner")`; `()` means all.
- `target_names`: regex includes.
- `exclude_names`: regex excludes.
- `layers`: layer indices, matching `.layers.<idx>.` in module names.

This structural targeting is why LoRA, DeLoRA, and IA3 can run on bnb-style `Linear4bit`/`Linear8bitLt` modules. PiSSA is different because it edits the base weight.

## Save format

Adapters are just:

```python
torch.save({"cfg": cfg.to_dict(), "state": lora_state_dict}, "adapter.pt")
```

`lora_state_dict` contains full-path keys with `"lora_"` in the name. Missing or unexpected adapter keys fail on load.

## Developer docs

See [docs/developer_guide.md](docs/developer_guide.md) for the variant API, data-calibrated init, and adapter roadmap.

## Citation

```bibtex
@misc{wassname2026loralite,
  title = {LoRA-Lite: A Hackable Adapter Library for Research},
  author = {Michael J. Clark},
  year = {2026},
  url = {https://github.com/wassname/lora-lite/}
}
```
