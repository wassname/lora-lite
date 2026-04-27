# lora-lite

Hackable PyTorch adapters for LoRA-family and small PEFT experiments.

## Hackable code


To keep it simple and hackable we make these choices:

- Simple forward hooks, no module replacement or custom modules.
- Simple code over fast performance
- No merge/unmerge
- Single test where we train on MetaMathQA and test on GSM8K for each variant

Take a look at [lora.py](src/lora_lite/variants/lora.py)

## Install

```bash
pip install -e git+https://github.com/wassname/lora-lite.git#egg=lora-lite
```

## Quickstart

```python
import torch, lora_lite as ll

model = MyTransformer()
cfg = ll.LoRAConfig(r=8, alpha=16, dtype=torch.bfloat16)
ll.attach(model, cfg)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
# train...

ll.save(model, "adapter.safetensors")
ll.detach(model)
ll.load(model, "adapter.safetensors")
```

## Does it work?

```bash
just check       # pytest + smoke + package build + metadata check
just bnb-smoke   # required CUDA bitsandbytes 4bit/8bit smoke
just qwen-probe  # Qwen/Qwen3-0.6B train/save-load probe
```

## Variants

| Variant                                       | 4bit/8bit | GSM8K % | Params     | Peak GPU (GB) |
| --------------------------------------------- | --------- | ------- | ---------- | ------------- |
| [LoRA](https://arxiv.org/abs/2106.09685)      | yes       | 63.2%   | 4.59M      | 11.3          |
| [PiSSA](https://arxiv.org/abs/2404.02948)     | no        | 63.2%   | 4.59M      | 11.3          |
| [DoRA](https://arxiv.org/abs/2402.09353)      | no        | 62.4%   | 4.67M      | 11.3          |
| [DeLoRA](https://arxiv.org/abs/2503.18225)    | yes       | 61.5%   | 4.59M      | 11.3          |
| [AntiPaSTO](https://arxiv.org/abs/2503.08696) | no        | 61.4%   | 35.8K      | 11.5          |
| [IA3-FF](https://arxiv.org/pdf/2205.05638)    | yes       | 61.4%   | 86K        | 11.4          |
| [EVA](https://arxiv.org/abs/2409.07871)       | no        | 60.3%   | 4.59M      | 11.3          |
| [IA3](https://arxiv.org/pdf/2205.05638)       | yes       | 60.0%   | 57K        | 11.4          |
| [HRA](https://arxiv.org/abs/2409.01434)       | yes       | 61.6%   | 1.84M      | 11.3          |

Params = trainable adapter params. Peak GPU = peak CUDA memory during train+eval (logged from this run onward; older runs predate the column).

Setup: Qwen3-0.6B-Base, MetaMathQA train (5k steps, batch 4 = 20k samples unless noted), r=32, all q/v targets, GSM8K test (1319 examples). HRA used batch 2 (10k samples) due to memory. AntiPaSTO used r=256 (default for this variant).

Reference: PEFT reports LoRA at 49.0% on Llama-3.2-3B (different model, different sample count). Our numbers are not directly comparable but suggest the adapters work.

AntiPaSTO at 59.5% with 4.5K trainable params (1000x fewer than LoRA's 4.59M). It trains singular-value deltas + block-Cayley rotation within the SVD subspace, so it can rescale and reorient existing directions but not create new ones. Higher rank (r>32) or data-driven dimension selection (from antipasto3) may close the gap further.


## Developer docs

See [docs/developer_guide.md](docs/developer_guide.md) for the variant API, data-calibrated init, and save/load format.

## Citation

```bibtex
@misc{wassname2026loralite,
  title = {LoRA-Lite: A Hackable Adapter Library for Research},
  author = {Michael J. Clark},
  year = {2026},
  url = {https://github.com/wassname/lora-lite/}
}
```

