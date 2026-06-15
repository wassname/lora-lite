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
| [AntiPaSTO](https://arxiv.org/abs/2601.07473) | no        | 61.4%   | 14.3K      | 11.3          |
| AntiPaSTO-CorDA                               | no        | 61.9%   | 14.3K      | 11.3          |
| AntiPaSTO-ablate                              | no        | 61.0%   | 14.4K      | 11.3          |
| AntiPaSTO-arrow                               | no        | 60.5%   | 17.5K      | 11.3          |
| [IA3-FF](https://arxiv.org/pdf/2205.05638)    | yes       | 61.4%   | 86K        | 11.4          |
| [EVA](https://arxiv.org/abs/2410.07170)       | no        | 60.3%   | 4.59M      | 11.3          |
| [IA3](https://arxiv.org/pdf/2205.05638)       | yes       | 60.0%   | 57K        | 11.4          |
| [HRA](https://arxiv.org/abs/2405.17484)       | yes       | 61.6%   | 1.84M      | 11.3          |

Params = trainable adapter params. Peak GPU = peak CUDA memory during train+eval (logged from this run onward; older runs predate the column).

This is a validation harness, not a leaderboard: every adapter trains on a MetaMathQA subset and is tested on GSM8K, the same protocol as [PEFT's method comparison](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA). A correct implementation clears ~48% (PEFT's LoRA mark on the larger Llama-3.2-3B); every row here does, which is the signal that each adapter trains. See [the benchmark script](scripts/metamath_gsm8k_benchmark.py) for hyperparameters (r=32 q/v; the AntiPaSTO family uses r=256).

AntiPaSTO is the novel row here: instead of adding trainable directions like LoRA, it freezes W's own top-r SVD and learns only a bounded per-direction gain `S_eff = S * (1 + ELU(g))`. The singular basis stays fixed and interpretable, and the adapter is O(r) params (~320x smaller than LoRA). The variants change only the basis or core: CorDA orients it by input covariance ([Yang+ 2024](https://arxiv.org/abs/2406.05223)), ablate learns a contractive directional ablation ([Arditi+ 2024](https://arxiv.org/abs/2406.11717)), arrow adds a small dense block for cross-direction mixing.


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

