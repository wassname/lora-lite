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

Trained on a MetaMathQA subset, tested on GSM8K, all on `Qwen/Qwen3.5-0.8B-Base` targeting
`down_proj` in all 24 layers (2500 steps, effective batch 8 = 20k samples).

| Variant                                       |    r | test % | valid % |  Params | +MACs/tok | fwd/bwd (ms) | init (s) |
| --------------------------------------------- | ---: | -----: | ------: | ------: | --------: | -----------: | -------: |
| [DoRA](https://arxiv.org/abs/2402.09353)      |   32 |   60.2 |    68.0 |   3.56M |     3.54M |    161 / 556 |     0.16 |
| [LoRA](https://arxiv.org/abs/2106.09685)      |   32 |   59.8 |    68.0 |   3.54M |     3.54M |    173 / 573 |     0.02 |
| [PiSSA](https://arxiv.org/abs/2404.02948)     |   32 |   59.8 |    76.0 |   3.54M |     3.54M |    146 / 549 |     2.04 |
| [EVA](https://arxiv.org/abs/2410.07170)       |   32 |   59.3 |    74.0 |   3.54M |     3.54M |    151 / 660 |     28.3 |
| [HRA](https://arxiv.org/abs/2405.17484)       |   32 |   59.2 |    70.0 |   2.75M |     2.75M |    225 / 948 |     0.04 |
| [AntiPaSTO](https://arxiv.org/abs/2601.07473) |  256 |   57.2 |    60.0 |  0.015M |     28.3M |    165 / 596 |      2.0 |
| [IA3-FF](https://arxiv.org/pdf/2205.05638)    |    — |   56.3 |    62.0 |  0.086M |        0M |    140 / 510 |     0.01 |
| [DeLoRA](https://arxiv.org/abs/2503.18225)    |   32 |   56.2 |    62.0 |   3.54M |     3.54M |    169 / 593 |     0.21 |
| [IA3](https://arxiv.org/pdf/2205.05638)       |    — |   52.3 |    62.0 |  0.006M |        0M |    161 / 515 |     0.01 |

r = adapter rank (— = not a low-rank method). test/valid % = GSM8K exact-match accuracy. Params =
trainable adapter params. +MACs/tok = added forward MACs per token (analytic, hardware-independent).
fwd/bwd = median ms over one batch. init = one-time calibration (EVA's PCA; ~0 for the rest). Peak
CUDA memory is ~9.8 GB for every row. Single seed, so accuracy differences within ~1.4pp (test
SE at n=1319) are noise.

We validate our adapters the same way [PEFT](https://github.com/huggingface/peft/tree/main/method_comparison) does: train on a MetaMathQA subset and check meaningful GSM8K accuracy. See [this file](scripts/metamath_gsm8k_benchmark.py) for details.

AntiPaSTO is the novel row here: instead of adding trainable directions like LoRA, it freezes W's own top-r SVD and learns only a per-direction singular-value delta plus a block-diagonal Cayley rotation of that frozen basis. The singular directions stay interpretable and the adapter is tiny (15K params, ~230x smaller than LoRA's 3.54M) yet stays within noise of the full-rank adapters. The default rotates the input basis (V); rotating the output (U), both, or neither are `rotate_basis` ablation axes.

The full AntiPaSTO family (rotation-free gain core, the U/both rotation arms, contractive directional ablation [Arditi+ 2024](https://arxiv.org/abs/2406.11717), a low-rank mixing core, and CorDA/ASVD covariance-oriented bases [Yang+ 2024](https://arxiv.org/abs/2406.05223) / [Yuan+ 2023](https://arxiv.org/abs/2312.05821)) lives on the [`antipasto-variants`](https://github.com/wassname/lora-lite/tree/antipasto-variants) branch with its own ablation table. On GSM8K/down_proj none of those arms separated from this one (the covariance-oriented bases cost 34-120 s of init for no gain; the V/U/both rotation order flips between two seeds, so the basis is within noise), so main keeps one cheap arm: rotation of V.


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

