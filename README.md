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
`down_proj` in all 24 layers (2500 steps, effective batch 8 = 20k samples). Standard adapters
use r=32; the AntiPaSTO family uses r=256 (it tunes only S-space gain, so it needs the rank).

| Variant                                       | test % | valid % | Params  | +MACs/tok | fwd/bwd (ms) | init (s) |
| --------------------------------------------- | -----: | ------: | ------: | --------: | -----------: | -------: |
| [LoRA](https://arxiv.org/abs/2106.09685)      |   59.8 |    68.0 |   3.54M |     3.54M |    173 / 573 |     0.02 |
| [PiSSA](https://arxiv.org/abs/2404.02948)     |   59.8 |    76.0 |   3.54M |     3.54M |    146 / 549 |     2.04 |
| [DoRA](https://arxiv.org/abs/2402.09353)      |   60.2 |    68.0 |   3.56M |     3.54M |    161 / 556 |     0.16 |
| [DeLoRA](https://arxiv.org/abs/2503.18225)    |   56.2 |    62.0 |   3.54M |     3.54M |    169 / 593 |     0.21 |
| [HRA](https://arxiv.org/abs/2405.17484)       |   59.2 |    70.0 |   2.75M |     2.75M |    225 / 948 |     0.04 |
| [EVA](https://arxiv.org/abs/2410.07170)       |   59.3 |    74.0 |   3.54M |     3.54M |    151 / 660 |    28.3 |
| [IA3](https://arxiv.org/pdf/2205.05638)       |   52.3 |    62.0 |  0.0061M |        0M |    161 / 515 |    0.01 |
| [IA3-FF](https://arxiv.org/pdf/2205.05638)    |   56.3 |    62.0 |  0.086M |        0M |    140 / 510 |    0.01 |
| [AntiPaSTO](https://arxiv.org/abs/2601.07473) |   56.0 |    62.0 |  0.0061M |    28.3M |    166 / 571 |      2.5 |
| AntiPaSTO-rot                                 |   57.2 |    60.0 |  0.0154M |    28.3M |    165 / 596 |      2.0 |
| AntiPaSTO-CorDA (full C)                      |   54.7 |    58.0 |  0.0061M |    28.3M |    146 / 576 |      120 |
| AntiPaSTO-ASVD (diag C)                       |   55.6 |    64.0 |  0.0061M |    28.3M |    150 / 533 |       34 |
| AntiPaSTO-ablate                              |   56.0 |    68.0 |  0.0062M |    28.3M |    166 / 580 |      2.2 |
| AntiPaSTO-dplr                                |   56.0 |    64.0 |  0.1044M |    28.4M |    153 / 582 |      3.6 |

test/valid % = GSM8K exact-match accuracy. Params = trainable adapter params. +MACs/tok = added
forward MACs per token (analytic, hardware-independent). fwd/bwd = median ms over one batch.
init = one-time calibration (CorDA's `d_in x d_in` covariance eigh; ~0 for the rest). Peak CUDA
memory is ~9.8 GB for every row. Empty rows fill in as the sweep lands.

We validate our adapters the same way [PEFT](https://github.com/huggingface/peft/tree/main/method_comparison) does: train on a MetaMathQA subset and check meaningful GSM8K accuracy. See [this file](scripts/metamath_gsm8k_benchmark.py) for details.

AntiPaSTO is the novel row here: instead of adding trainable directions like LoRA, it freezes W's own top-r SVD and learns only a bounded per-direction gain `S_eff = S * (1 + ELU(g))`. The singular basis stays fixed and interpretable, and the adapter is O(r) params (the 6.1K gain is ~580x smaller than LoRA's 3.54M). The variants change only the basis or core: rot learns a small block-rotation of the frozen basis, CorDA/ASVD orient it by the input second moment (full covariance vs diagonal-only, [Yang+ 2024](https://arxiv.org/abs/2406.05223) / [Yuan+ 2023](https://arxiv.org/abs/2312.05821)), ablate learns a contractive directional ablation ([Arditi+ 2024](https://arxiv.org/abs/2406.11717)), dplr adds a small low-rank core for cross-direction mixing.

CorDA (full C) and ASVD (diag C) are a metric-axis ablation against plain AntiPaSTO (C=I): does
covariance orientation earn its `d_in x d_in` eigh over the cheap diagonal or no calibration at
all? On GSM8K/down_proj the answer is no: C=I 56.0, diag C 55.6, full C 54.7 (single seed). The
off-diagonal orientation is the slowest arm (120 s init vs 2.5 s) and lands slightly *below* no
calibration, so plain top-r SVD is the right default for this bounded-gain adapter here.

AntiPaSTO-rot tunes that basis instead of the metric: a block-diagonal Cayley rotation of the
input (V), output (U), or both. The table row is V (the default); the ablation gives V 57.2 >
U 56.5 > both 55.6 (single seed). So rotating which inputs feed each frozen direction helps most,
the output-side rotation is slightly worse, and doing both is worst -- the second rotation is
redundant capacity that hurts. rot(V) is the best small-parameter arm overall (57.2 at 15K params
vs LoRA's 59.8 at 3.54M).


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

