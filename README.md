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
| [IA3-FF](https://arxiv.org/pdf/2205.05638)    | yes       | 61.4%   | 86K        | 11.4          |
| [EVA](https://arxiv.org/abs/2410.07170)       | no        | 60.3%   | 4.59M      | 11.3          |
| [IA3](https://arxiv.org/pdf/2205.05638)       | yes       | 60.0%   | 57K        | 11.4          |
| [HRA](https://arxiv.org/abs/2405.17484)       | yes       | 61.6%   | 1.84M      | 11.3          |

Params = trainable adapter params. Peak GPU = peak CUDA memory during train+eval (logged from this run onward; older runs predate the column).

Setup: Qwen3-0.6B-Base, MetaMathQA train (5k steps, batch 4 = 20k samples unless noted), r=32, all q/v targets, GSM8K test (1319 examples). HRA used batch 2 (10k samples) due to memory. The AntiPaSTO family used r=256 (default for these variants).

Reference: PEFT reports LoRA at 49.0% on Llama-3.2-3B (different model, different sample count). Our numbers are not directly comparable but suggest the adapters work.

### AntiPaSTO family

AntiPaSTO learns a per-direction gain on the frozen top-r SVD basis (`S_eff = S * (1 + ELU(coeff*g))`), so it rescales existing singular directions rather than creating new ones, hence ~320x fewer trainable params than LoRA at ~97% of its accuracy. All variants share the diagonal gain; they differ only in the basis they steer in or the extra structure on the top directions. All have `base_grad_leaks=0` (the frozen residual weight gets no gradient).

| Variant            | GSM8K % | Params | Basis / extra structure                                              |
| ------------------ | ------- | ------ | ------------------------------------------------------------------- |
| antipasto_corda    | 61.9%   | 14.3K  | covariance-oriented input projector `P = Vh·C^{-1/2}` (best of family) |
| antipasto          | 61.4%   | 14.3K  | plain weight-SVD basis, diagonal gain only                          |
| antipasto_rot      | 61.4%   | 35.8K  | + block-Cayley rotation of the basis (the version this replaces)    |
| antipasto_ablate   | 61.0%   | 14.4K  | contractive output ablation `(I - α ĉĉᵀ)diag(S)`, can't amplify     |
| antipasto_arrow    | 60.5%   | 17.5K  | dense b×b mixing block on the top-b directions + diagonal tail      |

The rotation buys nothing here: `antipasto_rot` matches plain `antipasto` to 3 s.f. (61.4%) at 2.5x the params and +20% wall-time, while paying a per-forward Cayley solve. Dropping it (the current default) is free. CorDA's data-oriented basis is the only structure that helps on this capability task; the ablation/arrow cores are aimed at steering and suppression, where the diagonal-only basis can't reach off-axis behavior, so they don't pay off for raw GSM8K accuracy.


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

