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

| Variant | 4bit/8bit | MetaMath acc (GSM8K %) | Notes |
|---|---|---|---|
| [LoRA](https://arxiv.org/abs/2106.09685) | yes | 63.2% | |
| [PiSSA](https://arxiv.org/abs/2404.02948) | no (edits weight) | — | |
| [DeLoRA](https://arxiv.org/abs/2503.18225) | yes | — | |
| [IA3](https://arxiv.org/pdf/2205.05638) | yes | — | output gate (ia3) or input gate (ia3_ff) |
| [DoRA](https://arxiv.org/abs/2402.09353) | no (reads weight) | — | |
| [HRA](https://arxiv.org/abs/2409.01434) | yes | — | input-side Householder; works on bnb |
| [EVA](https://arxiv.org/abs/2409.07871) | no (calibration SVD) | — | |
| [AntiPaSTO](https://arxiv.org/abs/2503.08696) | no (reads weight SVD) | — | |

Our test setup: We take Qwen3-0.6B-Base and train one MetaMathQA for 5000 steps. We use a rank of 32, and itnervene on all linear layer then test on GSM8K.


Is this a good accuracy? TODO we need a like-for-like comparison against PEFT LoRA in the same setup before drawing conclusions. But the [PEFT library](https://github.com/huggingface/peft#results) reports LoRA at 49.0% on Llama-3.2-3B (different model and sample count).


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

