from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

import lora_lite as ll


PROMPT = "LoRA-lite probe: Paris is the capital of France. The answer is"
EXPECTED_TARGETS = {
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj",
}


def cfg_for_variant(variant: str, dtype: torch.dtype, r: int, alpha: float) -> ll.LoraLiteConfig:
    return ll.LoraLiteConfig(
        variant=variant,
        r=r,
        alpha=r if variant == "pissa" else alpha,
        dtype=dtype,
        target_roles=(),
        target_names=(r"model\.layers\.0\.self_attn\.(q_proj|v_proj)$",),
        layers=(0,),
        variant_kwargs={"lambda0": 0.1} if variant == "delora" else {},
    )


def adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def assert_only_lora_trainable(model: torch.nn.Module) -> None:
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable), trainable[:20]


def assert_no_base_grads(model: torch.nn.Module) -> None:
    leaked = [name for name, p in model.named_parameters() if "lora_" not in name and p.grad is not None]
    assert leaked == [], leaked[:20]


def perturb_first_adapter(model: torch.nn.Module) -> None:
    for name, p in model.named_parameters():
        if "lora_lambda" in name:
            with torch.no_grad():
                p.add_(0.25)
            return
    for name, p in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.flatten()[0].add_(0.25)
            return
    raise AssertionError("no perturbable adapter parameter found")


def load_model(model_id: str, dtype: torch.dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.config.use_cache = False
    return model


def run_variant(args, variant: str, input_ids: torch.Tensor, labels: torch.Tensor, dtype: torch.dtype):
    model = load_model(args.model, dtype, args.device)
    model.train()
    cfg = cfg_for_variant(variant, dtype, args.r, args.alpha)

    with torch.no_grad():
        logits_base = model(input_ids=input_ids).logits.detach().clone()

    ll.attach(model, cfg)
    attached_targets = set(getattr(model, "_lora_lite_attached")["targets"])
    assert attached_targets == EXPECTED_TARGETS, attached_targets
    assert_only_lora_trainable(model)

    with torch.no_grad():
        logits_init = model(input_ids=input_ids).logits.detach().clone()
    identity_err = (logits_init - logits_base).abs().max().item()

    clean_adapter = adapter_state(model)
    perturb_first_adapter(model)
    with torch.no_grad():
        perturb_delta = (model(input_ids=input_ids).logits - logits_init).abs().max().item()
    assert perturb_delta > 1e-7, perturb_delta
    for name, value in clean_adapter.items():
        model.state_dict()[name].copy_(value)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    with torch.no_grad():
        loss0 = model(input_ids=input_ids, labels=labels).loss.item()

    before_train = adapter_state(model)
    first_grad_norm = math.nan
    loss_last = math.nan
    for step in range(args.steps):
        opt.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        assert_no_base_grads(model)
        grad_norm = sum(
            p.grad.detach().float().norm().item()
            for name, p in model.named_parameters()
            if "lora_" in name and p.grad is not None
        )
        assert math.isfinite(grad_norm), grad_norm
        if step == 0:
            first_grad_norm = grad_norm
        opt.step()
        loss_last = loss.item()

    after_train = adapter_state(model)
    adapter_delta = sum((after_train[k] - before_train[k]).float().norm().item() for k in before_train)
    assert first_grad_norm > 0, first_grad_norm
    assert adapter_delta > 0, adapter_delta
    assert loss_last < loss0, (loss0, loss_last)

    model.eval()
    with torch.no_grad():
        logits_trained = model(input_ids=input_ids).logits.detach().clone()

    out_path = args.out_dir / f"{variant}_adapter.pt"
    ll.save(model, str(out_path))
    saved = torch.load(out_path, weights_only=True, map_location="cpu")
    assert set(saved["state"]) == set(after_train)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    loaded_model = load_model(args.model, dtype, args.device)
    loaded_model.eval()
    ll.load(loaded_model, str(out_path))
    loaded_state = adapter_state(loaded_model)
    for name, value in saved["state"].items():
        assert torch.equal(loaded_state[name].cpu(), value)
    with torch.no_grad():
        logits_loaded = loaded_model(input_ids=input_ids).logits.detach().clone()
    reload_err = (logits_loaded - logits_trained).abs().max().item()
    assert reload_err < args.reload_tol, reload_err

    del loaded_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "variant": variant,
        "targets": len(attached_targets),
        "trainable": sum(v.numel() for v in after_train.values()),
        "id_err": identity_err,
        "perturb": perturb_delta,
        "loss0": loss0,
        "lossN": loss_last,
        "drop%": 100 * (loss0 - loss_last) / loss0,
        "grad": first_grad_norm,
        "dθ": adapter_delta,
        "reload": reload_err,
        "out": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--variants", nargs="+", default=["lora", "pissa", "delora"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--reload-tol", type=float, default=2e-2)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/qwen_train_probe"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the default Qwen probe. Pass --device cpu explicitly for local debugging.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(args.device)
    labels = input_ids.clone()

    print("SHOULD: exact q_proj/v_proj layer-0 targets, lora-only grads, lossN<loss0, perturb>0, reload<tol. ELSE hook/target/train/save bug.")
    rows = [run_variant(args, variant, input_ids, labels, dtype) for variant in args.variants]
    print(tabulate(rows, headers="keys", tablefmt="tsv", floatfmt=".4g"))
    print("ALL QWEN PROBES PASS")


if __name__ == "__main__":
    main()