from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import fcntl
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from tabulate import tabulate
from tqdm.auto import tqdm

import lora_lite as ll


PROMPT = "Question: {query} Think step by step.\nAnswer:"
DEFAULT_TARGETS = (r"(q_proj|v_proj)$",)

CFG_BY_VARIANT = {
    "lora": ll.LoRAConfig,
    "pissa": ll.PiSSAConfig,
    "delora": ll.DeLoRAConfig,
    "ia3": ll.IA3Config,
    "ia3_ff": ll.IA3FFConfig,
    "dora": ll.DoRAConfig,
    "hra": ll.HRAConfig,
    "eva": ll.EVAConfig,
    "antipasto": ll.AntiPaSTOConfig,
    "road": ll.RoadConfig,
}


@dataclass
class BenchmarkConfig:
    """MetaMathQA -> GSM8K benchmark config. Tyro turns this into the CLI."""

    model: str = "Qwen/Qwen3-0.6B-Base"
    variant: Literal["lora", "pissa", "delora", "ia3", "ia3_ff", "dora", "hra", "eva", "antipasto", "road"] = "lora"
    mode: Literal["benchmark", "probe"] = "benchmark"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    quantization: Literal["none", "4bit", "8bit"] = "none"
    r: int = 32
    alpha: float = 64.0
    delora_lambda0: float = 0.1
    road_group_size: int = 64
    target_name: list[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    layers: str = "all"
    train_dataset: str = "meta-math/MetaMathQA"
    eval_dataset: str = "openai/gsm8k"
    eval_config: str = "main"
    steps: int = 5000
    batch_size: int = 4
    batch_size_eval: int = 50
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_valid_samples: int | None = 50
    max_test_samples: int | None = None
    max_seq_length: int = 768
    max_new_tokens: int = 300
    lr: float = 1e-4
    weight_decay: float = 0.1
    grad_norm_clip: float = 1.0
    seed: int = 0
    log_examples: int = 3
    log_every: int = 250
    reload_tol: float = 2e-2
    output_dir: Path = Path("outputs/metamath_gsm8k")


def config_json(args: BenchmarkConfig) -> dict[str, Any]:
    data = asdict(args)
    data["output_dir"] = str(args.output_dir)
    return data


def normalize_number(text: str) -> str:
    return text.replace(",", "").strip().rstrip(".")


def extract_answer(text: str) -> str | None:
    tail = text.split("####")[-1]
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", tail)
    if not matches:
        return None
    return normalize_number(matches[-1])


def gsm8k_reference_answer(answer: str) -> str:
    extracted = extract_answer(answer)
    if extracted is None:
        raise ValueError(f"no numeric GSM8K reference answer in: {answer!r}")
    return extracted


def score_predictions(predictions: list[str], references: list[str]) -> dict[str, Any]:
    rows = []
    correct = 0
    for prediction, reference in zip(predictions, references, strict=True):
        pred_answer = extract_answer(prediction)
        ref_answer = gsm8k_reference_answer(reference)
        is_correct = pred_answer == ref_answer
        correct += int(is_correct)
        rows.append({"pred": pred_answer, "ref": ref_answer, "correct": is_correct})
    return {"accuracy": correct / len(rows), "correct": correct, "total": len(rows), "rows": rows}


def parse_layers(text: str) -> tuple[int, ...] | None:
    if text == "all":
        return None
    return tuple(int(part) for part in text.split(","))


def cfg_for_variant(args: BenchmarkConfig, dtype: torch.dtype) -> ll.AdapterConfig:
    extra = {"lambda0": args.delora_lambda0} if args.variant == "delora" else {}
    if args.variant == "road":
        extra = {"group_size": args.road_group_size}
    return CFG_BY_VARIANT[args.variant](
        r=args.r,
        alpha=args.r if args.variant == "pissa" else args.alpha,
        dtype=dtype,
        target_roles=(),
        target_names=tuple(args.target_name),
        layers=parse_layers(args.layers),
        **extra,
    )


def adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def assert_only_lora_trainable(model: torch.nn.Module) -> list[str]:
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise AssertionError("no trainable adapter parameters")
    if not all("lora_" in name for name in trainable):
        raise AssertionError(trainable[:20])
    return trainable


def count_base_grad_leaks(model: torch.nn.Module) -> int:
    return sum(1 for name, p in model.named_parameters() if "lora_" not in name and p.grad is not None)


def perturb_first_adapter(model: torch.nn.Module) -> None:
    priority = ("lora_B", "lora_g", "lora_U", "lora_A", "lora_lambda", "lora_gate", "lora_delta_s", "lora_rot_T", "lora_m", "lora_road_theta", "lora_road_alpha")
    for key in priority:
        for _, p in model.named_parameters():
            if p.requires_grad and key in _:
                with torch.no_grad():
                    if p.ndim == 0:
                        p.add_(0.25)
                    else:
                        p.flatten()[0].add_(0.25)
                return
    raise AssertionError("no perturbable adapter parameter found")


def load_model_and_tokenizer(model_id: str, dtype: torch.dtype, device: str, quantization: str = "none"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.eos_token is None:
        raise RuntimeError(f"tokenizer for {model_id} has no eos_token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if quantization == "none":
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    else:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_compute_dtype=dtype if quantization == "4bit" else None,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_cfg, device_map=device)
    model.config.use_cache = False
    return model, tokenizer


def load_datasets(args: BenchmarkConfig) -> dict[str, Any]:
    from datasets import load_dataset

    max_valid_samples = args.max_eval_samples or args.max_valid_samples
    max_test_samples = args.max_eval_samples or args.max_test_samples
    train = load_dataset(args.train_dataset, split="train").shuffle(seed=args.seed)
    if args.max_train_samples is not None:
        train = train.select(range(args.max_train_samples))
    gsm8k = load_dataset(args.eval_dataset, args.eval_config)
    valid = gsm8k["train"].shuffle(seed=args.seed)
    if max_valid_samples is not None:
        valid = valid.select(range(max_valid_samples))
    test = gsm8k["test"]
    if max_test_samples is not None:
        test = test.select(range(max_test_samples))
    return {"train": train, "valid": valid, "test": test}


def split_question_hashes(datasets: dict[str, Any], n: int = 3) -> dict[str, list[str]]:
    out = {}
    for split, dataset in datasets.items():
        key = "query" if split == "train" else "question"
        out[split] = [
            hashlib.sha1(dataset[i][key].encode("utf-8")).hexdigest()[:12]
            for i in range(min(n, len(dataset)))
        ]
    return out


def encode_train_example(tokenizer, query: str, response: str, max_seq_length: int) -> dict[str, torch.Tensor | int]:
    prompt = PROMPT.format(query=query)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + " " + response + tokenizer.eos_token, add_special_tokens=False).input_ids
    input_ids = full_ids[:max_seq_length]
    labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids):]
    labels = labels[:len(input_ids)]
    label_tokens = sum(label != -100 for label in labels)
    if label_tokens == 0:
        raise ValueError(f"no response labels left after truncation for query: {query[:120]!r}")
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_tokens": label_tokens,
    }


def pad_batch(examples: list[dict[str, torch.Tensor | int]], pad_token_id: int, device: str) -> dict[str, torch.Tensor | int]:
    max_len = max(len(example["input_ids"]) for example in examples)
    input_rows = []
    label_rows = []
    mask_rows = []
    label_tokens = 0
    for example in examples:
        ids = example["input_ids"]
        labels = example["labels"]
        pad = max_len - len(ids)
        input_rows.append(torch.cat([torch.full((pad,), pad_token_id), ids]))
        label_rows.append(torch.cat([torch.full((pad,), -100), labels]))
        mask_rows.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)]))
        label_tokens += int(example["label_tokens"])
    return {
        "input_ids": torch.stack(input_rows).to(device),
        "labels": torch.stack(label_rows).to(device),
        "attention_mask": torch.stack(mask_rows).to(device),
        "label_tokens": label_tokens,
    }


def make_train_batches(train_dataset, tokenizer, args: BenchmarkConfig) -> tuple[list[dict[str, torch.Tensor | int]], int]:
    needed = args.steps * args.batch_size
    examples = []
    skipped_prompt_too_long = 0
    for row in train_dataset:
        prompt = PROMPT.format(query=row["query"])
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if prompt_len >= args.max_seq_length:
            skipped_prompt_too_long += 1
            continue
        examples.append(encode_train_example(tokenizer, row["query"], row["response"], args.max_seq_length))
        if len(examples) == needed:
            break
    if len(examples) < needed:
        raise RuntimeError(
            f"only {len(examples)} usable train examples for {needed} requested "
            f"after skipping {skipped_prompt_too_long} prompt-too-long examples"
        )
    return [
        pad_batch(examples[i:i + args.batch_size], tokenizer.pad_token_id, args.device)
        for i in range(0, needed, args.batch_size)
    ], skipped_prompt_too_long


def cosine_lambda(step: int, total_steps: int) -> float:
    progress = min(step, total_steps) / total_steps
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(model: torch.nn.Module, batches: list[dict[str, torch.Tensor | int]], args: BenchmarkConfig) -> dict[str, float | int]:
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: cosine_lambda(step, args.steps))
    before = adapter_state(model)
    base_grad_leaks = 0
    first_grad_norm = math.nan
    first_loss = math.nan
    last_loss = math.nan
    train_total_tokens = 0
    probe_batch = batches[0]
    pbar = tqdm(batches, desc="train", mininterval=60.0, dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        opt.zero_grad()
        loss = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
        loss.backward()
        grad_norm = sum(
            p.grad.detach().float().norm().item()
            for name, p in model.named_parameters()
            if "lora_" in name and p.grad is not None
        )
        if step == 0:
            first_grad_norm = grad_norm
            first_loss = loss.item()
        base_grad_leaks += count_base_grad_leaks(model)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_norm_clip)
        opt.step()
        scheduler.step()
        last_loss = loss.item()
        train_total_tokens += int(batch["label_tokens"])
        pbar.set_postfix(loss=f"{last_loss:.4g}", grad=f"{grad_norm:.3g}", tok=train_total_tokens)
    pbar.close()
    after = adapter_state(model)
    adapter_delta = sum((after[k] - before[k]).float().norm().item() for k in before)
    model.eval()
    with torch.no_grad():
        probe_loss_after = model(
            input_ids=probe_batch["input_ids"],
            attention_mask=probe_batch["attention_mask"],
            labels=probe_batch["labels"],
        ).loss.item()
    if first_grad_norm <= 0 or not math.isfinite(first_grad_norm):
        raise AssertionError(f"bad first adapter grad norm: {first_grad_norm}")
    if adapter_delta <= 0:
        raise AssertionError(f"adapter did not move: {adapter_delta}")
    if base_grad_leaks != 0:
        raise AssertionError(f"base gradients leaked: {base_grad_leaks}")
    return {
        "train_loss_first": first_loss,
        "train_loss_last": last_loss,
        "train_loss_probe_after": probe_loss_after,
        "train_loss_probe_delta": probe_loss_after - first_loss,
        "first_grad_norm": first_grad_norm,
        "adapter_delta": adapter_delta,
        "base_grad_leaks": base_grad_leaks,
        "train_total_tokens": train_total_tokens,
    }


@torch.no_grad()
def evaluate(model, tokenizer, dataset, args: BenchmarkConfig, split: str) -> dict[str, Any]:
    model.eval()
    predictions = []
    references = []
    questions = []
    for start in range(0, len(dataset), args.batch_size_eval):
        rows = dataset[start:start + args.batch_size_eval]
        prompts = [PROMPT.format(query=q) for q in rows["question"]]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(args.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuations = generated[:, encoded.input_ids.shape[1]:]
        predictions.extend(tokenizer.batch_decode(continuations, skip_special_tokens=True))
        references.extend(rows["answer"])
        questions.extend(rows["question"])
    scored = score_predictions(predictions, references)
    scored["split"] = split
    scored["examples"] = [
        {
            "question": questions[i],
            "prediction": predictions[i],
            "pred_answer": scored["rows"][i]["pred"],
            "ref_answer": scored["rows"][i]["ref"],
            "correct": scored["rows"][i]["correct"],
        }
        for i in range(min(args.log_examples, len(predictions)))
    ]
    return scored


@torch.no_grad()
def probe_before_train(model, batch: dict[str, torch.Tensor | int], attached_targets: list[str]) -> dict[str, Any]:
    if not attached_targets:
        raise AssertionError("probe: no targets attached")
    logits_init = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.detach().clone()
    clean_adapter = adapter_state(model)
    perturb_first_adapter(model)
    perturb_delta = (model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits - logits_init).abs().max().item()
    if perturb_delta <= 1e-7:
        raise AssertionError(f"adapter perturbation did not affect logits: {perturb_delta}")
    for name, value in clean_adapter.items():
        model.state_dict()[name].copy_(value)
    return {"attached_targets": sorted(attached_targets), "perturb_delta": perturb_delta}


@torch.no_grad()
def check_probe_reload(
    args: BenchmarkConfig,
    cfg: ll.AdapterConfig,
    adapter_path: Path,
    batch: dict[str, torch.Tensor | int],
    logits_trained: torch.Tensor,
) -> dict[str, float | int]:
    del cfg  # cfg is saved in the checkpoint; keep the call-site explicit.
    gc.collect()
    torch.cuda.empty_cache()
    loaded_model, _ = load_model_and_tokenizer(args.model, getattr(torch, args.torch_dtype), args.device, args.quantization)
    loaded_model.eval()
    ll.load(loaded_model, str(adapter_path))
    from safetensors.torch import load_file
    saved_sd = load_file(str(adapter_path), device="cpu")
    loaded_state = adapter_state(loaded_model)
    if set(saved_sd) != set(loaded_state):
        raise AssertionError("loaded adapter keys differ from saved adapter keys")
    for name, value in saved_sd.items():
        if not torch.equal(loaded_state[name].cpu(), value):
            raise AssertionError(f"loaded adapter tensor differs: {name}")
    logits_loaded = loaded_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.detach().clone()
    reload_err = (logits_loaded - logits_trained).abs().max().item()
    if reload_err >= args.reload_tol:
        raise AssertionError(f"reload logits mismatch {reload_err} >= {args.reload_tol}")
    del loaded_model
    gc.collect()
    torch.cuda.empty_cache()
    return {"reload_err": reload_err, "saved_tensors": len(saved_sd)}


def print_final_report(row: dict[str, Any], result_path: Path, mode: str) -> None:
    # BLUF: status line first so log tails are immediately readable
    cue = "🟢" if row.get("base_grad_leaks", 0) == 0 and row.get("grad", 0) > 0 else "🔴"
    n = row.get("samples", "?")
    print()
    print(f"{cue} test_acc={row['test_acc']:.4g} valid_acc={row['valid_acc']:.4g} grad={row['grad']:.3g} dθ={row['dθ']:.3g} base_grad_leaks={row['base_grad_leaks']} N={n}")
    print("SHOULD: grad>0, dθ>0, base_grad_leaks=0; test/valid_acc meaningful only in benchmark mode. ELSE adapter or eval wiring is dead/wrong.")
    print()
    # ordered: most important / shortest columns first
    display_keys = ["variant", "test_acc", "valid_acc", "grad", "dθ", "base_grad_leaks", "steps", "samples", "loss0", "lossN", "commit"]
    if "perturb" in row:
        display_keys += ["perturb", "reload"]
    display_keys += ["run_id"]
    display_row = {k: row[k] for k in display_keys if k in row}
    print(tabulate([display_row], headers="keys", tablefmt="tsv", floatfmt=".4g"))
    print()
    print(f"argv: {' '.join(sys.argv)}  N={n} mode={mode}")
    print(f"out: {result_path}")


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def append_results_row(
    args: BenchmarkConfig,
    result_path: Path,
    result: dict[str, Any],
    run_commit: str,
) -> tuple[Path, Path]:
    results_dir = args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = results_dir / "summary.tsv"
    lock_path = results_dir / "summary.tsv.lock"
    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    finished_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = results_dir / f"{result['run_id']}__{finished_label}.json"
    snapshot_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    row = {
        "test_acc": result["test_acc"],
        "valid_acc": result["valid_acc"],
        "method": args.variant,
        "steps": args.steps,
        "samples": result["train_samples"],
        "model": args.model,
        "commit": run_commit[:12],
        "wall_time_s": round(result["wall_time_s"]),
        "time_utc": finished_at,
        "argv": " ".join(sys.argv),
        "result_json": str(snapshot_path),
        "latest_result_json": str(result_path),
    }
    header = "\t".join(row)
    values = "\t".join(str(value) for value in row.values())
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if not tsv_path.exists():
            tsv_path.write_text(header + "\n" + values + "\n", encoding="utf-8")
        else:
            with tsv_path.open("a", encoding="utf-8") as handle:
                handle.write(values + "\n")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return tsv_path, snapshot_path


def run(args: BenchmarkConfig) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu for plumbing smoke only")
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.torch_dtype)
    run_commit = current_git_commit()
    run_id = f"{args.model.replace('/', '--')}__{args.variant}__s{args.steps}__seed{args.seed}"
    out_dir = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(args)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype, args.device, args.quantization)
    batches, skipped_train_prompt_too_long = make_train_batches(datasets["train"], tokenizer, args)
    cfg = cfg_for_variant(args, dtype)
    if args.variant == "eva":
        calib = [
            {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
            for b in batches[: min(4, len(batches))]
        ]
        ll.attach(model, cfg, calibration_data=calib)
    else:
        ll.attach(model, cfg)
    attached = getattr(model, "_lora_lite_attached")
    trainable_names = assert_only_lora_trainable(model)
    probe_metrics = None
    if args.mode == "probe":
        probe_metrics = probe_before_train(model, batches[0], attached["targets"])
    model.train()

    started = time.time()
    train_metrics = train(model, batches, args)
    valid_metrics = evaluate(model, tokenizer, datasets["valid"], args, "valid")
    test_metrics = evaluate(model, tokenizer, datasets["test"], args, "test")

    adapter_path = out_dir / "adapter.safetensors"
    ll.save(model, str(adapter_path))
    if args.mode == "probe":
        model.eval()
        with torch.no_grad():
            logits_trained = model(input_ids=batches[0]["input_ids"], attention_mask=batches[0]["attention_mask"]).logits.detach().clone()
        probe_metrics |= check_probe_reload(args, cfg, adapter_path, batches[0], logits_trained)
    result = {
        "config": config_json(args),
        "run_id": run_id,
        "mode": args.mode,
        "model_id": args.model,
        "variant": args.variant,
        "r": args.r,
        "alpha": args.alpha,
        "target_names": list(args.target_name),
        "layers": args.layers,
        "attached_targets": attached["targets"],
        "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_param_names_sample": trainable_names[:20],
        "train_dataset": args.train_dataset,
        "eval_dataset": args.eval_dataset,
        "eval_config": args.eval_config,
        "dataset_fingerprints": {split: ds._fingerprint for split, ds in datasets.items()},
        "dataset_first_question_sha1": split_question_hashes(datasets),
        "dataset_sizes": {split: len(ds) for split, ds in datasets.items()},
        "skipped_train_prompt_too_long": skipped_train_prompt_too_long,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "batch_size_eval": args.batch_size_eval,
        "train_samples": args.steps * args.batch_size,
        "max_seq_length": args.max_seq_length,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_scheduler": "cosine",
        "grad_norm_clip": args.grad_norm_clip,
        "valid_acc": valid_metrics["accuracy"],
        "test_acc": test_metrics["accuracy"],
        "train": train_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
        "probe": probe_metrics,
        "adapter_path": str(adapter_path),
        "wall_time_s": time.time() - started,
    }
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.mode == "benchmark":
        results_tsv_path, result_snapshot_path = append_results_row(args, result_path, result, run_commit)
        result["results_tsv_path"] = str(results_tsv_path)
        result["result_snapshot_path"] = str(result_snapshot_path)
    result["commit"] = run_commit
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    row = {
        "run_id": run_id,
        "variant": args.variant,
        "steps": args.steps,
        "samples": args.steps * args.batch_size,
        "loss0": train_metrics["train_loss_first"],
        "lossN": train_metrics["train_loss_last"],
        "probeΔ": train_metrics["train_loss_probe_delta"],
        "grad": train_metrics["first_grad_norm"],
        "dθ": train_metrics["adapter_delta"],
        "base_grad_leaks": train_metrics["base_grad_leaks"],
        "valid_acc": valid_metrics["accuracy"],
        "test_acc": test_metrics["accuracy"],
        "commit": run_commit[:12],
        "result": str(result_path),
    }
    if probe_metrics is not None:
        row["perturb"] = probe_metrics["perturb_delta"]
        row["reload"] = probe_metrics["reload_err"]
    print_final_report(row, result_path, args.mode)
    return result


def main() -> None:
    import tyro

    args = tyro.cli(BenchmarkConfig)
    run(args)


if __name__ == "__main__":
    main()