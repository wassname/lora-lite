"""One-row-per-variant cost table: params, MACs/token, fwd/bwd ms, peak GPU, group_init.

Answers "which is best -- time / flops / adds / params?": MACs/token is the
deterministic apples-to-apples compute number; trainable_params is the size headline;
wall-time is the felt-but-noisy number; group_init is the one-time init cost.

Usage:
  uv run --extra benchmark python scripts/cost_report.py \
    --model Qwen/Qwen3-0.6B-Base --variants antipasto lora pissa \
    --target-name 'q_proj$' 'v_proj$' --r 32 --out logs/cost_qwen0.6b.log
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from tabulate import tabulate

import lora_lite as ll

_HERE = Path(__file__).resolve().parent
_BENCH = importlib.util.spec_from_file_location("metamath_benchmark", _HERE / "metamath_gsm8k_benchmark.py")
benchmark = importlib.util.module_from_spec(_BENCH)
sys.modules[_BENCH.name] = benchmark
_BENCH.loader.exec_module(benchmark)

_COST = importlib.util.spec_from_file_location("_cost", _HERE / "_cost.py")
cost = importlib.util.module_from_spec(_COST)
sys.modules[_COST.name] = cost
_COST.loader.exec_module(cost)


def build_cfg(variant: str, args, dtype) -> ll.AdapterConfig:
    """Reuse the benchmark's variant->config map; only need r/targets/dtype here."""
    bcfg = benchmark.BenchmarkConfig(
        model=args.model, variant=variant, r=args.r, alpha=float(args.r),
        target_name=list(args.target_name), layers=args.layers, torch_dtype=args.dtype,
    )
    return benchmark.cfg_for_variant(bcfg, dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--variants", nargs="+",
                    default=["lora", "pissa", "antipasto"])
    ap.add_argument("--target-name", nargs="+", default=[r"q_proj$", r"v_proj$"])
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--layers", default="all",
                    help="'all' or comma list e.g. '0,1' -- limit layers.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--calib-batches", type=int, default=4)
    ap.add_argument("--out", default="logs/cost.log")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    # eager attention: FlopCounterMode's sdpa_flop_count asserts on GQA (Qwen3) SDPA
    # shapes (q heads != kv heads). eager uses explicit matmuls it can count.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager"
    ).to(args.device)
    model.eval()

    n_tokens = args.batch * args.seq_len
    ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq_len), device=args.device)
    calib = [{"input_ids": torch.randint(0, model.config.vocab_size,
                                         (args.batch, args.seq_len), device=args.device)}
             for _ in range(args.calib_batches)]

    def fwd():
        model(input_ids=ids)

    def bwd_step():
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=ids).logits.float().pow(2).mean()
        loss.backward()

    # base (no-adapter) cost, so each row can report the adapter's ADDED MACs/token.
    base = cost.measure_cost(model, fwd, bwd_step_fn=bwd_step, n_tokens=n_tokens)
    base_macs = base["macs_per_token"]
    print(f"base (no adapter): MACs/tok={int(base_macs) if base_macs else None} "
          f"fwd_ms={round(base['fwd_ms'],2)} bwd_ms={round(base['bwd_ms'],2)}")

    # base = no adapter; model params left trainable, so this is the full-finetune
    # GPU-mem reference (its backward stores grads for every weight).
    total_params = sum(p.numel() for p in model.parameters())
    rows = [{
        "variant": "base(full-FT)", "train_params": total_params,
        "fwd_ms": round(base["fwd_ms"], 2), "bwd_ms": round(base["bwd_ms"], 2),
        "peak_GPU_MB": round(base["peak_gpu_mb"], 1) if base["peak_gpu_mb"] else None,
        "added_MACs/tok": 0 if base_macs else None,
        "ginit_ms": 0.0, "ginit_CPU_MB": 0.0,
    }]
    for variant in args.variants:
        cfg = build_cfg(variant, args, dtype)
        # group_init / attach cost (CorDA's eigh + C live here).
        with cost.group_init_meter() as gi:
            ll.attach(model, cfg, calibration_data=calib)
        c = cost.measure_cost(model, fwd, bwd_step_fn=bwd_step, n_tokens=n_tokens)
        ll.detach(model)
        rows.append({
            "variant": variant,
            "train_params": c["trainable_params"],
            "fwd_ms": round(c["fwd_ms"], 2),
            "bwd_ms": round(c["bwd_ms"], 2) if c["bwd_ms"] else None,
            "peak_GPU_MB": round(c["peak_gpu_mb"], 1) if c["peak_gpu_mb"] else None,
            # flat across same-r adapters; kept only as a sanity check, not a comparator.
            "added_MACs/tok": int(c["macs_per_token"] - base_macs) if (c["macs_per_token"] and base_macs) else None,
            "ginit_ms": round(gi.ms, 1),
            "ginit_CPU_MB": round(gi.peak_cpu_mb, 1),
        })
        print(f"  {variant}: params={rows[-1]['train_params']} "
              f"peak_GPU_MB={rows[-1]['peak_GPU_MB']} bwd_ms={rows[-1]['bwd_ms']} ginit_ms={rows[-1]['ginit_ms']}")

    table = tabulate(rows, headers="keys", tablefmt="pipe")
    header = (f"# cost report: {args.model}  targets={args.target_name}  r={args.r}  "
              f"seq={args.seq_len} batch={args.batch} dtype={args.dtype}\n"
              f"# COMPARATORS: train_params, peak_GPU_MB (fwd+bwd, process-local max), bwd_ms, ginit_ms.\n"
              f"# added_MACs/tok is flat across same-r adapters (sanity check only).\n"
              f"# ginit_CPU_MB undercounts: tracemalloc misses torch C++ tensor allocs (the CorDA C matrix).\n")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + table + "\n")
    print("\n" + header + table)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
