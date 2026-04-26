"""Smoke: end-to-end MetaMath->GSM8K plumbing for every variant on a tiny HF model.

Per-variant correctness invariants live in tests/test_lora_lite.py. This script
just confirms the full benchmark pipeline (data load, prompt encode, train step,
eval generate + answer extract) runs for each adapter type.
"""
from __future__ import annotations

import subprocess
import sys

VARIANTS = ["lora", "pissa", "delora", "ia3", "ia3_ff", "dora", "hra", "eva", "antipasto"]
MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def run_one(variant: str) -> int:
    cmd = [
        sys.executable,
        "scripts/metamath_gsm8k_benchmark.py",
        "--model", MODEL,
        "--variant", variant,
        "--steps", "2",
        "--batch-size", "2",
        "--max-train-samples", "8",
        "--max-eval-samples", "10",
        "--max-valid-samples", "10",
        "--max-new-tokens", "8",
        "--max-seq-length", "128",
        "--r", "4",
        "--alpha", "8",
        "--torch-dtype", "float32",
        "--device", "cpu",
    ]
    if variant == "ia3":
        cmd += ["--target-name", r"(k_proj|v_proj)$"]
    elif variant == "ia3_ff":
        cmd += ["--target-name", r"(down_proj)$"]
    print(f"\n=== smoke variant={variant} ===")
    print(" ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    failed = [v for v in VARIANTS if run_one(v) != 0]
    if failed:
        print(f"FAIL: {failed}")
        return 1
    print("ALL PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
