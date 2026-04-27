set shell := ["bash", "-cu"]

default:
	@just --list

check: test smoke build

test:
	uv run --extra test --extra benchmark pytest -q

smoke:
	uv run --extra test --extra benchmark pytest -q tests/test_metamath_smoke.py -k test_metamath_quick_train_save_load

bnb-smoke:
	uv run --extra test --extra benchmark --extra bnb-test pytest -q tests/test_metamath_smoke.py -k test_attach_on_bnb_loaded_base

build:
	rm -rf dist
	uv build
	uv run --extra build twine check dist/*

qwen-probe variants="lora pissa delora ia3" steps="5":
	#!/usr/bin/env bash
	set -euo pipefail
	for variant in {{variants}}; do
		uv run --extra benchmark python scripts/metamath_gsm8k_benchmark.py \
			--mode probe \
			--model Qwen/Qwen3-0.6B-Base \
			--variant "$variant" \
			--steps {{steps}} \
			--batch-size 1 \
			--batch-size-eval 10 \
			--max-train-samples 32 \
			--max-eval-samples 10 \
			--max-new-tokens 32 \
			--max-seq-length 384 \
			--r 4 \
			--alpha 8 \
			--layers 0 \
			--lr 5e-3 \
			--target-name 'model\.layers\.0\.self_attn\.(q_proj|v_proj)$'
	done

qwen-queue variants="lora pissa delora ia3" steps="16":
	#!/usr/bin/env bash
	set -euo pipefail
	pueue add \
		-l "why: verify Qwen0.6B train/save-load proof for {{variants}} at {{steps}} steps via benchmark probe mode; resolve: publish only if exact layer0 q/v targets, lora-only grads, perturb>0, reload<tol" \
		-w "$PWD" -o 1 -- \
		just qwen-probe "{{variants}}" "{{steps}}"

metamath-smoke variant="lora" steps="2" max_train_samples="8" max_eval_samples="8" model="hf-internal-testing/tiny-random-LlamaForCausalLM" device="cpu":
	uv run --extra benchmark python scripts/metamath_gsm8k_benchmark.py \
		--model {{model}} \
		--variant {{variant}} \
		--steps {{steps}} \
		--batch-size 2 \
		--max-train-samples {{max_train_samples}} \
		--max-eval-samples {{max_eval_samples}} \
		--max-new-tokens 8 \
		--max-seq-length 128 \
		--r 2 \
		--alpha 4 \
		--layers 0 \
		--torch-dtype float32 \
		--device {{device}}

metamath-queue variant="lora" steps="5000" model="Qwen/Qwen3-0.6B-Base":
	#!/usr/bin/env bash
	set -euo pipefail
	pueue add \
		-l "why: HF-style MetaMathQA->GSM8K benchmark for {{model}} {{variant}} at {{steps}} steps; resolve: result JSON under outputs/metamath_gsm8k proves grad>0 dθ>0 base_grad_leaks=0 and reports valid/test accuracy" \
		-w "$PWD" -o 1 -- \
		uv run --extra benchmark python scripts/metamath_gsm8k_benchmark.py --model {{model}} --variant {{variant}} --steps {{steps}}

# Run a single MetaMathQA->GSM8K benchmark for a given variant.
# Per-variant lr / target-name defaults are baked in here.
bench-variant model variant steps="5000":
	#!/usr/bin/env bash
	set -euo pipefail
	lr=1e-4
	target='(q_proj|v_proj)$'
	case "{{variant}}" in
		delora) lr=1e-3 ;;
		ia3)    lr=1e-3; target='(k_proj|v_proj)$' ;;
		ia3_ff) lr=1e-3; target='(down_proj)$' ;;
	esac
	exec uv run --extra benchmark python scripts/metamath_gsm8k_benchmark.py \
		--model '{{model}}' \
		--variant '{{variant}}' \
		--steps {{steps}} \
		--lr "$lr" \
		--target-name "$target" \
		--layers all --r 32 --alpha 64

metamath-queue-all model="Qwen/Qwen3-0.6B-Base" steps="5000" variants="lora pissa delora dora hra ia3 ia3_ff eva antipasto":
	#!/usr/bin/env bash
	set -euo pipefail
	for variant in {{variants}}; do
		pueue add \
			-l "why: benchmark {{model}} ${variant} on MetaMathQA->GSM8K at {{steps}} steps; resolve: outputs/metamath_gsm8k/results/benchmark_results.tsv gets a row with accuracy commit time method argv and result JSON for ${variant}" \
			-w "$PWD" -o 1 -- \
			just bench-variant '{{model}}' "$variant" {{steps}}
	done