set shell := ["bash", "-cu"]

default:
	@just --list

check: test smoke build

test:
	uv run --extra test pytest -q

smoke:
	uv run --extra test python tests/smoke.py

bnb-smoke:
	uv run --extra test --extra bnb-test python tests/smoke.py --require-bnb

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

metamath-queue-all model="Qwen/Qwen3-0.6B-Base" steps="5000" variants="lora pissa delora dora hra ia3 ia3_ff eva antipasto":
	#!/usr/bin/env bash
	set -euo pipefail
	for variant in {{variants}}; do
		lr=1e-4
		extra_args=(--target-name '(q_proj|v_proj)$' --layers all --r 32 --alpha 64)
		case "$variant" in
			delora)
				lr=1e-3
				;;
			ia3)
				lr=1e-3
				extra_args=(--target-name '(k_proj|v_proj)$' --layers all --r 32 --alpha 64)
				;;
			ia3_ff)
				lr=1e-3
				extra_args=(--target-name '(down_proj)$' --layers all --r 32 --alpha 64)
				;;
			eva)
				lr=1e-4
				;;
			antipasto)
				lr=1e-4
				;;
		esac
		pueue add \
			-l "why: benchmark {{model}} ${variant} on MetaMathQA->GSM8K at {{steps}} steps; resolve: outputs/metamath_gsm8k/results/benchmark_results.tsv gets a row with accuracy commit time method argv and result JSON for ${variant}" \
			-w "$PWD" -o 1 -- \
			uv run --extra benchmark python scripts/metamath_gsm8k_benchmark.py --model {{model}} --variant "$variant" --steps {{steps}} --lr "$lr" "${extra_args[@]}"
	done