set shell := ["bash", "-cu"]

# Base (NOT Instruct) text model: CorDA/PiSSA/ASVD decompose the pretrained weight and
# orient by calibration covariance -- the task must not be pre-baked by RLHF, or the
# variant differences ceiling out. AutoModelForCausalLM resolves Qwen3.5-0.8B-Base to
# the text-only Qwen3_5ForCausalLM (0.75B, no vision tower). It is a hybrid: 18 of 24
# layers are GatedDeltaNet (no q/v), 6 are full attention. So we target down_proj (dense
# nn.Linear in ALL 24 layers, d_in=3584) -- also CorDA/ASVD's canonical, highest-d_in target.
model := "Qwen/Qwen3.5-0.8B-Base"

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
			--model {{model}} \
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
			--target-name 'model\.layers\.0\.mlp\.down_proj$'
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

metamath-queue variant="lora" steps="5000" model=model:
	#!/usr/bin/env bash
	set -euo pipefail
	pueue add \
		-l "why: HF-style MetaMathQA->GSM8K benchmark for {{model}} {{variant}} at {{steps}} steps; resolve: result JSON under outputs/metamath_gsm8k proves grad>0 dθ>0 base_grad_leaks=0 and reports valid/test accuracy" \
		-w "$PWD" -o 1 -- \
		uv run --extra benchmark python scripts/metamath_gsm8k_benchmark.py --model {{model}} --variant {{variant}} --steps {{steps}}

# Run a single MetaMathQA->GSM8K benchmark for a given variant.
# Per-variant lr / target-name defaults are baked in here.
bench-variant model variant steps="5000" r_override="" lr_override="" rotate_basis="V" seed="0" target_override="":
	#!/usr/bin/env bash
	set -euo pipefail
	lr=1e-4
	# down_proj: dense nn.Linear in all 24 layers of the hybrid Qwen3.5 (q/v exist in
	# only 6 full-attention layers) and CorDA/ASVD's canonical highest-d_in target.
	target='(down_proj)$'
	r=32; alpha=64
	out_dir=outputs/metamath_gsm8k
	# IA3 lr: paper uses 3e-3 to 1e-2 (Liu et al. 2022 §3.3). Also a hard
	# bf16 floor: lora_g inits to 1.0 where bf16 spacing is ~7.8e-3, so
	# AdamW updates with lr<<3.9e-3 round back to 1.0 and the param freezes.
	# 5e-3 is paper-faithful AND clears the bf16 round-to-nearest threshold.
	case "{{variant}}" in
		delora) lr=1e-3 ;;
		ia3)    lr=5e-3; target='(k_proj|v_proj)$' ;;
		ia3_ff) lr=5e-3; target='(down_proj)$' ;;
		# antipasto tunes only S-space deltas + a small block rotation (tiny params),
		# so a small r leaves almost nothing trainable; r=256 is the variant default
		# and matches the published AntiPaSTO row. alpha=r (no extra scaling).
		antipasto) lr=5e-3; r=256; alpha=256 ;;
		# LoRA-XS trains only the r*r core R between frozen SVD factors. Ref LLaMA
		# math config sets lora_alpha=r (scale=1) and lr=4e-3 (run_math_tuning.sh);
		# keep r=32 to share the subspace dim with LoRA/PiSSA (all-else-equal rank axis).
		lora_xs) lr=4e-3; alpha=32 ;;
	esac
	# r override (e.g. low-rank sweep); alpha tracks r for antipasto.
	if [ -n "{{r_override}}" ]; then r="{{r_override}}"; alpha="{{r_override}}"; fi
	# lr override (e.g. a tamer lr than antipasto's 5e-3 default).
	if [ -n "{{lr_override}}" ]; then lr="{{lr_override}}"; fi
	# target override (e.g. LoRA-XS paper spreads across all q/k/v/o + FFN linears, not
	# just down_proj). run_id ignores target, so route overrides to their own dir to
	# avoid clobbering the canonical down_proj results the README table is built from.
	if [ -n "{{target_override}}" ]; then target="{{target_override}}"; out_dir=outputs/metamath_gsm8k_alllinear; fi
	# 0.8B + large vocab: HF ForCausalLMLoss upcasts logits to fp32 (bs*seq*vocab*4),
	# which OOMs the 24GB card at the old bs=4/seq=768. micro-batch 2 fits at ~10GB;
	# grad-accum 4 -> effective batch 8 (optimization quality without the memory).
	# expandable_segments curbs fragmentation. Same for all variants -> fair comparison.
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	exec uv run --extra benchmark python scripts/metamath_gsm8k_benchmark.py \
		--model '{{model}}' \
		--variant '{{variant}}' \
		--steps {{steps}} \
		--lr "$lr" \
		--target-name "$target" \
		--batch-size 2 --grad-accum 4 --max-seq-length 512 --batch-size-eval 16 \
		--layers all --r "$r" --alpha "$alpha" \
		--antipasto-rotate-basis '{{rotate_basis}}' \
		--output-dir "$out_dir" \
		--seed {{seed}}

metamath-queue-all model=model steps="2500" variants="lora pissa delora dora hra ia3 ia3_ff eva antipasto":
	#!/usr/bin/env bash
	set -euo pipefail
	# One pueue job per variant (each runs the live code at run time, so editing
	# while queued is safe). Re-queue here whenever the base model changes.
	for variant in {{variants}}; do
		pueue add \
			-l "why: benchmark {{model}} ${variant} on MetaMathQA->GSM8K at {{steps}} steps; resolve: outputs/metamath_gsm8k/results/benchmark_results.tsv gets a row with accuracy commit time method argv and result JSON for ${variant}" \
			-w "$PWD" -o 1 -- \
			just bench-variant '{{model}}' "$variant" {{steps}}
	done