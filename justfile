set shell := ["bash", "-cu"]

default:
	@just --list

check: test smoke build

test:
	uv run --extra test pytest -q

smoke:
	uv run --extra test python tests/smoke.py

build:
	rm -rf dist
	uv build
	uv run --extra build twine check dist/*

qwen-probe variants="lora pissa delora" steps="8":
	uv run --extra test --extra hf-test python scripts/qwen_train_probe.py --variants {{variants}} --steps {{steps}}

qwen-queue variants="lora pissa delora" steps="16":
	#!/usr/bin/env bash
	set -euo pipefail
	pueue add \
		-l "why: verify Qwen0.6B train/save-load proof for {{variants}} at {{steps}} steps; resolve: publish docs only if exact targets, lora-only grads, loss drop, reload pass" \
		-w "$PWD" -o 1 -- \
		just qwen-probe "{{variants}}" "{{steps}}"