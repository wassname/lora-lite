set shell := ["bash", "-cu"]

default:
	@just --list

test:
	uv run --extra test pytest -q

smoke:
	uv run --extra test python tests/smoke.py

qwen-probe:
	uv run --extra test --extra hf-test python scripts/qwen_train_probe.py