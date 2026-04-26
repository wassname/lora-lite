# 2026-04-26 code review: testing proof

## External review

Reviewer: Gemini 2.5 Flash CLI, read-only prompt.

Findings:

- Critical: `tests/smoke.py` could silently pass if base gradients leaked because it did not check non-`lora_*` grads.
- Important: `tests/smoke.py` did not explicitly assert the expected number of attached TinyModel targets.

Resolution:

- Added `assert_no_base_grads(model)` to the smoke training loop.
- Added `assert n_targets == 28` immediately after smoke attach.
- Re-ran `just test` and `just smoke`; both passed.

## Fresh-eyes subagent review

Verdict: PASS.

The reviewer could not name a remaining blocker for skipped targets, dead hooks, base-gradient leakage, or broken save/load producing the collected evidence. Caveat: Qwen coverage is intentionally narrow, layer-0 `q_proj`/`v_proj`, one prompt, tiny steps. This supports interface/training proof, not downstream finetuning quality.