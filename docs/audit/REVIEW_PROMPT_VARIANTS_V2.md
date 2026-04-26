# Per-variant paper-faithfulness audit V2 (with reference implementations)

Re-audit of `lora-lite` after adding canonical reference implementation URLs to
each variant docstring. Your job: for each variant, **directly compare** our
implementation against the reference impl (peft and/or paper-author repo), not
just against the paper text. This is round 2 — the previous review (you can
read `docs/audit/variants_review.md`) found:

- HRA gate=0 init kills `lora_U` gradient on step 0
- DeLoRA same pattern with lambda0=0
- IA3 targets q/v not paper k/v/ffn-down (deviation documented but untested)
- PiSSA bf16 init err 0.31 on Qwen
- Saved adapters don't preserve PiSSA W_res mutation

Your job now is to verify those findings against the **reference code**, and
look for anything the prior review missed once you have the reference in hand.

## Inputs

- Our code: `src/lora_lite/variants/{lora,pissa,dora,ia3,hra,delora}.py`
- Adapter plumbing: `src/lora_lite/{adapter.py,target.py,variant.py,config.py}`
- Papers (text): `docs/papers/*_*.txt`
- **Reference implementations** (just added):
  - `docs/refs/peft_lora_layer.py`   — peft LoRA Linear (and PiSSA init paths)
  - `docs/refs/peft_lora_dora.py`    — peft DoRA helper module
  - `docs/refs/peft_lora_variants.py` — peft per-variant init dispatch (PiSSA, OLoRA, etc.)
  - `docs/refs/peft_ia3_layer.py`    — peft IA3 layer
  - `docs/refs/peft_hra_layer.py`    — peft HRA layer (clean, has apply_GS toggle)
  - `docs/refs/peft_delora_layer.py` — peft DeLoRA layer (upstreamed)
  - `docs/refs/orig_pissa_init.py`   — PiSSA paper authors' init script (MuLabPKU)
  - `docs/refs/orig_hra_layer.py`    — HRA paper authors' OFT-with-HRA layer (DaShenZi721)
  - `docs/refs/orig_delora.py`       — DeLoRA paper authors' fork-of-peft impl (ExplainableML)
- Logs: `logs/smoke.log`, `logs/qwen_probe.log`
- Prior review: `docs/audit/variants_review.md` (do NOT just restate it)

## What to deliver per variant (LoRA, PiSSA, DoRA, IA3, HRA, DeLoRA)

1. **Reference impl ground-truth** — what does the *reference* code actually do
   for: parameter shapes, initialization, scale factor, forward equation,
   save/load, target placement? Quote ≤10 lines with file/line cites from
   `docs/refs/`.

2. **Our code** — quote our impl (≤10 lines, with `src/lora_lite/variants/<v>.py:LN` cites).

3. **Diff** — bullet list of every meaningful difference.
   Mark each one as: `[OK-doc]` (acceptable, documented), `[OK-undoc]` (acceptable,
   should add to docstring), `[BUG]` (likely wrong), `[STYLE]` (cosmetic).

4. **Did the prior review get it right?** Quote the relevant prior verdict
   line and either confirm or correct.

5. **Verdict** — Faithful / Faithful-with-doc-gap / Partial / Buggy.
   One-line reason.

## Final aggregate

Markdown table:

| variant | prior verdict | new verdict | new bugs found | doc gaps |

And a 5-bullet "what to fix next" list, ordered by severity.

## Hard rules

- Quote evidence from `docs/refs/` files. If you can't find the relevant
  reference function, say so explicitly — don't guess.
- Do NOT edit code. Output review only.
- Be specific about line numbers from the references. "peft does X" is not
  enough; "peft_lora_layer.py:L1234 does X" is.
- If you find a NEW bug not flagged in `variants_review.md`, mark it
  `[NEW-BUG]` and explain the failure mode.
- If the prior review was wrong (false positive), mark it `[OVERTURN]`.

Write to stdout. I will redirect to `docs/audit/variants_review_v2.md`.
