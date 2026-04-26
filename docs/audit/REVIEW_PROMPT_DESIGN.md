# Design review: should lora-lite borrow from peft EVA / baukit / antipasto3?

You are reviewing a minimal from-scratch LoRA library (`lora-lite`) and comparing
it to three reference implementations. Goal: identify cherry-picks that would
**reduce** complexity or unlock missing capability, **without bloating the lib**.

## Inputs

- lora-lite code: `src/lora_lite/` (adapter.py, target.py, variant.py, config.py, variants/*.py)
- Reference: `docs/refs/peft_eva.py` (peft's EVA: data-driven SVD-of-activations init)
- Reference: `docs/refs/peft_eva_finetuning.py` (example usage)
- Reference: `docs/refs/baukit_nethook.py` (nethook: forward/backward hook patterns)
- Reference: `docs/refs/antipasto3_svd_adapter.py` (wassname's earlier JAX SVD adapter)

## Project ethos (read first)

Lora-lite is fail-fast research code. Principles:
- No defensive programming, no fallbacks, no legacy compat
- Simplicity beats features. If you add X you must remove equivalent complexity.
- Each variant is one file with paper URL + honest deviation notes.
- Targets discovered by structural type-check, not name regex.
- Hooks via plain torch forward_pre_hook on a single layer, no global registry.

Read `AGENTS.md` if present.

## Questions to answer

For each reference, answer:

### A. peft EVA (`docs/refs/peft_eva.py` + `peft_eva_finetuning.py`)

1. What does EVA actually do? (1-paragraph summary; cite line numbers)
2. What would a *minimal* EVA variant in lora-lite look like? Sketch the API:
   - How does the user pass calibration data?
   - Where does the SVD-of-activations happen — in `init()` with a callback,
     or as a separate `calibrate(model, dataloader, cfg)` step before `attach`?
3. Does peft's implementation have anything we could **drop** if we re-implemented?
   (e.g. the rank-redistribution logic, the resume-from-checkpoint plumbing)
4. Does lora-lite's current `Variant.init(layer, cfg)` signature support EVA, or
   would we need to extend it? Recommend the **smallest** API change.

### B. baukit nethook (`docs/refs/baukit_nethook.py`)

1. What does `TraceDict` / `Trace` give us that our current per-layer
   `forward_pre_hook` registration does not?
2. Would adopting `baukit` for hook management (a) simplify our adapter.py,
   (b) complicate it, or (c) be neutral? Quote specific lines from
   `src/lora_lite/adapter.py` to justify.
3. Lora-lite's principle: minimize deps. Is baukit worth a dep? Or should
   we just **inline** the 1-2 useful patterns?

### C. antipasto3 SVD adapter (`docs/refs/antipasto3_svd_adapter.py`)

1. This is the user's earlier JAX work. Anything in there (init style, scale
   parameterization, save/load format) that lora-lite should adopt or
   deliberately diverge from?
2. Does it suggest a cleaner factoring for PiSSA-like methods?

## Output format

For each (A, B, C), end with:

**Recommendation: ADOPT / SKIP / PARTIAL**

If ADOPT or PARTIAL, list the specific lines/patterns to import and the
approximate net line-count impact on lora-lite (+ added, − removed).

## Hard rules

- Do NOT propose code edits. This is design notes only.
- Do NOT recommend adding a feature unless you can name what to remove or
  simplify in exchange.
- Be specific. "Could be cleaner" is not a recommendation; "Replace L42-L67
  in adapter.py with a 5-line TraceDict call" is.
- If a reference's pattern is worse than what lora-lite already has, say so.
