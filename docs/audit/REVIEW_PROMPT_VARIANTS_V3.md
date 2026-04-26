# V3 Variant Review — per-component audit

You are an expert ML engineer reviewing a from-scratch PEFT library
(`lora-lite`, ~500 LOC) that re-implements 8 LoRA variants. Two prior reviews
already happened (V1 paper-vs-code, V2 with reference implementations
provided). Your job is V3: a tight per-component audit focused on
correctness-of-mechanism rather than overall design.

# Scope

8 variants live in `src/lora_lite/variants/`:
- lora.py
- pissa.py
- delora.py
- ia3.py (two registered: `ia3` and `ia3_ff`)
- dora.py
- hra.py
- eva.py        (NEW since V2)
- antipasto.py  (NEW since V2)

Plus the runtime in `src/lora_lite/{adapter.py,variant.py,target.py,config.py}`
and the smoke test in `tests/smoke.py`.

Reference implementations are in `docs/refs/` and the URLs are also pasted in
each variant's module docstring. Compare against those.

# What I want from you (per variant, in this order, every time)

For EACH variant, work through these five checkpoints, using only that
variant's file and its referenced peft/author code:

1. **PARAMS** — list every spec returned by `param_specs`. For each:
   shape, dtype (cfg.dtype unless overridden), trainable, as_buffer.
   Does the shape match the reference impl? Are buffers vs Parameters
   chosen correctly (no Parameter that should be a buffer; no buffer
   that we want to learn)? Does as_buffer mean it persists in
   state_dict (check `register_buffer(..., persistent=True)` in
   adapter.py)?

2. **INIT** — what does `init()` (and `group_init()` if defined) do?
   Does it match the reference exactly? Pay special attention to
   ZERO INITS — they often kill gradient flow on dependent params.
   Walk the gradient: at t=0, given this init, which trainable params
   actually receive non-zero gradient on the first SGD step?
   Are dtype casts placed correctly (fp32 SVD, then to cfg.dtype)?

3. **DTYPE** — trace dtype through init -> param storage -> forward.
   Where could silent precision loss happen? Is bf16 or fp16 used
   anywhere it shouldn't be? Does identity-at-init survive bf16?

4. **FORWARD** — write the math the forward implements, in the same
   convention as the reference (peft/author paper). Compare term by
   term. Common mistakes to look for:
   - wrong scale (alpha/r vs 1/r vs alpha vs 1)
   - missing or doubled normalization
   - wrong basis (rotating U vs V; gating input vs output)
   - dropout placement (we have no dropout — flag if any variant
     references one; see config.py)

5. **LINK SANITY** — open the URLs in the docstring. Verify:
   - the paper arxiv link goes to the right paper
   - the github link points to a real file
   - the offline `docs/refs/` snapshot matches what the URL serves
     today (the snapshots may be stale; if so, flag the drift)

# Output format

For each variant, write at most ~60 lines. Use this template:

    ## <variant>

    ### params
    - <one bullet per ParamSpec; flag bug if any>

    ### init / group_init
    - <bullets; identify GRADIENT FLOW at t=0 explicitly>

    ### dtype
    - <bullets>

    ### forward
    Math:    <one-line equation in our convention>
    Ref math: <one-line equation in reference convention>
    Match?   YES / NO + one-line explanation

    ### links
    - paper: OK / WRONG / DEAD
    - peft ref: OK / DEAD
    - author ref (if any): OK / DEAD
    - offline snapshot drift: NONE / MINOR / MAJOR

    ### verdict
    CORRECT / PARTIAL / BUGGY  -- one-sentence reason

After all variants, write a "## summary" with a markdown table of verdicts and
a numbered list of MUST-FIX bugs (severity high) vs nice-to-haves.

# Hard rules

- Be specific. Cite line numbers (`src/lora_lite/variants/foo.py:NN`) for
  every claim.
- Do NOT propose redesigns. Only flag correctness issues against the
  references.
- If an issue is intentional and documented, say so and move on -- don't
  re-flag known deviations from the docstrings.
- If you can't tell whether something is a bug, say "AMBIGUOUS" with the
  question you'd need answered.
