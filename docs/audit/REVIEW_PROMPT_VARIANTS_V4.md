# V4 Variant Review — per-component vs reference + smoke/probe validity

You are an expert ML engineer reviewing a from-scratch PEFT library
(`lora-lite`, ~500 LOC) that re-implements 8 LoRA variants. Three prior
reviews already happened (V1 paper-vs-code, V2 with refs provided, V3
per-component). Your job is V4: re-run the per-component check and
additionally validate the test harness.

# Part A — per-variant audit (re-do, more rigorous)

8 variants live in `src/lora_lite/variants/`:
- lora.py
- pissa.py
- delora.py
- ia3.py (registers `ia3` and `ia3_ff`)
- dora.py
- hra.py
- eva.py
- antipasto.py

Plus runtime in `src/lora_lite/{adapter.py,variant.py,target.py,config.py}`.
Reference implementations are in `docs/refs/` and URLs are pasted in each
variant's module docstring.

## For EACH variant, in this order, every time:

1. **REFERENCE EXISTS** — verify the variant has a real, citeable
   reference. Required:
     - a paper (arxiv/conference) link, AND
     - either an upstream peft implementation OR the original author's
       code (GitHub).
   If the variant has NO paper, NO reference code, OR the references
   are dead/missing/clearly wrong, FLAG IT as `NO REFERENCE` -- this
   is severity HIGH because it means there's nothing to validate
   against.

2. **PARAMS** — every spec from `param_specs`: shape, dtype, trainable,
   as_buffer. Match against the reference. Buffers vs Parameters
   chosen correctly?

3. **INIT** — what does `init()` (and `group_init()` if defined) do?
   Match the reference exactly? Walk gradient flow at t=0: which
   trainable params actually receive non-zero gradient on step 1?

4. **DTYPE** — trace dtype through init -> storage -> forward.
   Silent precision loss? Identity-at-init survive bf16?

5. **FORWARD** — write the math the forward implements vs the math
   in the reference. Term-by-term comparison. Common mistakes:
   - wrong scale (alpha/r vs 1/r vs alpha vs 1)
   - missing/doubled normalization
   - wrong basis (rotating U vs V; gating input vs output)
   - dropout placement (we have NO dropout by design — flag if any
     code path depends on one)

6. **LINK SANITY** — actually open the URLs. Verify:
   - paper arxiv link goes to the right paper
   - github link points to a real file (not 404)
   - offline `docs/refs/` snapshot still matches what the URL serves
     today (snapshots may be stale; flag drift)

## Per-variant output (≤60 lines each):

    ## <variant>

    ### references
    - paper: <url>  -- OK / WRONG / DEAD / MISSING
    - peft ref: <url>  -- OK / DEAD / MISSING
    - author ref (if any): <url>  -- OK / DEAD / MISSING
    - offline snapshot (`docs/refs/...`): NONE / MATCH / DRIFT
    - VERDICT: HAS_REFERENCE / NO_REFERENCE

    ### params
    - <one bullet per ParamSpec; flag bug if any>

    ### init / group_init
    - <bullets; identify GRADIENT FLOW at t=0 explicitly>

    ### dtype
    - <bullets>

    ### forward
    Math (ours):    <one-line equation>
    Math (ref):     <one-line equation>
    Match?          YES / NO + one-line reason

    ### verdict
    CORRECT / PARTIAL / BUGGY  -- one-sentence reason

# Part B — validate the smoke test (`tests/smoke.py`)

Read `tests/smoke.py` end-to-end. For each per-variant SHOULD claim,
answer:

1. **Distinguishing power** — would a SILENT FAILURE (e.g. forward
   returning `y` unchanged, or training only the bias term, or
   loading an empty state dict) STILL pass this check? If yes,
   the check is WEAK -- name a stronger one.

2. **Tolerance sanity** — the bf16/fp16 tolerances are computed
   from `base_scale`. Are they too loose? Too tight? Could they
   pass on noise alone?

3. **Coverage** — what mechanisms are NOT tested? (e.g. multi-step
   convergence on real targets, dtype mismatch between attach and
   load, mixing variants, calibration data of len < r for EVA)

Output:

    ## smoke.py validity

    ### per-variant SHOULD checks
    | check | distinguishes silent failure? | tolerance ok? | notes |
    | ... |

    ### gaps
    - bullets

    ### must-add tests
    - bullets

# Part C — validate the qwen overfit probe (`scripts/qwen_train_probe.py`)

Read `scripts/qwen_train_probe.py` end-to-end. Same questions as Part B
but for the Qwen probe specifically:

1. Does `assert_only_lora_trainable` actually catch a leaked base
   parameter, given the way `requires_grad` is set in `adapter.py`?

2. `perturb_first_adapter` only perturbs ONE param per variant. Does
   `perturb_delta > 1e-7` distinguish "the variant uses that param in
   forward" from "the variant ignores that param"?

3. `loss_last < loss0` after 8 steps with lr=5e-3 -- could this pass
   purely from optimizer noise? What's the right held-out / validation
   check to add?

4. The reload check uses `args.reload_tol` (default 2e-2 in bf16). Is
   that loose enough to mask a real save/load bug?

5. Targets are restricted to `model.layers.0.self_attn.{q,v}_proj` --
   does this exercise the full attach path or hide bugs that only
   appear with multi-layer / FFN / lm_head edge cases?

Output:

    ## qwen_train_probe.py validity

    ### claim-by-claim
    | assertion | catches silent failure? | notes |
    | ... |

    ### gaps
    - bullets

    ### must-add tests
    - bullets

# Final summary

After parts A, B, C, write:

    ## summary

    ### variant verdicts
    | variant | has_ref | params | init | dtype | forward | verdict |

    ### MUST-FIX (severity HIGH, blocks correctness claim)
    1. ...
    2. ...

    ### NICE-TO-HAVE
    - ...

# Hard rules

- Be specific. Cite line numbers (`src/lora_lite/variants/foo.py:NN`)
  for every claim.
- Do NOT propose redesigns. Only flag correctness issues against
  references and validity issues in the test harness.
- If an issue is intentional and documented in the docstring, say so
  and move on -- don't re-flag known deviations.
- If you can't tell whether something is a bug, say "AMBIGUOUS" with
  the question you'd need answered.
- For Part B/C, focus on whether checks have DISTINGUISHING power
  (would a silent failure still pass?) -- not just whether they run.
