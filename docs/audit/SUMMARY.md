# External-Review Summary

Two independent reviews via `acpx` external models. Full reviews:
- [docs/audit/variants_review.md](variants_review.md) — per-variant paper-faithfulness audit
- [docs/audit/design_review.md](design_review.md) — peft EVA / baukit / antipasto3 vs lora-lite design

## Per-variant verdict

| variant | match | bugs found | confidence |
|---|---|---|---|
| lora    | Y       | none material                                                                          | High        |
| pissa   | Partial | bf16/Qwen init err 0.31; deviation `alpha==r` only in inline comment; residual not in saved adapter | Medium      |
| dora    | Y       | possible denominator-gradient mismatch with paper's "cost-saving" variant              | High        |
| ia3     | Partial | targets q/v not paper's k/v/ffn-down; deviation documented but not tested              | Medium      |
| hra     | Partial | gate=0 init -> dU/dx=0 first step (lora_U dead); not orthogonal when gate != 1         | Medium-Low  |
| delora  | Partial | no Eq.9 frozen-copy init; lambda0=0 -> A/B dead grad; lambda0=0.1 breaks identity      | Medium      |

## Three biggest risks (reviewer's words)

1. **Initialization vs gradient-flow tradeoffs are hidden by coarse tests.** HRA's `lora_U` and DeLoRA's `A/B` can be initially dead while `grad_nonzero=True` still passes (because *some* lora_* param has nonzero grad).
2. **Qwen probe pass criteria do not enforce paper identity.** PiSSA shows `id_err=0.31`, DeLoRA `id_err=0.72`, but log says PASS.
3. **Target semantics under-tested.** IA3's documented k/v/ffn deviation is never exercised by a positive test.

## Design recommendations

| ref | verdict | impact |
|---|---|---|
| peft EVA       | PARTIAL — add `calibrate(model, dataloader, cfg)` (~50 lines) | +50 lines, additive |
| baukit nethook | SKIP — current 5-line hook registration is simpler          | 0 |
| antipasto3 SVD | ADOPT concept (learnable delta_s) — no code change now      | 0 |

## Recommended follow-up tasks (need user approval before implementing)

A. **Per-param gradient probe**: extend smoke to assert grad on *each* lora_* param at step 0. Catches HRA/DeLoRA init-dead-param bug.

B. **Per-variant identity tolerance in qwen probe**: PiSSA/DeLoRA need a stricter check (or relative tol against `||y_base||`) instead of "passes if id_err < some constant".

C. **IA3 paper-faithful test row**: add one Qwen probe configuration with `target_names=k_proj|v_proj|down_proj` to exercise the documented IA3 placement.

D. **PiSSA equivalence test against `peft.PiSSA`**: same seed + alpha=r, compare `B@A` reconstruction. Adds `peft` to test extras only.

E. **EVA variant**: implement minimal `calibrate()` per design review (~50 lines). Optional, but provides our first data-driven init variant for the user's stated interest.
