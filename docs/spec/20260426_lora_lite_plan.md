# lora-lite plan and status

## Goal

Build a small, hackable LoRA-family adapter library for research experiments.

The core bet is that adapter variants should own the relationship between `(x, layer.weight, layer.lora_*)` and the layer output, while the library only handles targeting, parameter attachment, hooks, and save/load.

## Non-goals

- No PEFT compatibility layer.
- No module replacement.
- No merge/unmerge.
- No multiple named adapters per layer.
- No backward compatibility promises.
- No silent fallbacks.

## Design constraints

- Adapter params are attached directly to target layers as `lora_*` parameters.
- Save/load uses normal `state_dict()` keys, filtered by `"lora_"`.
- Forward hooks return the layer's new output, not just a delta.
- Targeting is structural: modules with `in_features`, `out_features`, and `weight` are linear-like.
- LoRA/DeLoRA support bnb-style 4/8-bit forward paths because the quantized base layer computes `y`; the hook only adds adapter math.
- PiSSA is fp-only in v1 because it mutates `layer.weight` into `W_res`.
- Data-calibrated variants use `group_init(model, targets, cfg, calibration_data)`; dataloaders stay out of `cfg` so checkpoints are serializable.

## Implemented v0.0.1

| Area | Status | Evidence |
|---|---:|---|
| `LoraLiteConfig` | done | `src/lora_lite/config.py` |
| Variant registry + `ParamSpec` | done | `src/lora_lite/variant.py` |
| Structural target discovery | done | `src/lora_lite/target.py` |
| `attach` / `detach` / `save` / `load` | done | `src/lora_lite/adapter.py` |
| LoRA | done | `src/lora_lite/variants/lora.py` |
| PiSSA | done, fp-only | `src/lora_lite/variants/pissa.py` |
| DeLoRA | done | `src/lora_lite/variants/delora.py` |
| Smoke tests | done | `tests/smoke.py` |
| bnb minimal forward smoke | done | `Linear8bitLt` and `Linear4bit` pass on CUDA |

## Current smoke evidence

Last verified log: `/home/wassname/.cache/agent-tmp/lora_lite_smoke_after_review.log`

| Check | Result |
|---|---|
| LoRA identity | `0.000e+00` |
| LoRA loss drop | `6.1%` |
| PiSSA identity | `1.550e-06` |
| PiSSA loss drop | `11.5%` |
| DeLoRA identity | `0.000e+00` |
| DeLoRA loss drop | `93.4%` |
| fake non-`nn.Linear` target | attaches, identity `0.000e+00`, grad nonzero |
| bnb `Linear8bitLt` | identity `0.000e+00`, grad nonzero |
| bnb `Linear4bit` | identity `0.000e+00`, grad nonzero |

## 2026-04-26 testing proof pass

Goal: upgrade from smoke-tested sketch to evidence that the current PEFT-lite interface trains on both toy models and a real HF Qwen model.

### Scope

In:

- Pytest coverage for LoRA, PiSSA, and DeLoRA correctness invariants.
- A real `Qwen/Qwen3-0.6B` probe that trains each current variant on layer-0 `q_proj` and `v_proj`.
- Repeatable `just` recipes and workspace-local logs/artifacts.

Out:

- Benchmark claims.
- Quantized Qwen proof for PiSSA. PiSSA remains fp-only because it mutates `weight`.
- Full default-target training over every Qwen layer.

### Requirements and evidence

| Requirement | Distinguishing check | Evidence |
|---|---|---|
| R1: toy tests catch skipped targets/hooks | Perturb only `lora_*`; output must change. Missing target must raise. | `just test` -> `8 passed in 2.43s` in `logs/pytest.log` |
| R2: toy tests catch base-gradient leakage | After backward, all non-`lora_*` grads are `None`; all trainable names contain `lora_`. | `just test` -> `8 passed in 2.43s` |
| R3: save/load is exact for adapters | Saved key set equals full-path `lora_*` state; reload tensors equal; missing/extra `lora_*` keys raise. | `just test` -> `8 passed in 2.43s` |
| R4: current variants train on tiny task | 28 TinyModel targets; non-`lora_*` grads stay `None`; 20-step loss drop >5%. | `just smoke` -> LoRA 6.1%, PiSSA 11.5%, DeLoRA 93.4% |
| R5: current variants train on real Qwen | Fresh Qwen per variant; exact targets are layer-0 `q_proj`/`v_proj`; perturb >0; lossN < loss0; reload err < tol. | `pueue` task 70, `logs/qwen_probe.log`, all probes pass |
| R6: cold review cannot explain evidence under silent failure | External review findings fixed, then fresh-eyes subagent says PASS. | `docs/spec/20260426_code_review.md` |

### Qwen proof table

Command:

```bash
pueue add --immediate --follow --label "why: verify warning-free current Qwen probe after dtype API cleanup; resolve: same pass table proves current script" --working-directory "$PWD" --priority 1 -- just qwen-probe
```

Result from task 70:

| variant | targets | trainable | id_err | perturb | loss0 | lossN | drop% | grad | dθ | reload | adapter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lora | 2 | 20480 | 0 | 0.375 | 5.25 | 3.131 | 40.36 | 1.432 | 4.262 | 0 | `outputs/qwen_train_probe/lora_adapter.pt` |
| pissa | 2 | 20480 | 0.3125 | 0.75 | 5.25 | 3.629 | 30.88 | 6.124 | 4.381 | 0 | `outputs/qwen_train_probe/pissa_adapter.pt` |
| delora | 2 | 20482 | 0.375 | 0.4062 | 5.246 | 5.166 | 1.537 | 0.04778 | 8.196 | 0 | `outputs/qwen_train_probe/delora_adapter.pt` |

Failure-mode interpretation:

- If targeting silently skipped, exact target-set assertion would fail before training.
- If hooks were attached but dead, perturb delta would be 0.
- If base params trained, the non-`lora_*` gradient check would fail.
- If adapter grads were absent, `grad` or `dθ` would be 0/non-finite.
- If save/load were broken, adapter tensor equality or reload logit error would fail.

## 2026-04-26 publishable workflow pass

Goal: make the repo behave like a small buildable library without adding framework surface area.

### Scope

In:

- Keep minimal functional integration tests as the default proof path.
- Add a GitHub CI hook for fast tests and package build.
- Add a `pueue` recipe for the larger Qwen proof.
- Tighten README structure so install, quickstart, core idea, tests, and status are easy to find.

Out:

- PyPI publishing token/workflow. Publishing should wait until the license decision is explicit.
- Implementing every named adapter variant in one pass. That would add complexity faster than tests can explain it.

### Requirements and evidence

| Requirement | Distinguishing check | Evidence |
|---|---|---|
| R7: fast CI catches broken tests/builds | `just check` must run pytest, smoke, `uv build`, and `twine check`; a broken test, wheel, sdist, or README metadata fails the hook. | `just check` -> pytest `8 passed in 9.53s`, smoke all pass, wheel/sdist built, `twine check dist/*` passed |
| R8: large proof is queued, not hidden in CI | `just qwen-queue` must create a pueue task in the repo cwd with why/resolve label and intended Qwen command. | `just qwen-queue && pueue status` -> task 74 queued at `/media/wassname/SGIronWolf/projects5/2026/lora-lite` with `just qwen-probe lora pissa delora 16` |
| R9: README is publishable enough to judge | Reader sees install, quickstart, pseudocode core, testing commands, proof caveat before variant wishlist. | `README.md` reordered and human note removed |
| R10: variant roadmap buys simplicity | Next variant is ranked by fit to current hook contract; non-hook variants are deferred rather than half-supported. | this section |

Fresh review first blocked on weak `qwen-queue` evidence and README citation/comment junk. Fixes: queued real pueue task 74, added `twine check`, fenced citation, removed the stray README note. Final fresh review verdict: PASS.

### Adapter roadmap, ranked by simplicity

| Variant | Why it fits or waits | Next check |
|---|---|---|
| IA3 | Multiplicative vector on activations. Probably the smallest new file and no base-weight mutation. | Identity with ones, perturb changes output, loss drops, save/load exact. |
| DoRA | Fits additive hook for fp layers; bnb norm handling must be explicit or fail-fast. | fp smoke first; quantized proof only after norm semantics are obvious. |
| SSVD / PiSSA-family | Fits current `weight`-SVD pattern and teaches the SVD adapter path. | Reconstruction/identity invariant plus train proof. |
| HRA / OFT / ROAD | Interesting, but likely wants orthogonal or weight-transform semantics. Keep until hook-only formulation is clear. | Pseudocode first, then one invariant that distinguishes real rotation from dead code. |
| S-steer / AntiPaSTO | Research adapters. Should use `group_init` and activation evidence, not be squeezed into plain LoRA tests. | Calibration is consumed, hooks removed, load does not need calibration data. |

## Review history

A cold subagent review first returned `PASS_WITH_BLOCKERS`:

1. bnb modules were not targeted.
2. Hook cast `y` to `cfg.dtype`, which could round base outputs.
3. PiSSA overclaimed bnb support.
4. `load()` did not fail on missing adapter keys.
5. Data-calibrated init needed model-level access.

Fixes applied:

1. Structural `is_linear_like()` target predicate.
2. Hook only casts `x`, keeps `y` in base output dtype.
3. PiSSA fail-fast rejects non-plain `nn.Linear`.
4. `load()` fails on missing or unexpected `lora_` keys.
5. `attach(..., calibration_data=None)` plus optional `group_init(model, targets, cfg, calibration_data)`.

Second cold review verdict: `PASS` for the minimal 4bit-enabled scope.

## TODO / status

### Next implementation goals

- [ ] Add DoRA.
  - Verify: fp32/bf16 identity at init, finite gradients, and smoke loss drop.
  - Caveat: bnb DoRA needs explicit weight dequantization for norm computation or should be fp-only at first.

- [ ] Add VeRA.
  - Verify: shared buffers are allocated once, target slices match shape, identity or near-identity at init.

- [ ] Add SSVD or AntiPaSTO-style SVD variant.
  - Verify: reconstruction or intended rotation invariant at init.

- [ ] Add real activation-calibrated toy variant using `group_init`.
  - Verify: `calibration_data` is consumed during `attach`, hooks are removed, checkpoint is serializable, and `load()` does not require calibration data.

- [ ] Add load path that can skip calibration init for future `group_init` variants.
  - Current caveat: `load()` calls `attach(model, cfg)` with `calibration_data=None`; fine for current variants, but future calibrated variants should separate param creation from calibration.

- [ ] Add a tiny HF-model smoke when convenient.
  - Verify: target names look like real transformer modules and state dict keys match full paths.

### Design TODOs

- [ ] Decide whether `group_init` should run before or after forward hooks are registered.
  - Current choice: after params are attached, before adapter forward hooks are registered.

- [ ] Decide whether replacing variants need `runs_base_layer=False` or can always transform `y`.
  - OFT-like variants can rotate `y`; variants that truly avoid base forward need module replacement or pre-hook rewriting, likely out of v1.

- [ ] Add `weight_mode` for BitFit/SHiRA if those variants become in-scope.
  - Minimal surface: `weight_mode in {"frozen", "bias_only", "sparse_grad"}`.

## Variant contract

```python
class Variant:
    name: str

    @staticmethod
    def param_specs(d_in, d_out, cfg) -> dict[str, ParamSpec]: ...

    @staticmethod
    def init(layer, cfg) -> None:
        # weight-only init; may mutate plain fp weights
        ...

    @staticmethod
    def group_init(model, targets, cfg, calibration_data) -> None:
        # optional model-level init for data-calibrated or cross-layer variants
        ...

    @staticmethod
    def forward(layer, x, y) -> Tensor:
        # return NEW output; additive variants return y + delta
        ...
```

## Done means

This repo is good enough for a first real experiment when:

1. A Qwen/Llama model can attach LoRA adapters to intended target layers.
2. A 4bit or 8bit loaded model can train LoRA/DeLoRA params with nonzero gradients.
3. Saved adapter tensors use full-path keys and reload without calibration data.
4. Smoke tests distinguish target-skipping, hook identity drift, and missing-key load failure.

see interesting adapters here https://github.com/wassname/adapters_as_hypotheses
how peft handle 4bit here https://github.com/huggingface/peft/blob/6030f9160ed2fc17220f6f41382a66f1257b6a93/src/peft/tuners/lora/layer.py