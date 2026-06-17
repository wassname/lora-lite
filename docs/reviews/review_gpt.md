## BLOCKER

- `src/lora_lite/adapter.py:57` / `:64` / `:66-70`

  > `variant.init(layer, cfg)`  
  > `group_init(model, attached_targets, cfg, calibration_data)`  
  > `for _, layer, _ in attached_targets:`

  `init()` crops every targeted `layer.weight` to `W_res` before calibration, but the runtime hooks that add back the frozen SVD component are only registered after `group_init()`. Therefore `group_init()` runs the calibration forward pass through a damaged model. Inputs/covariances for downstream target layers are collected from activations produced by earlier cropped layers, so Wanda re-selection / CorDA covariance bases are silently data-wrong.

  **Fix:** install the identity adapter hooks before running data-driven `group_init()`, or collect calibration activations before mutating weights. With current variants, hooks-before-`group_init` should preserve identity because `g=0`, `B=0`, and `P/Vh` initially reconstruct the cropped component.

- `src/lora_lite/adapter.py:75`, `:111`, `:121-122`

  > `base_weight_keys = [f"{n}.weight" for n in attached_names] if ran_data_init else []`  
  > `metadata = {"cfg": json.dumps(state["cfg"].to_dict())}`  
  > `handles = attach(model, cfg, _skip_group_init=True)`  
  > `missing, unexpected = model.load_state_dict(sd, strict=False)`

  Persisting rewritten base residuals is correct for the shown data-driven mutators: `antipasto`, `antipasto_corda`, and `antipasto_dplr` all rewrite `layer.weight` in `group_init()` when data is used. But the fix is incomplete: `base_weight_keys` is process-local attach state, not checkpoint metadata. After loading a data-driven checkpoint, `attach(..., _skip_group_init=True)` sets `base_weight_keys=[]`; `load_state_dict` may restore the saved weights, but a subsequent `save()` drops them again. Also, old/broken checkpoints missing the residual weights load silently with plain-SVD residuals plus data-oriented `U/S/P`, which is wrong.

  **Fix:** save `base_weight_keys` in safetensors metadata, validate those keys exist in `sd` during load, and restore `model._lora_lite_attached["base_weight_keys"]` after load. For `antipasto_corda`, absence of expected base weights should be a hard error for any valid data-oriented checkpoint.

- `src/lora_lite/adapter.py:92-99`

  > `for pname in variant.param_specs(layer.in_features, layer.out_features, layer._lora_cfg):`  
  > `del layer._parameters[pname]`  
  > `del layer._buffers[pname]`

  `detach()` deletes the adapter state but never restores the frozen component into `layer.weight`. For SVD-mutating variants, the base weight remains `W_res`, so the detached model is silently not the original model.

  **Fix:** before deleting params/buffers, restore the frozen base component:
  - `antipasto`: `weight += (U * S) @ Vh`
  - `antipasto_dplr`: same frozen `U diag(S) Vh`; do not merge learned `g/A/B` unless this is a separate `merge()`
  - `antipasto_corda`: `weight += (U * S) @ P`

  Prefer a variant method like `restore_base(layer)` to avoid hard-coding basis names in `adapter.py`.

## SHOULD

- `src/lora_lite/variants/antipasto_corda.py:12`

  > `W = U diag(S) P    (exactly)`

  This is false as written for stored rank `r`. Exact reconstruction only holds for the full SVD of `W C_eps^{1/2}`. The implemented rank-`r` adapter reconstructs via residual:

  `W = W_res + U_r diag(S_r) P_r`

  **Fix:** change the pseudocode to include `W_res`. If making the weighted approximation claim, state it precisely: SVD of `W C^{1/2}` gives the Eckart–Young best rank-`r` approximation in `E ||(W - B)x||_2^2 = ||(W-B)C^{1/2}||_F^2`.

- `src/lora_lite/variants/antipasto_corda.py:10-11`

  > `U S Vht = SVD(W C^{1/2})`  
  > `P = Vht C^{-1/2}`

  The `C^{1/2}` choice is mathematically defensible; it is not obviously wrong. It corresponds to optimal low-rank approximation in the input-covariance-weighted output norm. But it is not the same as canonical PEFT CorDA if that uses `W @ C`.

  **Fix:** document this as a deliberate CorDA-like / covariance-weighted Eckart–Young variant. If exact PEFT compatibility is intended instead, change the algebra to match PEFT and accept that it optimizes a different weighting.

- `src/lora_lite/variants/antipasto_corda.py:16`

  > `No calibration_data -> plain SVD (== antipasto).`

  Stale after the fail-fast change. `group_init()` now raises on `calibration_data is None`.

  **Fix:** remove this sentence or replace with “load uses plain-SVD seeding; public CorDA attach requires calibration data.”

- `src/lora_lite/adapter.py:57` + `src/lora_lite/variants/antipasto_corda.py:102-106`

  > `variant.init(layer, cfg)`  
  > `if calibration_data is None:`  
  > `raise ValueError(...)`

  Fail-fast is fine, but this fails after `init()` has already cropped the weights and before `_ATTACHED_ATTR` is set. In an interactive run, catching the `ValueError` leaves a corrupted model that `detach()` cannot find.

  **Fix:** validate CorDA’s calibration-data requirement before mutating weights, e.g. a variant flag like `requires_calibration_data = True` checked in `attach()` before the init loop.

- `src/lora_lite/variants/antipasto_dplr.py:53` / `:67`

  > `# Params = r (gain) + 2*r*lora_rank. k=0 degenerates to plain antipasto.`  
  > `if not 0 < k <= r:`

  Comment says `k=0` is supported; code rejects it.

  **Fix:** either allow `k=0` by omitting `lora_A/lora_B` and branching forward, or change the comment.

## NICE

- `src/lora_lite/variants/antipasto_corda.py:95-98`

  > `mandatory -- fail loud rather than silently degrade to plain SVD`  
  > `just antipasto and was the bug that made every corda run a no-op`

  Changelog narration. Remove the “was the bug” history; keep only the invariant.

- `src/lora_lite/variants/antipasto.py:187-193`

  > `# Why 1+ELU and not the obvious alternatives:`

  Too lecture-like for slim research code. Compress to one equation/comment plus citation or move rationale to docs.

- `src/lora_lite/variants/antipasto_dplr.py:4-8`

  > `The arrowhead tried a dense b x b block...`

  Historical narration. Keep the current model equation and maybe one citation; drop the archaeology.

- `src/lora_lite/variants/antipasto_corda.py:109-117`

  Long CPU/OOM implementation note. Useful, but too much in-code prose. Compress to “accumulate full C on CPU; diagonal C loses correlations” and move sizing advice elsewhere.

- PERF: no hot-loop `einops` issue. The `rearrange` use is only calibration/init-side, not per-token forward.