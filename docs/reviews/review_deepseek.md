## Review

**BLOCKER:** none found. The math, save/load round-trip, and variant coverage all check out.

---

### SHOULD

**1. antipasto_corda.py:13-18 — docstring falsely claims fallback behavior**
```python
Identity at g=0 or coeff=0: S_eff=S. P is oblique (rows not orthonormal -- C^{-1/2}
skews them); fine for gain reweighting and for output-side ablation (the obliqueness
is input-side; U stays orthonormal). No calibration_data -> plain SVD (== antipasto).
```
The last sentence is now **wrong**: `group_init` raises `ValueError` when `calibration_data is None` (by design, after the bugfix). The docstring advertises a silent fallback that no longer exists. Remove or replace with "requires calibration_data (raises otherwise)".

**2. antipasto_corda.py:55-67 — group_init docstring narrates the bugfix history**
```python
"""Re-orient each target's SVD by its input covariance C = E[x x^T].
...
Covariance orientation IS this variant's identity, so calibration_data is
mandatory -- fail loud rather than silently degrade to plain SVD (which is
just antipasto and was the bug that made every corda run a no-op).
...
Do not call group_init after training has updated g."""
```
The commentary "was the bug that made every corda run a no-op" is changelog narration. The last sentence lectures the reader. Replace with a single line like `# Requires calibration_data; raises otherwise. Call only at attach-time (before training).`

**3. antipasto_corda.py:80-88 — long design-rationale block is rambling**
```python
# accumulate C = sum x x^T on CPU. Peak GPU cost would otherwise be
# sum_targets d_in^2 fp32 held at once; for down_proj (d_in=intermediate,
# e.g. 14336) that is ~0.8 GB *per layer* and OOMs. CPU accumulation bounds
# GPU use to the live activation; the eigh/SVD below run on CPU (one-time).
# Diagonal C is NOT a usable shortcut: it misses cross-channel correlation,
# which is where the orientation gain lives (measured ~= plain SVD).
# If down_proj's d_in^2 is too big even on CPU/RAM, exclude it from CorDA
# (leave it on plain antipasto) or use a low-rank C (top-k eig of subsampled
# inputs) -- not implemented here.
```
This is lecture/rambling and future-work speculation. Slim to `# CPU accumulation: d_in^2 per layer OOMs GPU.` The rest is for a design doc or paper, not inline.

**4. antipasto_dplr.py:14-20 — docstring narrates abandoned variants**
```python
antipasto's core is diagonal (a per-direction gain); it rescales each singular
direction but cannot mix one into another. The arrowhead tried a dense b x b block
on the top-b directions, but a dense block is the wrong shape (b^2 params, mixes only
the top-b) and -- sitting on the S-scaled coords -- its perturbation is amplified by
the largest singular values, so it destabilizes. The fix is LoRA's lesson: a low-rank
core. ...
```
History of "the arrowhead" variant and "the fix is LoRA's lesson" is changelog narration. The paragraph starting "Why the low-rank part is ADDED…" (lines 23-26) repeats the same. Trim to: `# Additive low-rank core B@A in the frozen SVD basis. Independent of S → no amplification edge.` and drop the storytelling.

**5. antipasto_dplr.py:26 vs 144 — docstring claims operator is `B A`, code computes `(B A)^T`**
Docstring:
```
y = x @ W_res.T + ( (Vh x) * S_eff  +  coeff * B (A (Vh x)) ) @ U.T
```
Code:
```python
h = p * S_eff + coeff * (p @ A.T) @ B.T    # p = x @ Vh.T
```
The effective operator in S-space is `diag(S_eff) + coeff * (B @ A).T`, not `B @ A`. The parameterization can represent the same matrices (swap B↔A^T), so it's not a math bug — but the docstring describes the wrong composition. Fix the docstring or swap `A.T`/`B.T` to `B`/`A` in the forward pass to match the stated convention. The LoRA convention is `x @ A.T @ B.T` for `ΔW = B @ A`; here the natural convention would be `p @ B @ A` for core `B @ A`, but the code produces the transpose.

---

### NICE

**6. antipasto_corda.py:12 — `C^{1/2}` justification is solid; user's note is mistaken**
The user asks whether canonical CorDA uses `W @ C` (not `C^{1/2}`) and whether `C^{1/2}` is defensible. **Both the canonical CorDA paper (arXiv:2406.05223) and this code use `W @ C^{1/2}`** — CorDA does Cholesky `C = L L^T` and SVD on `W L`, which is `W @ C^{1/2}`. The `C^{1/2}` form is **correct**: minimizing `||(W - W_r) C^{1/2}||_F` minimizes the expected reconstruction error `E[||W x - W_r x||^2]` for `x ~ N(0, C)`. The symmetric eigh square-root used here is equivalent to the Cholesky factor. The comment on line 93 ("CorDA whitens with full C") in an earlier version of the docstring would be factually wrong — but the current file doesn't contain that claim; only the user's prompt does.

**7. antipasto.py:97 — misleading comment about CPU capture**
```python
proj = X.to(Vh_full) @ Vh_full.T   # input in S-coords (X captured on CPU)
```
`X` was captured on CPU, but `X.to(Vh_full)` moves it to GPU for the projection (since `Vh_full` is on GPU after `torch.linalg.svd(W_orig)`). The parenthetical is either stale or ambiguous. Clarify: `# X was accumulated on CPU; moved here to GPU for the projection.`

**8. antipasto.py vs antipasto_corda.py — duplicated ELU logic**
`antipasto.py:forward()` inlines `1.0 + F.elu(coeff * g)` while `antipasto_corda.py` has a factored-out `_gain()` helper. The `antipasto_dplr.py` inlines it again. Unify to `_gain` or a shared utility. Low priority.

**9. adapter.py:57 — `base_weight_keys` uses `attached_names` from `for` loop, but `attached_names` is only populated if no early exceptions occur**
```python
attached_names: list[str] = []
attached_targets = []
for name, layer, role in targets:
    ...
    attached_names.append(name)
    attached_targets.append((name, layer, role))
```
If `variant.init()` or `param_specs` raises mid-loop, `attached_names` is inconsistent with `attached_targets`. Harmless since the exception propagates and `attach()` is discarded, but the partial `attached_targets` is passed to `group_init` which would see only the first N layers. Since `group_init` uses `targets` as a dict keyed by name, and `attached_targets` is only needed for the hook registration after group_init, this could cause group_init to silently miss later layers if an exception were caught — but nothing catches it. Not a bug in current fail-fast style.

---

### Save/load completeness

**Coverage:** All three variants (`antipasto`, `antipasto_corda`, `antipasto_dplr`) have `group_init` that mutates `layer.weight` when `calibration_data` is provided. The `ran_data_init` guard in `attach()` (line 64) only sets `base_weight_keys` when `group_init` is present AND `calibration_data is not None` AND `_skip_group_init` is False — this correctly covers all three. ✓

**Ordering at load:** `attach(_skip_group_init=True)` runs `init()` (plain-SVD crop), then `load_state_dict` overwrites all buffers AND the base weight (if present in checkpoint). The final state is the checkpoint's, not init's. ✓

**No double-restore:** `base_weight_keys` is only used during `save()`. During `load()`, the checkpoint's keys are applied via `load_state_dict(strict=False)` indiscriminately; base weights are restored if present. The `base_weight_keys` list from the **loading** model (always `[]` since `_skip_group_init=True`) is irrelevant — it's not consulted during load. ✓

**Checkpoint cross-variant safety:** If a checkpoint saved from `antipasto_corda` (with base weights) is loaded into an `antipasto` model, the `lora_P` key in checkpoint has no matching parameter in the model → `unexpected_lora` catches it and raises. If a corda checkpoint saved WITHOUT base weights (pre-bugfix, `ran_data_init=False`) is loaded into corda, the checkpoint has plain-SVD `lora_P` but no base weight → model's `init()` W_res and checkpoint's `lora_P` are both plain SVD, so they match. ✓

**No gap:** Every variant whose `group_init` rewrites `layer.weight` is covered by the `ran_data_init` → `base_weight_keys` path. ✓

---

### Perf (minor)

- No `einops`/`einsum` in forward hot loops. `einops.rearrange` only appears in `group_init` hook callbacks (one-time calibration). ✓

---

### Verdict

No math errors found. The `C^{1/2}` approach is equivalent to canonical CorDA (via Cholesky). Save/load round-trip is correct and complete. The main issues are docstring rot (stale fallback claim, changelog narration, operator transpose mismatch) and overly chatty inline comments.