## Code Review: Four S-space weight adapters (anti-pasto family + sspace reference + cost script)

### Summary
The code adds four PiSSA-style adapters (antipasto, antipasto_rot, antipasto_ablate, antipasto_corda) that store frozen top-r SVD buffers and a small trainable gain/core. The key mathematical claims in the docstrings (identity at g=0, contraction of ablation core, exact reconstruction of the CorDA decomposition, signed cosine gate) are correct. However, `group_init()` re-selects/re-orients the SVD basis but does **not** reset the trainable parameters (g, delta_s, rot_T, c, alpha), which introduces a latent but important correctness problem — after a data-driven reinit, gains and direction vectors become misaligned with their intended singular axes.

---

### Critical (must fix)
- **`antipasto_corda.py`, `antipasto_rot.py`, `antipasto_ablate.py`:** `group_init()` re-orients or re-selects the top-r SVD basis but leaves the existing trainable parameters (`lora_g`, `lora_delta_s`, `lora_rot_T`, `lora_c`, `lora_alpha`) untouched, still attached to the old indices.  
  - For `antipasto_corda`, `lora_g` is initialised with `N(0, 4e-4)` in `init()`. After `group_init`, those small random gains now multiply a different set of singular directions, producing an uncontrolled (though tiny) perturbation.  
  - For `antipasto_rot`, the `+4e-4` bias on `lora_delta_s` remains, now applied to arbitrary resorted directions, and the block rotation `lora_rot_T` is disconnected from the new block structure.  
  - For `antipasto_ablate`, the ablation directions and strengths (`lora_c`, `lora_alpha`) are not reset; if any warm-starting or training has happened, they point in the wrong subspace.  
  **Fix:** After re-selection or re-orientation, either re-initialise the trainable parameters (e.g., zero for g, zeros/small for delta_s and rot_T, random-normalised for c, init_alpha for alpha) or re-index them with the same `idx` mapping used for the buffers. Document that `group_init` must be called **before any training** and that the trainable parameters will be fresh after it.

### Important (should fix)
- **`antipasto_ablate.py`** – Contraction claim and coeff bounds: the forward pass applies `h = h - coeff * (proj * alpha) @ Chat.T`. The core is a contraction only when both `coeff` and `alpha` are in `[0, 1]`. The config’s `coeff` can be *any* float (the docstring mentions `coeff<0` for amplification). There is no runtime clamping of `coeff`. If the intention is to keep the contraction property structurally enforced, clamp `coeff` to `[0, 1]` inside `forward()` or at least validate it.
- **`sspace.py`** – Division by `sqrtS` without epsilon: `xS = (y_eff @ U_r) / sqrtS`. For small or zero singular values (e.g., when r is large or W is low-rank) this can produce NaNs/infs. Add a small `ε` denominator (consistent with the ε used elsewhere).
- **`antipasto_ablate.py`** – `_orthonormal()` calls `torch.linalg.qr(c.float())` every forward pass. For large `r` and many layers this adds non-trivial cost. A lighter reparameterisation (e.g., maintaining `c` as a matrix with a normalisation step that avoids full QR) might be warranted, but for small `(r,k)` pairs the current approach is acceptable. At minimum, add a comment noting the per-forward cost.

### Suggestions
- **`antipasto.py` group_init:** after the idx sort, `idx = idx.sort().values` ensures a stable, canonical ordering. This is a nice touch for reproducibility.
- **`antipasto_ablate.py`** – The docstring says “CorDA-orient the basis from input covariance … the ablation is OUTPUT-side and CorDA's U stays orthonormal …”. The forward code correctly uses the orthonormal `U` for output projection, so the contraction in S-space carries over to the output.
- **`sspace.py`** – The signed gate correctly preserves `cos` sign, so anti-aligned tokens receive a negative `gate * alpha` and are pushed opposite to `dS_hat`. Verified.

### Positive
- The documentation is thorough, explaining the design choices (why 1+ELU, why contraction, why CorDA), and includes references.  
- The PiSSA-style `W_res` decomposition is implemented correctly across all variants.  
- `antipasto.py`’s `S_eff = S * (1 + ELU(coeff*g))` is indeed C1, positive, and identity at `g=0` — no sign-flip bugs.  
- `antipasto_ablate.py` enforces orthonormal `Chat` and clamps `alpha` to `[0,1]`, making the contraction property safe when `coeff` is also bounded.

### Verdict
**REQUEST CHANGES**

`group_init()` must reset the trainable parameters after it changes the SVD basis; without this, the adapters silently poison their own steering gains with a different set of directions. Fix that and the code is ready.

---

*Note: The `sspace.py` and `_cost.py` files are part of the same move into lora-lite and are free of the parameter-reset issue; the only concern in `sspace.py` is the unprotected division by singular values.*