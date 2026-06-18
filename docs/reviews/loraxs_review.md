**Overall verdict:** the LoRA-XS core math is mostly correct and faithful to the official repo. The main things I would flag are **config/default scaling**, **documentation/orientation wording**, and **buffer serialization assumptions**, not the SVD factor shapes.

### `lora_xs.py`

- **`A = (Sr[:, None] * Vhr)` / `B = Ur` — verdict: correct.**  
  Given PyTorch stores `layer.weight` as `(d_out, d_in)` and computes `y = x @ W.T`, the SVD is of stored `W = U S Vh`. The code stores:
  - `lora_A`: `(r, d_in)` = `diag(Sr) Vhr`
  - `lora_B`: `(d_out, r)` = `Ur`

  Forward gives:

  ```python
  x @ A.T @ R.T @ B.T
  ```

  i.e.

  ```text
  x @ V_r @ diag(Sr) @ R.T @ U_r.T
  ```

  This matches the official PyTorch-module implementation `lora_B(R(lora_A(x)))`. In paper row-vector notation it is `x A_paper R_paper B_paper` with `A_paper = A.T`, `B_paper = B.T`, and `R_paper = layer.lora_R.T`. So the tensor shapes are right. Only caveat: if you ever load official checkpoints directly, confirm whether `R` needs transposition.

- **`h = h @ R.T` — verdict: acceptable, but orientation-sensitive.**  
  Since `R` is square and unconstrained, training from scratch is mathematically fine. But this means the stored tensor represents the transpose of the paper-row-vector `R`. Not a runtime bug, but worth documenting for checkpoint conversion.

- **`class LoRAXSConfig(AdapterConfig): variant = "lora_xs"` — verdict: suspicious.**  
  The file does not visibly enforce the reference default `alpha = r`. If `AdapterConfig` inherits the library’s usual LoRA/PiSSA default, especially `alpha = 2r`, then:

  ```python
  scale = cfg.alpha / cfg.r
  ```

  will silently use `scale=2` instead of the paper/repo’s `scale=1`. That is the most concrete faithfulness risk in this snippet.

- **`scale = cfg.alpha / cfg.r` with `R ~ normal(0, 1e-5)` — verdict: not a vanishing-gradient problem.**  
  The tiny `R` only makes the initial delta tiny. The gradient w.r.t. `R` does **not** vanish because `A` and `B` are frozen nonzero SVD factors. `lr ~ 4e-3` is plausible; the early gradient scale is governed by activations, `A`, `B`, loss gradients, and `alpha/r`, not by the magnitude of `R`.

- **`lora_A/lora_B ... as_buffer=True, trainable=False` — verdict: correct, with checkpoint caveat.**  
  This prevents grad leakage and keeps only `lora_R` trainable. Buffers should move with `.to()` and normally appear in `state_dict`. But adapter-only saving must include buffers; otherwise load will miss the frozen SVD factors. Also, checkpoint size is not just `r*r` if buffers are persisted.

- **Docstring: `"R sits between two frozen, near-orthonormal bases"` — verdict: inaccurate.**  
  `B = Ur` is orthonormal, but `A = diag(Sr) Vhr` is not; its rows have norms `Sr`. This matters for optimizer geometry and gradient conditioning.

- **Docstring: `"h = W x + (alpha/r) B R A x"` — verdict: misleading for this library.**  
  The implementation is row-vector PyTorch style:

  ```python
  y + scale * x @ A.T @ R.T @ B.T
  ```

  The docstring uses column-vector ordering. Not a code bug, but easy to confuse.

### `pissa.py` contrast

- **`Sr_eff = Sr / scale`, `sqrtS`, `W - scale * BA` — verdict: correct for PiSSA, not something to copy to LoRA-XS.**  
  PiSSA must split `sqrt(S)` and subtract the top-r component from `W` to preserve identity. LoRA-XS intentionally leaves `W` intact and trains only `R`; folding all `S` into `A` matches the reference repo. Splitting `sqrt(S)` would be a different parameterization with different optimizer dynamics, even if expressively transformable.