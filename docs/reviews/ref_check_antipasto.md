# Reference-fidelity check: antipasto.py vs Wanda / ASVD / PiSSA

Scope: math/algorithm fidelity and citation honesty only. Fail-fast research code, so
missing None-checks / fallbacks / backward-compat are explicitly NOT flagged.

File under review: `src/lora_lite/variants/antipasto.py`
Papers (read in full where relevant):
- Wanda: `docs/papers/md/wanda_2306.11695.md`
- ASVD: `docs/papers/md/asvd_2312.05821.md`
- PiSSA: `docs/papers/pissa_2404.02948.txt`

Legend per point: block quote (paper, location) -> code (file:line) -> VERDICT.
"obs" = directly read; "inf" = my inference from those reads.

---

## 1. Wanda metric vs our per-direction selection score

Paper, Wanda Eq.(1), Sec.3 "Pruning Metric" (wanda_2306.11695.md:83):

> 𝐒_ij = |𝐖_ij| · ‖𝐗_j‖₂

and (wanda_2306.11695.md:62, Fig.1 caption):

> we compute the weight importance as the elementwise product between the weight
> magnitude and the norm of input activations (|𝐖| · ‖𝐗‖₂). Weight importance
> scores are compared on a per-output basis (within each row in 𝐖), rather than
> globally across the entire matrix.

Our code (antipasto.py:159, with :154-158):

```python
proj = X.to(Vh_full) @ Vh_full.T          # (N, r), input projected onto right singular vecs
act_mag = proj.pow(2).mean(0).sqrt()      # 'rms'  (L2-style, per-direction)
#       = proj.abs().mean(0)              # 'mean_abs'
scores = S_full * act_mag                 # score[i] = S[i] * pool|X @ Vh[i]|
```

obs: Wanda's score is per scalar WEIGHT element `W_ij` (a `C_out x C_in` grid of
scores), magnitude `|W_ij|` times the L2 norm of the corresponding INPUT-CHANNEL
activation `X_j`, compared within each output row to decide which entries to zero.

obs: Our score is per singular DIRECTION `i in [0,r)`: singular value `S[i]` (a
property of the whole rank-1 component `U[:,i] S[i] Vh[i]`) times the pooled
magnitude of the activation projected onto the right singular vector `Vh[i]`.
The comparison group is the spectrum of one weight matrix; the action is direction
SELECTION (keep top-r), not weight zeroing.

inf: The shared idea is genuine: "importance = magnitude x how much the input
actually drives this coordinate," estimated from calibration activations in a
single forward pass. With `pool='rms'`, `pool|X @ Vh[i]|` IS exactly an L2-style
norm of the activation in the rotated (singular) basis, so it is the Wanda norm
applied to `X @ Vh^T` instead of raw `X`. If `Vh` were the identity (axis-aligned
channels), `S[i]·rms(X[:,i])` would reduce to a per-channel Wanda score.

inf: Where the analogy breaks (three real differences, none hidden by the code):
(a) granularity: Wanda scores individual weights `W_ij`; ours scores rank-1
SVD components. (b) basis: Wanda works in the raw input-channel basis; ours in the
right-singular-vector basis (`X @ Vh^T`). (c) operation: Wanda PRUNES (sets weights
to 0, keeping the matrix shape); ours SELECTS which directions land in the trainable
low-rank core vs the frozen residual. So `|W_ij|` -> `S[i]` is an analogy, not an
identity: `S[i]` is the magnitude of a whole component, not of one weight.

VERDICT: DEVIATES-OK (honest analogy, not an over-claim). The docstring labels it
"Wanda-style pooling" / "Wanda/ASVD" (antipasto.py:50, 95) -- "-style" is the right
hedge. It would only be an OVERCLAIM if it claimed to BE Wanda; it says
"Wanda-style," which is accurate. Note: Wanda found L2 beats L1/L_inf
(wanda_2306.11695.md:85); our `'rms'` pool matches that recommendation, our default
(antipasto.py:52) is `'rms'`.

---

## 2. ASVD: whitening vs intuition-only citation

Paper, ASVD Sec.3.3, transform + scaled SVD (asvd_2312.05821.md:215-249):

> 𝐖 = 𝐖𝐒𝐒⁻¹ = (𝐖𝐒)𝐒⁻¹.  ... apply SVD to the transformed matrix 𝐖𝐒 ...
> (𝐖𝐒):,i = 𝐖:,i 𝐒_ii ... 𝐒_ii⁻¹ scales the i-th channel of the activation

Paper, magnitude rule for S (asvd_2312.05821.md:254, Eq.8):

> 𝐒_ii := (1/n Σ_j |𝐗_ij|)^α

obs: ASVD's mechanism is: build a diagonal (or Cholesky) scaling `S` from activation
statistics, decompose the SCALED matrix `W S`, then fold `S^-1` back into `V`. The
SVD basis is changed by the activation statistics. ASVD's default magnitude rule is
absolute-MEAN per channel raised to alpha (Eq.8), with alpha=0.5 in their experiments
(asvd_2312.05821.md:315).

obs: Our code does plain `torch.linalg.svd(W_orig)` (antipasto.py:153) on the raw,
UNscaled weight. There is no `S` whitening matrix, no `W S`, no `S^-1` fold-back.
Activations enter ONLY through the post-hoc selection score (antipasto.py:154-159),
never into the decomposition basis.

inf: So antipasto does NOT implement ASVD whitening. It borrows one narrow idea: that
an outlier-sensitive (L2/rms) pooling of activations is the right statistic, vs the
outlier-robust mean-abs. That is the `act_pool` knob (antipasto.py:51-52, 155-158).
ASVD itself uses mean-abs (Eq.8); the "rms = ASVD intuition" comment is a slight
liberty -- ASVD's STATED motivation is absorbing activation OUTLIERS
(asvd_2312.05821.md:113, 132), and rms/L2 is more outlier-sensitive than mean-abs, so
the rms choice is "in the spirit of ASVD's outlier-awareness," even though ASVD's own
formula uses mean-abs. The docstring is careful: "'rms' is outlier-sensitive (ASVD
intuition)" (antipasto.py:51) and the corda docstring (antipasto.py:99) explicitly
says re-orienting the basis "is CorDA -> antipasto_corda.py," i.e. antipasto does NOT
whiten.

VERDICT: DEVIATES-OK / honest. Citation is for INTUITION (outlier-sensitive pooling),
not IMPLEMENTATION (whitening), and the word "intuition" is right there in the
comment. No over-claim that antipasto performs activation-aware SVD. Minor nit (not a
bug): ASVD's own scaling statistic is mean-abs, so attributing rms specifically to
ASVD is loose; the honest attribution is "ASVD-style outlier-awareness, but
L2-pooled." Optional one-word fix, not required.

---

## 3. PiSSA: top-r init vs training the components

Paper, PiSSA abstract (pissa_2404.02948.txt:13-18):

> PiSSA ... initializes the adaptor matrices A and B with the principal components
> of the original matrix W, and put the remaining components into a residual matrix
> W_res ∈ R^{m×n} which is frozen during fine-tuning. ... PiSSA updates the
> principal components while freezing the "residual" parts.

Paper, PiSSA Table 1 (pissa_2404.02948.txt:70-99):

> A = U[:,:r] S^{1/2}[:r,:r];  B = S^{1/2}[:r,:r] V^T[:,:r];
> W_res = U[:,r:] S[r:,r:] V^T[:,r:];  "Fine-tunes principal parts freezing W_res."

obs: PiSSA makes the top-r principal components `A,B` (i.e. `U_r, S_r, V_r`)
TRAINABLE and freezes `W_res = W - U_r S_r V_r`. The singular vectors themselves move
during fine-tuning.

obs: Our init (antipasto.py:78-87) takes the same top-r SVD and the same residual:
`W_res = W - (U_r * S_r) @ Vh_r`, written into `layer.weight` (antipasto.py:86-87),
matching PiSSA's `W_res`. BUT `lora_U, lora_S, lora_Vh` are registered
`trainable=False, as_buffer=True` (antipasto.py:64-66); the ONLY trainable parameter
is `lora_g` (antipasto.py:68), a per-direction gain. The forward (antipasto.py:195)
keeps `U, S, Vh` frozen and only learns `S_eff = S*(1+ELU(coeff*g))`.

inf: So antipasto shares PiSSA's INITIALIZATION (top-r SVD + frozen residual) but NOT
its training target. PiSSA trains the full `U,S,V`; antipasto freezes the basis and
learns only a scalar gain per direction. These are different methods; antipasto is
much more constrained (r+r... actually r buffers + r trainable scalars).

obs: The docstring does NOT over-claim. It cites PiSSA precisely as "top-r SVD init"
(antipasto.py:21, 95: "init(): top-r by S alone (PiSSA-style)") and the init() error
message says "mutates layer.weight into W_res (like PiSSA)" (antipasto.py:75) --
scoped to the W_res construction, not to training the components. Line 14-15 of the
module docstring states the basis is frozen and only the gain is learned, the opposite
of PiSSA's claim, so no reader would conflate them.

VERDICT: MATCHES (citation correctly scoped). PiSSA is invoked only for the top-r SVD
init / W_res residual idea, explicitly "PiSSA-style," and the docstring repeatedly
states the basis is FROZEN -- no false PiSSA-equivalence. Not an over-claim.

---

## 4. SVD sign disambiguation (the user's specific question)

obs: None of the three papers canonicalizes singular-vector signs.
- Wanda never decomposes via SVD; it scores `|W_ij|·‖X_j‖₂` on raw weights. Sign is
  irrelevant by construction (absolute value). No svd_flip.
- ASVD does SVD on `W S` (asvd_2312.05821.md:217) but its objective is the
  reconstruction `U_k Σ_k V_k^T` and the Frobenius output error `‖ΔY‖_F`
  (asvd_2312.05821.md:186, 260). A simultaneous sign flip of column `U[:,i]` and row
  `V[:,i]^T` leaves the product (hence the reconstruction and the error) invariant, so
  ASVD has no reason to canonicalize and the paper does not mention sign/`svd_flip`.
- PiSSA sets `A = U[:,:r] S^{1/2}`, `B = S^{1/2} V^T[:,:r]` (pissa_2404.02948.txt:71-76).
  The product `A B = U_r S_r V_r^T` is sign-invariant under a paired column/row flip,
  and PiSSA TRAINS `A,B` afterward, so an initial sign is just a starting point. No
  sign canonicalization in the paper.

obs: Our code performs NO sign flip anywhere (no svd_flip, no max-abs-positive
convention). `torch.linalg.svd` returns whatever signs LAPACK gives.

inf: Omitting sign canonicalization is correct here, because every place the signs
could matter is sign-invariant:
- `S > 0` always (singular values are nonnegative), and the gain rides on `S`:
  `S_eff = S*(1+ELU(coeff*g))` (antipasto.py:195). `1+ELU(.) > 0`, so `S_eff > 0`
  regardless of `U/Vh` sign. The reconstruction `((x @ Vh^T) * S_eff) @ U^T`
  (antipasto.py:197-198) is invariant under a paired flip of `Vh[i]` and `U[:,i]`
  because the flips cancel in the rank-1 term `(x·Vh[i]) S_eff[i] U[:,i]`.
- The selection score uses `|X @ Vh[i]|` via `proj.pow(2)` or `proj.abs()`
  (antipasto.py:156-158), both even in the sign of `Vh[i]`. So a flipped `Vh[i]`
  gives the identical score.
- `lora_g` init is 0 (antipasto.py:68), a sign-symmetric starting point; the learned
  gain multiplies `S` (positive), not `U/Vh`, so its meaning does not depend on basis
  sign.

inf: A sign convention WOULD matter if antipasto ever (a) compared `U`/`Vh` across
layers/checkpoints, (b) initialized `g` from a signed activation projection (e.g.
`X @ Vh` without abs), or (c) added the rank-1 terms with separate trainable signs.
It does none of these. Re: Bro et al. 2008 (sign-determination by data alignment) --
none of the three cited papers use it, and antipasto does not need it.

VERDICT: MATCHES (omission is correct). No cited paper canonicalizes signs, and our
two sign-touching quantities (`S`-rode gain, `|X@Vh|` score) are both sign-invariant.
Adding svd_flip would be dead code here.

---

## 5. The "1+ELU" gain: attribution

obs: The `S_eff = S*(1+ELU(coeff*g))` reparameterization (antipasto.py:7, 195) is not
in Wanda, ASVD, or PiSSA -- none of them learn a per-direction gain at all (Wanda
prunes, ASVD truncates, PiSSA trains full A,B). The module header attributes the
overall method to "wassname 2026 https://arxiv.org/abs/2601.07473" (antipasto.py:3)
and "paper: https://github.com/wassname/AntiPaSTO" (antipasto.py:18).

obs: The Refs block (antipasto.py:17-21) lists Wanda/ASVD under "selection" and PiSSA
under "top-r SVD init" only. Neither the `1+ELU` line nor the forward() rationale
(antipasto.py:188-194) cites any of the three papers for the gain. The gain is
presented as the authors' own.

VERDICT: MATCHES (no false attribution). The 1+ELU gain is correctly presented as the
authors' own contribution; the three citations are confined to selection and init.
(Per instructions, the linear/exp/tanh rationale comment at antipasto.py:189-194 is
intentional and not flagged.)

---

## 6. Citations: arXiv ids, surnames, years

obs (from the source files' own headers/URLs):
- Wanda: `wanda_2306.11695.md:22-30` -- "A Simple and Effective Pruning Approach...",
  Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter; arXiv html id `2306.11695v3`.
  Code -> antipasto.py:20 "Wanda (Sun+ 2023, arXiv:2306.11695)". Sun, 2023, id match.
- ASVD: `asvd_2312.05821.md:26-68` -- "ASVD: Activation-aware Singular Value
  Decomposition...", Zhihang Yuan (first author) et al.; arXiv html id `2312.05821v5`.
  Code -> antipasto.py:20 "ASVD (Yuan+ 2023, arXiv:2312.05821)". Yuan, 2023, id match.
- PiSSA: `pissa_2404.02948.txt:1-44` -- "PiSSA: Principal Singular Values and Singular
  Vectors Adaptation...", Fanxu Meng, Zhaohui Wang, Muhan Zhang; NeurIPS 2024;
  "arXiv:2404.02948v4". Code -> antipasto.py:21 "PiSSA (Meng+ 2024, arXiv:2404.02948)".
  Meng, 2024, id match.

inf: All three (surname, year, arXiv id) check out against the papers' own front
matter. Wanda and ASVD first appeared on arXiv in 2023 (v1), PiSSA is NeurIPS 2024 --
the "+ year" tags are the submission years, which is the conventional choice.

VERDICT: MATCHES (all citations correct). No CITATION-WRONG.

---

## Bottom line

No real math/algorithm bugs and no dishonest citations: Wanda/ASVD are honestly cited
as "-style"/"intuition" (selection + outlier-aware pooling, not literal pruning or
whitening), PiSSA is correctly scoped to the top-r SVD/W_res init (not training the
components), the no-sign-flip choice is correct because every sign-sensitive quantity
is sign-invariant, the 1+ELU gain is the authors' own and not mis-attributed, and all
three arXiv ids/surnames/years match.
