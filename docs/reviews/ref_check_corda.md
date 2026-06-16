# Ref-check: `antipasto_corda.py` vs CorDA (Yang+ 2024, arXiv:2406.05223)

Reviewer: skeptical ML reviewer. Scope: math/algorithm fidelity + citation only.
Files:
- impl: `/media/wassname/SGIronWolf/projects5/2026/lite/lora-lite/src/lora_lite/variants/antipasto_corda.py`
- paper: `/media/wassname/SGIronWolf/projects5/2026/lite/lora-lite/docs/papers/md/corda_2406.05223.md`
- secondary: `/media/wassname/SGIronWolf/projects5/2026/lite/lora-lite/docs/papers/md/asvd_2312.05821.md`

Pre-note on the whitening question (drives points 2,3): PEFT's CorDA does `C = L L^T`
(Cholesky), `SVD(W L)`, then unwinds with `L^{-1}`. Our code does `C^{1/2}` symmetric via
`eigh`, `SVD(W C^{1/2})`, unwinds with `C^{-1/2}`. The paper's *prose* says `SVD(W C)` with
`C^{-1}` (full covariance, Eq. 2-3), which is a THIRD object. The distinction matters and is
worked through in point 2.

---

## 1. Covariance definition + which-half (KPM vs IPM)

> **Paper, Sec. 3.2 (Context-Oriented Decomposition):** "Denote $X \in \mathbb{R}^{d_{in}\times BL}$
> as the input activation of a linear layer ... We have the covariance matrix
> $C = XX^T \in \mathbb{R}^{d_{in}\times d_{in}}$."

> **Paper, Sec. 3.4 (Mode 2: Instruction-Previewed Adaptation):** "we use the first $r$
> components with the largest $r$ singular values ... $B = U_{[:,:r]}\sqrt{\Sigma}_{[:r]}$,
> $A = \sqrt{\Sigma}_{[:r]}(V^T C^{-1})_{[:r,:]}$" — covariance from "instruction and response
> from the training data used for fine-tuning".

Observation: `C = XX^T` is the un-centered input second moment (a Gram matrix over tokens),
NOT a mean-centered covariance, and NOT normalized by token count. CorDA's KPM (Mode 1, Eq. 4)
keeps the SMALLEST r (freezes the principal/knowledge directions); IPM (Mode 2, Eq. 5) trains
the LARGEST r.

Code (`antipasto_corda.py:110-113, 145, 151-153`):
```python
g = x.T @ x                                   # (d_in, d_in)  -> sum_tokens x x^T  (XX^T)
cov[name] = ... cov[name] + g
...
C = cov[name] / cnt[name]                      # normalized to a MEAN second moment E[x x^T]
...
Ut, St, Vht = torch.linalg.svd(W_orig @ Chalf, ...)
Ur = Ut[:, :r]; Sr = St[:r]                    # keeps TOP-r
```

Inference: Code accumulates `sum x x^T` = the paper's `XX^T` exactly (same un-centered second
moment). The `/ cnt` rescale (line 145) is a global positive scalar on `C`; it multiplies every
singular value of `W C^{1/2}` by `sqrt(1/cnt)` and leaves the singular VECTORS (hence the top-r
subspace, U, P-directions) untouched. The absolute `S` differs by `sqrt(1/cnt)`, but `S` is
only a frozen init for the trainable gain `g` (`S_eff = S*(1+ELU(coeff*g))`), so the scale is
absorbed and behaviorally inert.

Keeping TOP-r = the paper's IPM (Mode 2). Code collects cov on `calibration_data` passed at
attach-time; the docstring calls it "downstream-task samples" — consistent with IPM (instruction
data orients the decomposition, train the largest r). The code does NOT implement KPM (bottom-r).

VERDICT: **MATCHES** (covariance = paper's `XX^T`; `/cnt` is an inert global scale; TOP-r is
IPM, which the code is — KPM simply not implemented, not a discrepancy).

---

## 2. Whitening object: `SVD(W C^{1/2})` vs paper's `SVD(W C)` vs PEFT's `SVD(W L)`

> **Paper, Sec. 1 / Eq. 2:** "$\verb|SVD|(WC) = U\Sigma V^T = \sum_{i=1}^R \sigma_i \mathbf{u}_i \mathbf{v}_i^T$"
> **Paper, Eq. 3:** "$\hat W = \verb|SVD|(WC)\,C^{-1} = U\Sigma(V^T C^{-1})$".

Code (`antipasto_corda.py:146-154`):
```python
lam, Q = torch.linalg.eigh(C)
lam = lam.clamp_min(0) + eps
Chalf    = (Q * lam.sqrt())  @ Q.T            # symmetric C^{1/2}
Cinvhalf = (Q * lam.rsqrt()) @ Q.T            # symmetric C^{-1/2}
Ut, St, Vht = torch.linalg.svd(W_orig @ Chalf, full_matrices=False)
...
Pr = (Vht[:r] @ Cinvhalf)
```

This is the crux. Three candidate objects:

(a) Paper literal `SVD(W C)`: C is symmetric PSD, so `W C` has singular values = those of
`W C` directly. Since `C = C^{1/2} C^{1/2}`, `W C = (W C^{1/2}) C^{1/2}`. The singular values of
`W C` are NOT equal to those of `W C^{1/2}` in general (they are squared-ish in the C-spectrum:
`W C` weights directions by `lambda_i`, `W C^{1/2}` by `sqrt(lambda_i)`). So the paper's literal
`SVD(WC)` and our `SVD(W C^{1/2})` give DIFFERENT singular values and DIFFERENT top-r subspaces.

(b) PEFT's `SVD(W L)`, `C = L L^T` Cholesky. `L` is a (different) square-root factor of C.

(c) Our `SVD(W C^{1/2})`, symmetric square root.

Key linear-algebra fact for (b) vs (c): both `L` and `C^{1/2}` satisfy `M M^T = C`. Any two such
factors relate by `C^{1/2} = L Q` for some orthogonal `Q` (polar/QR freedom). Then
`W C^{1/2} = (W L) Q`. Right-multiplying by orthogonal `Q` leaves SINGULAR VALUES identical and
LEFT singular vectors `U` identical; only the right singular vectors rotate (`V_sym = Q^T V_chol`).
Therefore:
- captured top-r left-subspace `U_r`: **identical** between (b) and (c).
- singular values `S_r`: **identical** between (b) and (c).
- the input-side row space of `Pr = Vht[:r] C^{-1/2}` vs PEFT's `(V^T)[:r] L^{-1}`: both equal
  `(W L)`/`(W C^{1/2})` top-r right-vectors unwound by the matching inverse factor. Since
  `V_sym^T C^{-1/2} = (Q^T V_chol)^T (L Q)^{-1} = V_chol^T Q Q^{-1} L^{-1} = V_chol^T L^{-1}`,
  the projector `Pr` is **identical** between (b) and (c) too.

So our symmetric-sqrt form is exactly PEFT's Cholesky form for both (a) captured top-r subspace
and (b) the reconstruction. They are interchangeable — confirmed rigorously, not hand-waved.

But neither equals the paper's LITERAL `SVD(WC)` (object (a) above). The PEFT reference impl and
our code both implement the square-root-whitening variant, which is the numerically sane reading
(`WC` mixes a `d_out x d_in` weight with a `d_in x d_in` covariance giving the "wrong" energy
weighting `lambda` instead of `sqrt(lambda)`; the Eckart-Young-optimal-under-`x~N(0,C)` story the
docstring tells is the `sqrt` version). PEFT — the authors' own HF integration linked in the
paper header — uses Cholesky `W L`, confirming the paper's `SVD(WC)` prose is loose and the
intended/implemented object is the square-root one.

Reconstruction exactness (point 3 overlaps): code line 155 `W_res_new = W_orig - (Ur*Sr)@Pr` and
`W_orig = U_r S_r P_r + W_res`. With `Pr = Vht[:r] C^{-1/2}`:
`(Ur Sr Vht[:r]) C^{-1/2} = [SVD(W C^{1/2}) truncated] C^{-1/2}`. The FULL (untruncated) product
`U S Vht C^{-1/2} = (W C^{1/2}) C^{-1/2} = W` exactly. Truncated to r it is the top-r piece, and
`W_res` carries the exact remainder by subtraction (same trick the paper uses in Eq. 4:
"$W' = W - BA$ ... to avoid the numerical error"). Reconstruction `W = W_res + U_r S_r P_r` is
exact by construction.

VERDICT: **DEVIATES-OK**. Symmetric `C^{1/2}` differs from the paper's *printed* `SVD(WC)` but is
provably identical to PEFT's `SVD(WL)` Cholesky reference for both top-r subspace and
reconstruction. The paper's literal `WC` is the loose/wrong-energy form; matching PEFT (authors'
own impl) is the correct choice. Worth a one-line code comment noting the symmetric-sqrt vs
paper-prose `WC` discrepancy so a future reader is not confused.

---

## 3. Projector `Pr = Vht[:r] @ Cinvhalf` and reconstruction exactness

> **Paper, Eq. 3:** "$\hat W = U\Sigma(V^T C^{-1})$", with $\hat{\mathbf v}_i^T$ = i-th row of
> $V^T C^{-1}$.
> **Paper, Eq. 5 (IPM):** "$A = \sqrt{\Sigma}_{[:r]}(V^T C^{-1})_{[:r,:]}$".

Code (`antipasto_corda.py:154-155`):
```python
Pr = (Vht[:r] @ Cinvhalf)                      # (r, d_in)
W_res_new = (W_orig - (Ur * Sr) @ Pr)
```

Observation: paper unwinds with `C^{-1}` (because it whitened by full `C`); code unwinds with
`C^{-1/2}` (because it whitened by `C^{1/2}`). These are the matched inverse-factors for their
respective forward objects — paper: `SVD(WC) C^{-1}` recovers `W`; code:
`SVD(W C^{1/2}) C^{-1/2}` recovers `W`. Both are self-consistent; the code's is the PEFT-equivalent
square-root form (point 2). The projector is `(d_in)`-side, oblique (rows not orthonormal because
`C^{-1/2}` skews them) — matches the paper's `\hat v_i` being rows of a whitened `V^T C^{-...}`.

Note one structural difference vs paper's adapter split: paper puts `sqrt(Sigma)` into BOTH `B`
and `A` (`B = U sqrt(Sigma)`, `A = sqrt(Sigma) V^T C^{-1}`) so the trained product `B*A* `
re-learns the magnitude. Our code keeps `S` whole on the projector side via the runtime gain
`S_eff` and an orthonormal `U` (`y + ((x@P^T)*S_eff)@U^T`). This is an intentional architectural
choice (gain-reweighting antipasto, not LoRA-style free B/A retraining), not a fidelity bug —
the captured subspace and the exact-reconstruction identity are unchanged.

VERDICT: **MATCHES** (reconstruction exact; projector is the matched square-root unwind; the
S-on-one-side split is an intentional antipasto design difference, not a CorDA discrepancy).

---

## 4. SVD sign disambiguation

Paper: searched for "sign", "svd_flip", "flip" — **no mention**. CorDA never canonicalizes
singular-vector signs. Eq. 2-5 use `U`, `V^T` straight from SVD; the reconstruction `U S V^T C^{-1}`
is sign-invariant anyway (a sign flip on column `u_i` and row `v_i^T` cancels in `u_i sigma_i v_i^T`).

Code: no `svd_flip` / max-abs / data-alignment anywhere (`init` line 82, `group_init` line 151).
Docstring/param note lines 68-70: "No sign-symmetry hack needed (1+ELU is sign-preserving, basis
frozen)".

Inference: Our forward is `S_eff = S*(1 + ELU(coeff*g))`, `g` trained from 0, gain rides on `S>0`.
A sign flip on `u_i` flips the corresponding row of `P` (=`v_i`-derived) too, so the rank-1 term
`u_i (S_eff)_i (P row_i)` is invariant to the joint sign — exactly as in the paper's
sign-invariant `u_i sigma_i v_i^T`. `g` is a scalar magnitude per direction, not tied to any
fixed sign convention, and the basis is frozen after `group_init`. So omitting sign
canonicalization is correct: neither the paper needs it nor do we.

(Caveat, not a bug: if any downstream analysis inspected `U` or `P` rows individually and assumed
a sign convention, it would break. None does here — the gain is sign-agnostic.)

VERDICT: **MATCHES** (paper does not canonicalize signs; our reconstruction + sign-invariant gain
make omission correct).

---

## 5. Citation

> **Paper header:** "CorDA: Context-Oriented Decomposition Adaptation ... Yibo Yang, Xiaojie Li,
> Zhongzhu Zhou, Shuaiwen Leon Song, Jianlong Wu, Liqiang Nie, Bernard Ghanem". arXiv:2406.05223.

Code docstring (`antipasto_corda.py:5, 20`): "CorDA (Yang+ 2024, arXiv:2406.05223)".

Observation: first author surname is **Yang** (Yibo Yang). arXiv id 2406.05223 matches the paper
file and header. Year 2024 matches (NeurIPS 2024; arXiv June 2024).

Secondary cite (`antipasto_corda.py:103-104`): "Yuan+ 2023, ASVD, arXiv:2312.05821 is the diagonal
case". ASVD paper confirms a diagonal scaling matrix `S` (Sec 3.3, "set the transform matrix as a
diagonal matrix", Eq. 8) as the simple case, and a Cholesky `L` of `XX^T` as the better variant
(ASVD+, lines 261-267) — so calling ASVD "the diagonal case" of covariance-whitening is accurate.
ASVD first author is Zhihang Yuan; arXiv 2312.05821 correct.

VERDICT: **MATCHES** (Yang+ 2024 / 2406.05223 correct; ASVD Yuan+ 2023 / 2312.05821 correct;
"diagonal case" characterization accurate).

---

## Bottom line

No real bugs. The one substantive math note: code whitens with symmetric `C^{1/2}` (eigh), which
is provably identical to PEFT's Cholesky `W L` reference (same top-r U/S, same projector, exact
reconstruction) but differs from the paper's loosely-printed `SVD(WC)` full-covariance form — an
intentional, correct deviation; add a one-line comment flagging it. Everything else matches.
