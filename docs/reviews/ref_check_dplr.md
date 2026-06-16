# Reference-fidelity check: `antipasto_dplr.py` vs LoRA (Hu+ 2021)

Implementation: `/media/wassname/SGIronWolf/projects5/2026/lite/lora-lite/src/lora_lite/variants/antipasto_dplr.py`
Paper: `docs/papers/lora_2106.09685.txt` (arXiv:2106.09685v2)

Scope: only math/algorithm fidelity and citation. Fail-fast research code; defensive-programming gaps are out of scope.

---

## 1. Low-rank core == LoRA's BA (shapes + rank + A/B mapping)

Paper, Section 4.1, lines 199-204:

> For a pre-trained weight matrix W0∈ Rd×k, we constrain its update by representing the latter with a low-rank decomposition W0 + ∆W = W0 + BA, where B∈ Rd×r, A∈ Rr×k, and the rank r≪ min(d, k). [...] For h = W0x, our modiﬁed forward pass yields: h = W0x + ∆W x = W0x + BAx (3)

Code, the core term (file:165):

> `h = p * S_eff + coeff * (p @ A.T) @ B.T               # (..., r)`

with params (file:68-69):

> `lora_A=ParamSpec((k, r), init="kaiming"),`
> `lora_B=ParamSpec((r, k), init="zeros"),`

Observation: here the "input" to the core is `p` of dim `r` (the SVD subspace coordinate, file:160), and the core maps `r -> r`. So the core's effective weight matrix is square `(r, r)` in subspace coords, NOT `(d, k)`. The bottleneck rank is `k = lora_rank` (the paper's `r`).

Shape trace for `core(p) = (p @ A.T) @ B.T` with `p: (..., r)`:
- `A.T: (r, k)`, so `p @ A.T : (..., k)`  -- this is the DOWN-projection `r -> k` (paper's `A: R^{r_paper x k_paper}`, the down-proj).
- `B.T: (k, r)`, so `(p @ A.T) @ B.T : (..., r)` -- this is the UP-projection `k -> r` (paper's `B: R^{d x r_paper}`, the up-proj).

As an operator on a column vector `p`, `core(p) = B @ A @ p` where `A` is `(k, r)` and `B` is `(r, k)`, i.e. the matrix `B@A` of shape `(r, r)` with rank <= `k`. This composes and matches the paper's `BA` form (paper's `B@A` is `(d, k)` rank-`r_paper`; here it is `(r, r)` rank-`k`).

Mapping (ours -> paper, using paper's subscript `_p`):
- our `lora_A (k, r)` <-> paper `A_p (r_p, k_p)`: the down-projection. (our `k` = paper `r_p`; our `r` = paper `k_p`).
- our `lora_B (r, k)` <-> paper `B_p (d_p, r_p)`: the up-projection. (our `r` = paper `d_p`; our `k` = paper `r_p`).
- bottleneck rank: our `lora_rank = k` <-> paper `r`. Docstring/config already say "the low-rank mixing core (LoRA's r, but inside the frozen subspace)" (file:43).

VERDICT: MATCHES. Shapes compose; bottleneck rank is `lora_rank=k`; A is down-proj (`r->k`), B is up-proj (`k->r`), consistent with the paper's `BA` once you note the input is `p` (dim `r`), not `x` (dim `d_in`).

---

## 2. Zero-init identity (which side is zero; kaiming vs Gaussian)

Paper, Section 4.1, lines 205-206:

> We use a random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the beginning of training.

Code (file:68-69):

> `lora_A=ParamSpec((k, r), init="kaiming"),`
> `lora_B=ParamSpec((r, k), init="zeros"),`

Observation: paper zeros `B` (the up-projection) and random-inits `A` (the down-projection). Ours zeros `lora_B`, which I established in point 1 is the up-projection (`k->r`), and random-inits `lora_A`, the down-projection (`r->k`). So the zero is on the up-projection on BOTH sides. The core `(p @ A.T) @ B.T = ... @ 0 = 0` at init regardless of `A`. Combined with `g=0 -> 1+ELU(0)=1`, the adapter output is `h @ U.T` with `h = p*S` = exact reconstruction of the top-r part, so the layer is identity at init. Docstring confirms intent (file:17): "Identity at init (B=0, g=0)".

Inference on kaiming vs Gaussian: the paper's only stated requirement is that the random side be a random init so symmetry is broken and `BA=0` purely from the zero side. Both kaiming-uniform and `N(0, sigma^2)` are zero-mean symmetry-breaking inits; the network output at init is identical (0, because the zero side kills it) and gradients to `B` at step 1 depend only on `A`'s scale, not its distribution family. Kaiming-uniform is the PyTorch `nn.Linear` default and is what the reference `microsoft/LoRA` code actually uses (`kaiming_uniform_(A, a=sqrt(5))`) despite the paper text saying "Gaussian". So this is not a meaningful deviation; if anything it matches the reference implementation more closely than the paper prose.

VERDICT: MATCHES (zero on the up-proj on both sides; identity at init holds). Kaiming-vs-Gaussian on the random side: DEVIATES-OK, immaterial.

---

## 3. Scaling: paper's alpha/r vs our `coeff`

Paper, Section 4.1, lines 206-211:

> We then scale ∆W x by α/r, where α is a constant in r. When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately. [...] This scaling helps to reduce the need to retune hyperparameters when we vary r (Yang & Hu, 2021).

Code (file:155, 165):

> `coeff = float(cfg.coeff)`
> `h = p * S_eff + coeff * (p @ A.T) @ B.T               # (..., r)`

Observation: ours multiplies the core by `coeff` (a runtime scalar, default 1.0, file:47) and has no `alpha/r` factor. `coeff` also scales the gain via `S_eff = S * (1 + ELU(coeff * g))` (file:161), so it is a shared global knob, not a per-adapter LoRA `alpha`.

Inference, init-time identity: the paper's `alpha/r` is a constant multiplier on a quantity (`BA`) that is exactly zero at init. A constant times zero is zero. So `alpha/r` has NO effect on init-time identity, and neither does its absence here -- identity at init is governed entirely by `B=0`, which holds (point 2). The absence of `alpha/r` does not break identity.

Inference, training dynamics: `alpha/r` is a fixed scalar folded into the effective learning rate for the LoRA branch. Its purpose (per the quote and Yang & Hu 2021 / muP) is to keep update magnitudes stable as you sweep `r`, so you do not have to retune LR per rank. Dropping it means: if you sweep `lora_rank` here, the effective step size on the core is not auto-normalized, so the optimal LR may shift with rank. That is a real difference in training dynamics, but it is a tuning-convenience factor, not a correctness bug -- the reachable function class is identical (any `alpha/r` is absorbable into `A`,`B` magnitude and LR). `coeff` here additionally couples gain and core scaling, which is a deliberate design choice (single knob: 0=identity), not the paper's per-branch `alpha`.

VERDICT: DEVIATES-OK. No effect on init-time identity (zero is scale-invariant). For training dynamics it removes the rank-stabilizing `alpha/r` convenience; reachable functions unchanged, but rank sweeps may need LR retuning. Not a bug.

---

## 4. Subspace restriction (deliberate deviation from full-space ΔW)

Paper, Section 4.1, lines 199-200:

> For a pre-trained weight matrix W0∈ Rd×k, we constrain its update by representing the latter with a low-rank decomposition W0 + ∆W = W0 + BA, where B∈ Rd×r, A∈ Rr×k

Observation: LoRA's `BA` lives in the full `(d, k)` space; `B`'s column space and `A`'s row space are free, unconstrained by `W0`. Ours instead acts on `p = x @ Vh.T` (file:160), the projection of the input into the frozen top-r right-singular basis, and writes back via `@ U.T` (file:166), the frozen top-r left-singular basis. So the entire adapter (gain + core) is sandwiched as `U @ (...) @ Vh` and is confined to `W`'s top-r row/column subspace by construction.

Inference, soundness: with `p` in the `Vh`-row-space and output through `U`, the core can only read directions in `W`'s top-r right-singular subspace and only write directions in its top-r left-singular subspace. The `(r,r)` matrix `B@A` mixes WITHIN that subspace -- it can rotate/mix singular direction `i` into singular direction `j` for `i,j <= r`, but cannot inject any component outside the stored `U`/`Vh` span. This is exactly the stated design ("a low-rank mixing core in the frozen SVD basis", file:1; "the rank-k term is LoRA's core ... restricted to W's top-r subspace", file:14-15). It is a deliberate, internally consistent restriction, and the math confirms the core cannot escape the top-r span.

VERDICT: DEVIATES-OK. Stated explicitly as a deliberate deviation (docstring file:1-17); sound -- the `U .. Vh` sandwich provably confines mixing to W's top-r subspace.

---

## 5. SVD sign disambiguation

Relevant code: basis frozen at init from `torch.linalg.svd` (file:78), core learned from zero (`lora_B` zeros, file:69), forward reads/writes through `U`/`Vh` (file:160, 166).

There is no paper quote for this -- it is a property of using SVD vectors as a fixed basis, so I reason from the math.

SVD is sign-ambiguous: for each `i`, flipping `(u_i, v_i) -> (-u_i, -v_i)` leaves `W = sum_i s_i u_i v_i^T` unchanged (the two sign flips cancel). The diagonal gain term is sign-immune already: `p_i = x . v_i`, output contribution `(p_i * S_eff_i) u_i`; flipping both `u_i, v_i` sends `p_i -> -p_i` and `u_i -> -u_i`, product unchanged.

For the core, consider the learned operator in subspace coords `M = B@A` (shape `(r,r)`), so output `= (p^T M) @ U` writing `y += sum_i (pM)_i u_i^T`. Let `D = diag(+-1)` be any per-direction sign flip applied to the frozen basis (`U -> U D`, `Vh -> D Vh`, hence `p -> D p`). The full reconstruction `U D ... D Vh` is invariant for the gain. For the core, the post-training output uses the LEARNED `M`; under a sign flip of the basis, the SAME target output is produced by the gradient-reachable `M' = D M D`. Because `M` is learned from `M=0` (`B=0` at init) with no prior tying it to any particular sign convention, the optimizer is free to land on `M'` instead of `M` -- the loss landscape is identical up to the relabeling `M <-> D M D`. There is no asymmetry (no nonzero init, no regularizer referencing absolute sign, ELU acts only on the separate gain `g` not on the core) that would make one sign choice reachable and the mirror not.

Therefore a global (or per-direction) sign flip of the frozen singular vectors is absorbed by the learned-from-zero core; the represented function class and the optimum are sign-invariant.

VERDICT: MATCHES (no canonicalization needed). Sign ambiguity is absorbed by the from-zero core; confirmed from the math, no paper claim required.

---

## 6. Citation

Code (file:14, file:17, file:19):

> `subspace (Hu+ 2021, arXiv:2106.09685).`  /  `lora.py (low-rank core)`  /  docstring header.

Paper identity (lines 1-3, 67):

> LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS -- Edward Hu, Yelong Shen, ... -- arXiv:2106.09685v2 [cs.CL] 16 Oct 2021.

Observation: first author Hu, year 2021, arXiv ID 2106.09685 all match.

VERDICT: CITATION-CORRECT.

---

## Bottom line

No real bugs: all six points MATCH or DEVIATE-OK (the two intentional deviations -- subspace restriction and dropping alpha/r -- are stated and sound; kaiming-vs-Gaussian is immaterial). Citation correct.
