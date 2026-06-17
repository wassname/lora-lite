# Reference-fidelity check: `antipasto_ablate.py` vs Arditi+ 2024 (arXiv:2406.11717)

Scope: math/algorithm fidelity + citation only. Fail-fast research code, so no
defensive-programming, None-check, or backward-compat flags. Secondary `cov_orient`
block checked against CorDA (arXiv:2406.05223).

Files:
- impl: `/media/wassname/SGIronWolf/projects5/2026/lite/lora-lite/src/lora_lite/variants/antipasto_ablate.py`
- paper: `/media/wassname/SGIronWolf/projects5/2026/lite/lora-lite/docs/papers/md/arditi_2406.11717.md`
- secondary: `/media/wassname/SGIronWolf/projects5/2026/lite/lora-lite/docs/papers/md/corda_2406.05223.md`

---

## 1. Ablation operator: residual-stream projection vs per-layer output-singular projection

Paper, directional ablation operator (arditi md:153-157):

> 𝐱′ ← 𝐱 − 𝐫̂ 𝐫̂⊺ 𝐱.  [...] We perform this operation at every activation 𝐱ᵢ(l) and 𝐱̃ᵢ(l), across all layers l and all token positions i. This effectively prevents the model from ever representing this direction in its residual stream.

Paper, the weight-space equivalent (the form our code structurally resembles), Eq 5 (arditi md:246-248):

> W_out′ ← W_out − 𝐫̂ 𝐫̂⊺ W_out. [...] the matrices that write to the residual stream are: the embedding matrix, the positional embedding matrix, attention out matrices, and MLP out matrices. Orthogonalizing all of these matrices [...] with respect to the direction 𝐫̂ effectively prevents the model from ever writing 𝐫̂ to its residual stream.

Code core (antipasto_ablate.py:182-186):

```
h = (x @ Vh.T) * S                          # (..., r) output S-coords
proj = h @ Chat                             # (..., k)
h = h - coeff * (proj * alpha) @ Chat.T     # contractive removal in r-dim U-space
return y + h @ U.T                          # map back to d_out
```

Observation (math): Chat are orthonormal in the r-dim coordinate space that indexes
the columns of U (lora_U is (d_out, r), Chat is (r, k); ablate at :183-185 happens on
h which lives in this r-space, before `h @ U.T` lifts to d_out at :186). Since U has
orthonormal columns, a unit direction `c` in r-space maps to the unit direction
`U @ c` in d_out (residual) space, and the projector obeys
`U (I - Chat Chatᵀ) Uᵀ = I_{d_out} - (U Chat)(U Chat)ᵀ` on the column space of U. So
within the rank-r output subspace this is exactly an Arditi-style outer-product
projector `I - d̂ d̂ᵀ` with `d̂ = U @ ĉ`.

Inference (the mismatch, three parts):
1. Per-layer, not shared. Arditi ablates ONE direction 𝐫̂ read off the residual
   stream and applies the SAME 𝐫̂ at every component (Eq 5 lists embed/attn-out/
   mlp-out all orthogonalized w.r.t. the same 𝐫̂). Our code learns a SEPARATE
   direction per target layer (lora_c is per-layer, param_specs:70). These are
   different objects: Arditi's d̂ is global; ours is a bouquet of per-layer d̂.
2. Restricted to top-r U-span. Arditi's projector acts on the full d_model residual
   vector. Ours can only remove components that lie in span(U[:, :r]); anything in
   W_res (the frozen remainder, init:86-87) is untouched. With cov_orient=False this
   is plain-SVD top-r, which need not contain the behavior direction (the docstring
   itself flags this: ":52-53 measured 1.00 vs 0.65 capture at r=16").
3. Output-side only vs residual-stream. Arditi's residual-stream ablation (Eq 4) and
   its weight-equivalent (Eq 5) zero the direction the layer WRITES. Our `y + h@U.T`
   subtracts from this layer's additive output contribution, which IS a residual
   writer for down_proj/o_proj. The docstring scopes the variant to exactly those
   ("target residual writers (down_proj, o_proj)", :17-18), so for those targets the
   output-side framing is the correct analogue of Eq 5. For a non-writer Linear it
   would not be (correctly excluded by design).

VERDICT: DEVIATES-OK. The single-direction outer-product structure is faithfully
reproduced inside each layer's output subspace, and for residual-writer targets the
output-side action matches Eq 5's "don't write 𝐫̂". The deviations (per-layer learned
direction instead of one shared difference-in-means direction; confined to top-r) are
deliberate design choices, not the paper's algorithm, and the docstring's framing
("trainable form of directional ablation") signals the analogy rather than claiming
identity. No bug, but the correspondence is "Arditi-style projector, per-layer,
within rank-r" — not the verbatim global single-direction ablation.

---

## 2. Contraction / "ablation cannot amplify"

Paper does not state a contraction theorem in the algebraic sense, but the operator
itself is a projection. Closest textual support, Eq 4 (arditi md:155) is the
orthogonal projector `I - 𝐫̂𝐫̂ᵀ` (idempotent, eigenvalues {0,1}); and the paper
repeatedly frames ablation as removal/erasure, never amplification:

> Directional ablation "zeroes out" the component along 𝐫̂ for every residual stream activation (arditi md:151)

> these loss metrics suggest that directional ablation is more surgical than activation addition based methods (arditi md:323)

(The paper's amplification is a SEPARATE intervention — activation *addition*, Eq 3,
md:144 — confirming ablation is the non-amplifying side.)

Code (forward:179, 185 and clamp):

```
alpha = layer.lora_alpha.to(x.dtype).clamp(0.0, 1.0)   # :179
h = h - coeff * (proj * alpha) @ Chat.T                # :185
```

Observation (math): with coeff=1 the core operator on r-space is
`M = I - Chat diag(alpha) Chatᵀ`. Chat orthonormal ⇒ eigenvalues are `1 - alphaⱼ`
along each chat_j and `1` on the orthogonal complement. With alpha clamped to [0,1]
(:179), every eigenvalue lies in [0,1]. So M is a contraction (operator norm ≤ 1):
it cannot amplify. Arditi's exact projector is the alpha=1 endpoint (eigenvalue 0 =
full erasure); alpha<1 is partial ablation, still non-amplifying.

Caveat (the code's own warning, :20-21, :49): coeff is OUTSIDE the clamp. coeff<0
flips the sign and ADDS the direction back (eigenvalue `1 + |coeff|·alpha` > 1), and
coeff>1 over-subtracts (eigenvalue `1 - coeff·alpha` can go negative, |·|>1). The
contraction guarantee holds only for coeff ∈ [0,1], which the docstring states
explicitly ("<0 adds the direction back (the side that can grow, so bound coeff
there)"). That is correct and self-documented, not a silent bug.

VERDICT: MATCHES (for the documented operating range coeff∈[0,1], alpha∈[0,1]).
Eigenvalues in [0,1] confirmed; the alpha clamp is the load-bearing line. The coeff<0
amplifying branch is intentional and flagged.

---

## 3. Direction source: fixed difference-in-means vs trainable

Paper, the direction is a FIXED difference-of-means, selected once (arditi md:124, 134):

> We then compute the difference-in-means vector 𝐫ᵢ(l) = 𝛍ᵢ(l) − 𝛎ᵢ(l).

> We notate the selected vector as 𝐫, and its corresponding unit-norm vector as 𝐫̂.

i.e. 𝐫̂ is computed from mean(harmful) − mean(harmless) and then frozen; the paper
performs NO gradient descent on it ("does not require gradient-based optimization",
md:236).

Code: lora_c is a trainable parameter (param_specs:70 `trainable` default; random
normal init), optionally warm-started from a contrastive S-space direction dS
(docstring init:88-90 "group_init() should warm-start lora_c from the S-space
contrastive direction dS").

Observation: ours is structurally the same OBJECT (a unit direction that gets
projected out) but obtained by a different procedure (SGD, optionally seeded from a
contrastive diff) rather than a closed-form difference-of-means.

Inference: calling it "the trainable form of directional ablation (Arditi+ 2024)"
(:16-17) is defensible AS A FRAMING: the operator (Eq 4/5 outer-product removal) is
Arditi's; the novelty claimed is making the direction learnable. The optional
warm-start from a contrastive dS is even closer to Arditi's diff-of-means seed. This
is a fair "trainable variant of X" claim, not a misattribution. It would be wrong to
claim Arditi's *method* (which is explicitly gradient-free) — but the code claims the
*ablation operator*, not the extraction method.

VERDICT: DEVIATES-OK. "Trainable form of directional ablation" is an honest framing:
same operator, deliberately different (learned) direction source. Recommend the
docstring keep the word "form"/"trainable" prominent so it is not read as
reproducing Arditi's gradient-free diff-of-means extraction.

---

## 4. SVD sign disambiguation

Paper: SILENT on singular-vector sign. Arditi never does an SVD of a weight to get
its ablation direction — 𝐫̂ comes from difference-of-means (md:117-124), and the
weight-orthogonalization Eq 5 (md:246) uses the outer product 𝐫̂𝐫̂ᵀ. So the paper
offers no quote on SVD sign; I reason from the math.

Code: ablation core is the outer product `Chat Chatᵀ` (forward:183-185), and U is
orthonormal from `torch.linalg.svd` (init:80).

Observation (math):
- Chat enters only as `Chat Chatᵀ` (proj `h @ Chat` then `@ Chat.T`, :183-185).
  Flipping any column sign `chat_j → −chat_j` leaves `chat_j chat_jᵀ` unchanged ⇒
  the operator is sign-invariant in c. (lora_c is trainable anyway, so its sign is
  not even an SVD artifact.)
- U enters only as `h @ U.T` (:186) AND the projector identity from point 1 is
  `(U c)(U c)ᵀ`; a sign flip `u_i → −u_i` with the matching `vh_i → −vh_i` leaves
  `W = (U S) Vh` invariant (init:86) and leaves `(U c)(U c)ᵀ` invariant. So neither
  the reconstructed weight nor the ablation projector depends on per-vector sign.
- S is non-negative by SVD definition; no sign issue.

Inference: no sign canonicalization is needed anywhere in this file. The ablation is
a quadratic form in both the (trainable, hence sign-free) direction and the
orthonormal basis, and every place U/Vh appear they appear in sign-paired products
or outer products. This is the correct situation (a basis/span use, not a
sign-sensitive coordinate use).

VERDICT: MATCHES (math-derived; paper silent on SVD sign, stated). No
canonicalization required and none missing.

---

## 5. Citation check

Docstring (:16-17, :23):

> This is the trainable form of directional ablation (Arditi+ 2024 [...]). Refs: [...] directional ablation Arditi+ 2024 arXiv:2406.11717.

Paper title/authors (arditi md:1-21):

> # Refusal in Language Models Is Mediated by a Single Direction
> Andy Arditi [...] Oscar Obeso [...] Aaquib Syed [...] Daniel Paleka [...] Nina Rimsky [...] Wes Gurnee [...] Neel Nanda

Observation: arXiv:2406.11717 = "Refusal in Language Models Is Mediated by a Single
Direction", first author surname Arditi. The term "directional ablation" is the
paper's own (§2.4 heading, md:148). The arXiv id, surname, year, and method name all
match.

cov_orient / CorDA attribution. Code config comment (:50) and group_init docstring
(:94) say "CorDA" by name:

> CorDA-orient the basis from input covariance (group_init [...]) (:50)
> re-orient each target's SVD by input covariance C=E[x xᵀ] (CorDA) (:94)

CorDA paper core (corda md:23, line "C=XXᵀ", and `SVD(WC)=UΣVᵀ`, reconstruct
`Ŵ=UΣVᵀC⁻¹`):

> obtain the covariance matrix of the input activation [...] C=XXᵀ [...] perform singular value decomposition for the weight multiplied by the covariance matrix, i.e. SVD(WC)=UΣVᵀ [...] the inverse of these covariance matrices is multiplied with the decomposed components to hold the same inference result with the original model

Observation: CorDA IS attributed by name (:50, :94). One math nuance worth recording
(not a citation error): CorDA whitens with `C` on one side and reconstructs with
`C⁻¹` (`SVD(WC)`, then `VᵀC⁻¹`). The code instead uses SYMMETRIC whitening
`SVD(W C^{1/2})` with `Pr = Vh C^{-1/2}` (group_init:145-150). Both preserve the
forward map (`W C^{1/2} · C^{-1/2} = W`) and both put data-relevant output
directions in the top-r, but the orientation is not bit-identical to CorDA's `WC`.
This is an acknowledged variant (task list #22 "Ablate whitening: C^1/2 (mine) vs C
(PEFT)"), so the "CorDA-orient" name is a fair attribution of the IDEA
(covariance-oriented SVD with inverse-covariance reconstruction), with the symmetric
square-root being this repo's choice.

VERDICT: MATCHES (Arditi citation correct: id, surname, method name). CorDA is
attributed by name; the C^{1/2} symmetric-whitening detail is a labeled variant of
CorDA's C/C⁻¹, not a mis-citation.

---

## Bottom line

No real bugs: the operator reproduces Arditi's outer-product ablation faithfully
inside each layer's rank-r output subspace, is a proven contraction for the documented
coeff∈[0,1]/alpha∈[0,1] range, needs no SVD sign canonicalization, and both Arditi and
CorDA are correctly attributed (the per-layer-learned direction and C^{1/2} whitening
are deliberate, self-documented design choices, not deviations from a claimed
reproduction).
