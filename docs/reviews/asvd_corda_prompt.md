# Review request: CorDA / ASVD covariance-oriented SVD adapter init

You are reviewing the linear-algebra correctness of two PEFT-adapter init routines in a
research codebase. This is a frozen-basis bounded-gain adapter ("AntiPaSTO"): it takes the
top-r SVD of a Linear weight W (d_out x d_in), freezes (U, S, P), and trains only a
per-direction gain g via S_eff = S * (1 + ELU(coeff*g)). At g=0 the adapter must be an
EXACT identity (output equals the original W x).

Two init variants re-orient the SVD basis by the input second moment of calibration data:
- CorDA (Yang+ 2024, arXiv:2406.05223): full covariance C = E[x x^T], via eigh.
- ASVD (Yuan+ 2023, arXiv:2312.05821): diagonal only, M = diag(E[x_i^2]).

The two share one function `_covariance_orient(..., diag)`; only the `diag` flag differs.

## Claims I want you to verify or refute, each with reasoning

1. **Reconstruction is lossless / identity-at-init holds.** After re-orientation, the code
   sets `W_res_new = W_orig - (U_r S_r) P_r` and stores (U_r, S_r, P_r). The forward adds
   `((x @ P^T) * S_eff) @ U^T` to `x @ W_res^T`. At g=0 (S_eff=S_r), is the total output
   exactly `x @ W_orig^T`, in exact arithmetic? Note P_r is the TRUNCATED top-r projector,
   not full rank. Is `W_res_new + U_r S_r P_r == W_orig` exactly, or only approximately?

2. **CorDA whitening form is correct.** The code computes (full case):
   `C^{1/2}, C^{-1/2}` via eigh; `U,S,Vh = svd(W @ C^{1/2})`; `P_r = Vh[:r] @ C^{-1/2}`;
   `U_r = U[:, :r]`, `S_r = S[:r]`. Question: is `U_r diag(S_r) P_r` the rank-r truncation
   that is Eckart-Young optimal for reconstructing W under inputs x ~ N(0, C)? i.e. does
   minimizing `||(W - W_hat) x||` over rank-r W_hat with x~N(0,C) reduce to truncated SVD
   of `W C^{1/2}` followed by right-multiply by `C^{-1/2}`? Show the algebra.

3. **ASVD diagonal form is the consistent diagonal special case.** With `c = E[x_i^2]`
   (a d_in vector), code does `svd(W * c.sqrt())` (broadcast scales COLUMNS of W) and
   `P_r = Vh[:r] * c.rsqrt()` (scales COLUMNS of Vh). Is this exactly variant 2 with C
   replaced by diag(c)? Is the column-broadcast `W * c.sqrt()` equal to `W @ diag(sqrt(c))`?

4. **The eps damping does not break identity.** `lam = lam.clamp_min(0) + eps` (full) and
   `c = (...).clamp_min(0) + eps` (diag). The eps enters BOTH the forward map C^{1/2} used
   in the SVD AND the inverse C^{-1/2} in P. Does the damped C^{1/2} and damped C^{-1/2}
   still compose to identity inside the reconstruction (so claim 1 still holds with eps>0),
   or does eps introduce a reconstruction error? Specifically: the SVD is of `W @ M^{1/2}`
   and P uses the SAME M's `M^{-1/2}`; does `(W M^{1/2}) truncated-then-times M^{-1/2}`
   telescope regardless of what M is, as long as M^{1/2} and M^{-1/2} are true inverses?

5. **Covariance estimator.** `m = x.T @ x` summed over tokens, divided by total token
   count `cnt` = sum of b*s. This is the UNCENTERED second moment E[x x^T], not the
   centered covariance. Is uncentered correct for this use (we want to reconstruct W x well
   on the actual activation distribution, which includes the mean)? Any concern?

6. **Anything wrong, risky, or non-obvious** — numerical (eigh of a d_in x d_in ~3584^2
   moment in fp32), the clamp_min(0) before adding eps, the `cnt < r` guard, dtype round
   trips (buffers bf16, math in fp32), or the oblique P (rows not orthonormal) interacting
   with the gain. Be concrete and cite the line.

Structure findings by severity (blocker / should-fix / nit). If a claim is correct, say so
plainly with the one-line reason; do not invent problems. Answer from the code below; do
not say you will read files.

--- FILE: src/lora_lite/variants/antipasto_corda.py ---
