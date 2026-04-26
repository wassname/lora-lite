"""SVD adapter for bidirectional steering via block-diagonal Cayley rotations.

Flax NNX port.

kernel = U @ diag(S) @ Vh + W_res   (kernel is (in, out), standard Flax convention)
Learnable: delta_s (additive S scaling), rotation_params (block-diagonal V rotation).
alpha scales both: S + alpha*delta_s, U @ R(alpha).

Why Cayley (not Givens or matrix exponential):
Cayley gives exact analytical reversibility: R(-alpha) = R(alpha)^{-1}.
This is critical -- at alpha=+1 and alpha=-1 the adapter is an exact inverse of
itself, making bidirectional steering symmetric by construction.

At alpha=0: U_rot = U and S_scaled = S, so the layer is identical to frozen weights.
"""

import math

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int
from einops import rearrange
from flax import nnx
from loguru import logger


# -- Custom variable types for gradient filtering ----------------------------

class SVDParam(nnx.Param):
    """Trainable SVD adapter parameter (base class -- use subclasses for per-group LR)."""
    pass


class DeltaSParam(SVDParam):
    """Trainable delta_s scaling parameters (full LR)."""
    pass


class RotationParam(SVDParam):
    """Block-diagonal rotation parameters (lower LR via rotation_lr_scale config)."""
    pass


class SVDFrozen(nnx.Variable):
    """Frozen SVD component. Not differentiated."""
    pass


# -- SVD Steering Linear (replaces nnx.Linear) ------------------------------

class SVDSteeringLinear(nnx.Module):
    """SVD steering adapter replacing a linear layer.

    y = ((x @ U_rot) * S_scaled) @ Vh + x @ W_res

    where U_rot, S_scaled depend on alpha (the steering coefficient).
    Frozen: U, S, Vh, W_res.  Learnable: delta_s, rotation_params.
    """

    def __init__(
        self,
        U: jax.Array,               # (in_dim, r) - input singular vectors
        S: jax.Array,               # (r,) - singular values
        Vh: jax.Array,              # (r, out_dim) - output singular vectors
        W_res: jax.Array,           # (in_dim, out_dim) - residual
        rotation_block_size: int,
        max_rotation_angle: float,
        rotate_U: bool = True,
        rotate_V: bool = False,
        use_delta_s: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        r = S.shape[0]
        bs = min(rotation_block_size, r)
        assert r % bs == 0, f"r={r} must be divisible by block_size={bs}"

        # Frozen SVD components
        self.svd_U = SVDFrozen(U.astype(jnp.float32))
        self.svd_S = SVDFrozen(S.astype(jnp.float32))
        self.svd_Vh = SVDFrozen(Vh.astype(jnp.float32))
        self.svd_W_res = SVDFrozen(W_res.astype(jnp.bfloat16))

        # Trainable: delta_s with small positive bias for symmetry breaking.
        # The +4e-4 nudges the optimizer to scale up selected dims rather than just
        # rotating them. Rotation alone cannot break sign symmetry at init.
        key_s, key_r = jax.random.split(rngs.params())
        self.delta_s = DeltaSParam(
            jax.random.truncated_normal(key_s, -2.0, 2.0, (r,)) * 4e-4 + 4e-4
        )

        # Block-diagonal skew-symmetric rotation params (lower LR via RotationParam type)
        # Upper-triangle parameterization: store only bs*(bs-1)/2 elements per block,
        # like OFT/PSOFT. Avoids dead diagonal gradients and redundant (i,j)/(j,i) states.
        n_blocks = r // bs
        n_triu = bs * (bs - 1) // 2
        self.rotation_params = RotationParam(
            jax.random.truncated_normal(key_r, -2.0, 2.0, (n_blocks, n_triu)) * 1e-4
        )
        # Pre-compute upper-triangle indices for skew-symmetric reconstruction
        rows, cols = jnp.triu_indices(bs, k=1)
        self._triu_rows = rows
        self._triu_cols = cols

        # Steering coefficient (mutated during 3-pass forward)
        self.alpha = nnx.Variable(jnp.float32(1.0))
        self.max_angle = max_rotation_angle
        self.block_size = bs
        self.r = r
        self.rotate_U = rotate_U
        self.rotate_V = rotate_V
        self.use_delta_s = use_delta_s

    def __call__(self, x: Float[Array, "*batch in_features"]) -> Float[Array, "*batch out_features"]:
        alpha = self.alpha.value
        U = self.svd_U.value
        S = self.svd_S.value
        Vh = self.svd_Vh.value
        W_res = self.svd_W_res.value
        params = self.rotation_params.value  # (n_blocks, n_triu)
        bs = self.block_size
        n_blocks = params.shape[0]

        # Reconstruct skew-symmetric from upper-triangle params (like OFT/PSOFT).
        # 0.5 factor matches BOFT convention: cancels the 2x gradient from A - A^T.
        A = jnp.zeros((n_blocks, bs, bs), dtype=jnp.float32)
        A = A.at[:, self._triu_rows, self._triu_cols].set(params.astype(jnp.float32))
        A = 0.5 * (A - jnp.swapaxes(A, -1, -2))

        # Angle clamping (element-wise tanh, bounds bidirectional symmetry error)
        a_limit = 2 * math.tan(self.max_angle / 2)
        A = a_limit * jnp.tanh(A / a_limit)

        # Cayley transform in float32: R = (I - X)^{-1}(I + X)
        eye = jnp.eye(bs, dtype=jnp.float32)
        X = alpha * A / 2
        R_blocks = jnp.linalg.solve(
            eye[None] - X,
            eye[None] + X,
        )

        # Apply rotation to U (input singular vectors)
        if self.rotate_U:
            U_reshaped = U.reshape(U.shape[0], n_blocks, bs)
            U_rot = jnp.einsum('dnb,nbc->dnc', U_reshaped, R_blocks)
            U_rot = U_rot.reshape(U.shape)
        else:
            U_rot = U

        # Apply rotation to Vh (output singular vectors); off by default
        # (output rotation changes the upstream basis, making adaptation harder)
        if self.rotate_V:
            Vh_reshaped = Vh.reshape(n_blocks, bs, Vh.shape[1])
            Vh_rot = jnp.einsum('nbc,nbj->ncj', R_blocks, Vh_reshaped)
            Vh_rot = Vh_rot.reshape(Vh.shape)
        else:
            Vh_rot = Vh

        S_scaled = S + alpha * self.delta_s.value if self.use_delta_s else S

        dt = x.dtype
        out = (x @ U_rot.astype(dt)) * S_scaled.astype(dt)
        out = out @ Vh_rot.astype(dt)
        out = out + x @ W_res.astype(dt)
        return out


def create_svd_adapter(
    kernel: jax.Array,
    r: int,
    rotation_block_size: int,
    max_rotation_angle: float,
    rngs: nnx.Rngs,
    selected_indices: jax.Array | None = None,
    rotate_U: bool = True,
    rotate_V: bool = False,
    use_delta_s: bool = True,
) -> SVDSteeringLinear:
    """Create SVD adapter from a kernel matrix (in_dim, out_dim).

    If selected_indices is provided, uses those SVD dimensions.
    Otherwise uses top-r by singular value.
    """
    kernel_f32 = kernel.astype(jnp.float32)
    U_full, S_full, Vh_full = jnp.linalg.svd(kernel_f32, full_matrices=False)


    r_actual = min(r, S_full.shape[0])
    # Ensure divisible by block size
    bs = min(rotation_block_size, r_actual)
    r_actual = (r_actual // bs) * bs
    if r_actual == 0:
        r_actual = bs

    if selected_indices is not None:
        indices = selected_indices[:r_actual]
    else:
        indices = jnp.arange(r_actual)

    U = U_full[:, indices]
    S = S_full[indices]
    Vh = Vh_full[indices, :]
    W_res = kernel_f32 - U @ jnp.diag(S) @ Vh

    return SVDSteeringLinear(
        U, S, Vh, W_res,
        rotation_block_size=bs,
        max_rotation_angle=max_rotation_angle,
        rotate_U=rotate_U,
        rotate_V=rotate_V,
        use_delta_s=use_delta_s,
        rngs=rngs,
    )


# -- Dimension selection (data-aware) ----------------------------------------

def score_l1_trip(
    acts_projected: Float[Array, "n k"], S: Float[Array, " k"], r: int,
) -> Int[Array, " selected"]:
    """L1 trip scoring: union of top dims from 4 pools (cho, rej, diff_pos, diff_neg).

    Why not top-r by singular value? That picks globally "important" dimensions but
    ignores whether they are active in the contrastive data. This approach takes:
      r/3 cho-active, r/3 rej-active, r/6 diff_pos, r/6 diff_neg
    ensuring all signal types (absolute activation and contrastive difference) are
    represented in the selected subspace.
    """
    k = S.shape[0]
    assert r < k
    act_cho = acts_projected[::2]
    act_rej = acts_projected[1::2]

    l1_cho = jnp.abs(act_cho).mean(axis=0)
    l1_rej = jnp.abs(act_rej).mean(axis=0)
    diff = (act_cho - act_rej).mean(axis=0)

    scores_cho = S * l1_cho
    scores_rej = S * l1_rej
    scores_diff_pos = S * jax.nn.relu(diff)
    scores_diff_neg = S * jax.nn.relu(-diff)

    third = r // 3
    sixth = (r - 2 * third) // 2
    sixth_rem = r - 2 * third - 2 * sixth

    top_cho = jnp.argsort(-scores_cho)[:third]
    top_rej = jnp.argsort(-scores_rej)[:third]
    top_diff_pos = jnp.argsort(-scores_diff_pos)[:sixth + sixth_rem]
    top_diff_neg = jnp.argsort(-scores_diff_neg)[:sixth]

    combined = jnp.unique(jnp.concatenate([top_cho, top_rej, top_diff_pos, top_diff_neg]))

    if combined.shape[0] < r:
        scores_union = jnp.maximum(
            jnp.maximum(scores_cho, scores_rej),
            jnp.maximum(scores_diff_pos, scores_diff_neg),
        )
        # Mask out already-selected indices
        mask = jnp.zeros(k, dtype=jnp.bool_)
        mask = mask.at[combined].set(True)
        scores_union = jnp.where(mask, -jnp.inf, scores_union)
        extra = jnp.argsort(-scores_union)[:r - combined.shape[0]]
        combined = jnp.concatenate([combined, extra])

    return jnp.sort(combined[:r])



def polarity_interleave(acts_projected: jax.Array, indices: jax.Array) -> jax.Array:
    """Reorder indices so consecutive pairs alternate cho/rej-favoring dims.

    Block-diagonal rotation couples dims within each block of block_size.
    If all dims in a block favor the same direction (all cho-favoring), the block
    cannot learn bidirectional steering. Interleaving forces each block to have a
    mix of cho-favoring and rej-favoring dims, enabling bidirectional learning.
    """
    r = indices.shape[0]
    assert r % 2 == 0
    diff_signed = (acts_projected[::2, :][:, indices] - acts_projected[1::2, :][:, indices]).mean(axis=0)
    rank_order = jnp.argsort(-diff_signed)
    n_half = r // 2
    cho_ranked = rank_order[:n_half]
    rej_ranked = rank_order[n_half:][::-1]
    interleaved = jnp.stack([cho_ranked, rej_ranked], axis=1).reshape(-1)
    return indices[interleaved]


# -- Attention output adapter ------------------------------------------------

class SVDAttnOutAdapter(nnx.Module):
    """Drop-in for tunix Einsum('BTNH,NHD->BTD').

    Reshapes encoded [b,t,N,H] -> [b,t,N*H], applies SVDSteeringLinear, returns [b,t,D].
    Exposes .shape = (N, H, D) so tunix Attention.head_dim/.features still work.
    """
    def __init__(self, svd_linear: SVDSteeringLinear, num_heads: int, head_dim: int):
        self.svd_linear = svd_linear
        # Tuple attribute: read-only metadata for tunix property access
        self.shape = (num_heads, head_dim, svd_linear.svd_Vh.value.shape[1])

    def __call__(self, encoded: jax.Array) -> jax.Array:
        b, t, N, H = encoded.shape
        return self.svd_linear(encoded.reshape(b, t, N * H))


class SVDAttnQAdapter(nnx.Module):
    """Drop-in for tunix Einsum('BTD,NDH->BTNH') -- Q projection in GQA.

    Reshapes weight (N, D, H) -> (D, N*H) for SVD. On forward, applies SVDSteeringLinear
    then rearranges output back to [b, t, N, H].
    Exposes .shape = (N, D, H) so tunix Attention.num_heads reads shape[0].
    """
    def __init__(self, svd_linear: SVDSteeringLinear, num_heads: int, features: int, head_dim: int):
        self.svd_linear = svd_linear
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.shape = (num_heads, features, head_dim)

    def __call__(self, x: Float[Array, "b t D"]) -> Float[Array, "b t N H"]:
        b, t, _ = x.shape
        flat = rearrange(x, 'b t d -> (b t) d')
        out = self.svd_linear(flat)
        return rearrange(out, '(b t) (N H) -> b t N H', b=b, t=t, N=self.num_heads, H=self.head_dim)


class SVDAttnKVAdapter(nnx.Module):
    """Drop-in for tunix Einsum('BSD,CKDH->CBSKH') -- KV projection in GQA.

    K and V share one einsum with weight (C=2, K, D, H). Reshapes to (D, C*K*H) for SVD.
    On forward, applies SVDSteeringLinear then rearranges to [C, b, t, K, H].
    Output is tuple-unpacked: key_proj, value_proj = kv_einsum(x).
    Exposes .shape = (C, K, D, H) so tunix Attention.num_kv_heads reads shape[1].
    """
    def __init__(self, svd_linear: SVDSteeringLinear, C: int, num_kv_heads: int, features: int, head_dim: int):
        self.svd_linear = svd_linear
        self.C = C
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.shape = (C, num_kv_heads, features, head_dim)

    def __call__(self, x: Float[Array, "b t D"]) -> Float[Array, "C b t K H"]:
        b, t, _ = x.shape
        flat = rearrange(x, 'b t d -> (b t) d')
        out = self.svd_linear(flat)
        return rearrange(out, '(b t) (C K H) -> C b t K H', b=b, t=t, C=self.C, K=self.num_kv_heads, H=self.head_dim)


# -- Utilities ---------------------------------------------------------------

def set_alpha(model: nnx.Module, alpha: float):
    """Set steering coefficient for all SVD adapter layers."""
    for _, value in nnx.iter_graph(model):
        if isinstance(value, SVDSteeringLinear):
            value.alpha.value = jnp.float32(alpha)


def get_svd_modules(model: nnx.Module) -> list[SVDSteeringLinear]:
    """Get all SVD steering modules in the model."""
    modules = []
    for _, value in nnx.iter_graph(model):
        if isinstance(value, SVDSteeringLinear):
            modules.append(value)
    return modules


def monitor_svd_adapters(model: nnx.Module) -> dict:
    """Monitor ||delta_s||/||S|| ratio."""
    ratios = []
    for m in get_svd_modules(model):
        S = m.svd_S.value
        ds = m.delta_s.value
        ratios.append(float(jnp.linalg.norm(ds) / jnp.linalg.norm(S)))
    return {"adapter_ratio": max(ratios) if ratios else 0.0}
