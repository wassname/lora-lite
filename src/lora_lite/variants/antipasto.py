"""AntiPaSTO: SVD steering with learnable singular-value deltas + block-diagonal Cayley rotation.

wassname 2026  https://arxiv.org/abs/2601.07473

    W = U diag(S) Vh + W_res        (top-r SVD; W_res = W - U_r S_r Vh_r)
    learn: delta_s (r,), rot_T (n_blocks, bs(bs-1)/2)
    R = block_diag(Cayley(skew(rot_T)));  Vh_eff = R @ Vh (or U_eff = U @ R.T)
    y = x @ W_res.T + ((x @ Vh_eff.T) * (S + delta_s)) @ U_eff.T

Identity at t=0: rot_T=0 -> R=I, delta_s~4e-4 -> y ≈ x @ W^T (fp32 SVD round-trip, tiny positive bias on delta_s breaks sign symmetry).

Scope cut vs antipasto3: this is a fine-tuning adapter, not the full runtime
steering interface. There is no per-call alpha, so it does not expose the
bidirectional R(+alpha) / R(-alpha) inference symmetry. The V-basis path uses the
opposite chirality to antipasto3's default U-basis path, so checkpoints are not
portable without a sign/basis convention.

Refs:
  - paper: https://github.com/wassname/AntiPaSTO
  - lite port of: https://github.com/wassname/antipasto3
    (offline: docs/refs/antipasto3_svd_adapter.py)
"""
import math
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import nn, Tensor as T

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config

CalibrationBatch = dict | tuple | list | T
CalibrationData = Iterable[CalibrationBatch]


@register_config
@dataclass
class AntiPaSTOConfig(AdapterConfig):
    variant: str = "antipasto"
    # Higher default than LoRA (r=8) since trainable params scale as r + r/bs*bs*(bs-1)/2, not r*(d_in+d_out).
    r: int = 256
    # Block size for the block-diagonal Cayley rotation. r must be divisible by it.
    block_size: int = 4
    # Cayley map saturation: bounds rotation angle to ~max_rotation_angle radians.
    max_rotation_angle: float = 0.5
    # Which singular basis to rotate: 'V' (input), 'U' (output), or 'none'.
    rotate_basis: Literal["V", "U", "none"] = "V"


def _cayley(skew: torch.Tensor) -> torch.Tensor:
    """R = (I - X)^-1 (I + X) for X = skew/2; preserves orthogonality."""
    bs = skew.shape[-1]
    eye = torch.eye(bs, dtype=skew.dtype, device=skew.device).expand_as(skew)
    X = skew / 2
    return torch.linalg.solve(eye - X, eye + X)


def _build_rotation(rot_T: torch.Tensor, bs: int, max_angle: float) -> torch.Tensor:
    """rot_T: (n_blocks, bs*(bs-1)/2) -> R: (n_blocks, bs, bs) Cayley rotation."""
    n_blocks, _ = rot_T.shape
    rows, cols = torch.triu_indices(bs, bs, offset=1, device=rot_T.device).unbind(0)
    A = torch.zeros(n_blocks, bs, bs, dtype=rot_T.dtype, device=rot_T.device)
    A[:, rows, cols] = rot_T
    A = 0.5 * (A - A.transpose(-1, -2))
    a_limit = 2.0 * math.tan(max_angle / 2.0)
    A = a_limit * torch.tanh(A / a_limit)
    return _cayley(A)


@register
class AntiPaSTO:
    name = "antipasto"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        r = cfg.r
        bs = int(cfg.block_size)
        if r % bs != 0:
            raise ValueError(f"AntiPaSTO requires r={r} divisible by block_size={bs}")
        specs = dict(
            # Frozen SVD components captured at init.
            lora_U=ParamSpec((d_out, r), init="zeros", trainable=False, as_buffer=True),
            lora_S=ParamSpec((r,), init="zeros", trainable=False, as_buffer=True),
            lora_Vh=ParamSpec((r, d_in), init="zeros", trainable=False, as_buffer=True),
            # Trainable: per-singular-value delta.
            # antipasto3 uses 4e-4 + N(0, 4e-4): small positive bias breaks sign
            # symmetry (rotation alone can't); zero-init works but trains slower.
            lora_delta_s=ParamSpec((r,), init=lambda t: t.normal_(0, 4e-4).add_(4e-4)),
        )
        if cfg.rotate_basis != "none":
            n_blocks = r // bs
            n_triu = bs * (bs - 1) // 2
            specs["lora_rot_T"] = ParamSpec((n_blocks, n_triu), init="zeros")
        return specs

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        if type(layer) is not nn.Linear:
            raise TypeError(
                "AntiPaSTO mutates layer.weight into W_res (like PiSSA), so v1 "
                "only supports plain nn.Linear, not bnb 4/8-bit."
            )
        with torch.no_grad():
            W = layer.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = cfg.r
            Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
            layer.lora_U.copy_(Ur.to(layer.lora_U.dtype))
            layer.lora_S.copy_(Sr.to(layer.lora_S.dtype))
            layer.lora_Vh.copy_(Vhr.to(layer.lora_Vh.dtype))
            W_res = (W - (Ur * Sr) @ Vhr).to(layer.weight.dtype)
            layer.weight.data.copy_(W_res)
            # group_init() refines this to input-aligned directions if calibration_data is given.

    @staticmethod
    def group_init(model: nn.Module, targets, cfg, calibration_data: CalibrationData | None) -> None:
        """Wanda-style data-driven dimension selection within the weight SVD.

        init() picks the top-r singular dimensions by S alone (PiSSA-style).
        group_init() re-selects based on S[i] * mean|X @ Vh[i]|: dimensions
        that are both large in W AND active given real inputs.

        If calibration_data is None the weight-SVD init from init() is kept.
        """
        if calibration_data is None:
            return

        layers = {name: layer for name, layer, _ in targets}
        captured: dict[str, list[T]] = {n: [] for n in layers}

        def make_hook(name):
            def _h(module, args, kwargs):
                x = args[0].detach()
                captured[name].append(rearrange(x, "... d -> (...) d").to(torch.float32).cpu())
            return _h

        handles = [
            layers[n].register_forward_pre_hook(make_hook(n), with_kwargs=True)
            for n in layers
        ]
        try:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                for batch in calibration_data:
                    if isinstance(batch, dict):
                        model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        model(*batch)
                    else:
                        model(batch)
            if was_training:
                model.train()
        finally:
            for h in handles:
                h.remove()

        r = cfg.r
        for name, layer in layers.items():
            X = torch.cat(captured[name], dim=0)          # (N, d_in)
            if X.shape[0] < r:
                raise RuntimeError(
                    f"AntiPaSTO at {name}: only {X.shape[0]} calibration tokens, need >= r={r}"
                )

            # Recover W_orig: init() wrote W_res into layer.weight and stored top-r components
            W_res = layer.weight.data.float()
            U_old = layer.lora_U.float()                  # (d_out, r)
            S_old = layer.lora_S.float()                  # (r,)
            Vh_old = layer.lora_Vh.float()                # (r, d_in)
            W_orig = W_res + (U_old * S_old.unsqueeze(0)) @ Vh_old

            # Full SVD to score all dimensions
            U_full, S_full, Vh_full = torch.linalg.svd(W_orig, full_matrices=False)
            # score[i] = S[i] * mean|X @ Vh[i]|  (Wanda: weight magnitude × activation magnitude)
            act_mag = (X @ Vh_full.T).abs().mean(dim=0)  # (k,)
            scores = S_full * act_mag
            idx = scores.argsort(descending=True)[:r]    # top-r by joint importance
            idx = idx.sort().values                       # stable ordering

            Ur, Sr, Vhr = U_full[:, idx], S_full[idx], Vh_full[idx]
            W_res_new = (W_orig - (Ur * Sr.unsqueeze(0)) @ Vhr).to(layer.weight.dtype)

            with torch.no_grad():
                layer.lora_U.copy_(Ur.to(layer.lora_U))
                layer.lora_S.copy_(Sr.to(layer.lora_S))
                layer.lora_Vh.copy_(Vhr.to(layer.lora_Vh))
                layer.weight.data.copy_(W_res_new)

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        bs = int(cfg.block_size)
        max_angle = float(cfg.max_rotation_angle)
        rotate_basis = cfg.rotate_basis

        U = layer.lora_U.to(x.dtype)                          # (d_out, r)
        S = layer.lora_S.to(x.dtype)                          # (r,)
        Vh = layer.lora_Vh.to(x.dtype)                        # (r, d_in)

        if rotate_basis == "none":
            U_eff, Vh_eff = U, Vh
        else:
            R_blocks = _build_rotation(layer.lora_rot_T.float(), bs, max_angle).to(x.dtype)
            n_blocks = R_blocks.shape[0]                      # R_blocks: (n, bs, bs)
            if rotate_basis == "V":
                Vh_blocks = rearrange(Vh, "(n a) i -> n a i", n=n_blocks)
                Vh_rot = einsum(R_blocks, Vh_blocks, "n a b, n b i -> n a i")
                Vh_eff = rearrange(Vh_rot, "n a i -> (n a) i")
                U_eff = U
            elif rotate_basis == "U":
                U_blocks = rearrange(U, "d (n b) -> d n b", n=n_blocks)
                U_rot = einsum(U_blocks, R_blocks, "d n b, n c b -> d n c")
                U_eff = rearrange(U_rot, "d n c -> d (n c)")
                Vh_eff = Vh
            else:
                raise ValueError(f"rotate_basis must be 'U', 'V', or 'none', got {rotate_basis!r}")

        S_eff = S + layer.lora_delta_s.to(x.dtype)            # (r,)
        h = einsum(x, Vh_eff, "... i, r i -> ... r")          # x @ Vh_eff.T
        h = h * S_eff                                         # diag(S_eff)
        delta = einsum(h, U_eff, "... r, o r -> ... o")       # @ U_eff.T
        return y + delta
