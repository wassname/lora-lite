"""EVA: Explained-Variance Adaptation. Paischer et al. 2024.

Paper: https://arxiv.org/abs/2410.07170 (also referred to as ICLR'25 EVA).

Idea: instead of random A and zero B (LoRA) or SVD of W (PiSSA), initialize
`lora_A` to the top-r right singular vectors of the LAYER INPUT distribution
on a small calibration set. Forward = `y + scale * (B @ A @ x)` exactly like
LoRA; with `lora_B = 0` the adapter is identity at t=0. Only B trains
afterwards (A frozen). The result: each rank slot points along a direction
that actually carries information at this layer.

This is a stripped-down EVA; we do NOT implement:
  - rank redistribution across layers via explained-variance ratios
    (peft EVA computes an explained_variance_ratio per layer then redistributes
    the global rank budget; we use a uniform `cfg.r` per layer).
  - Incremental PCA over many micro-batches (we run one full SVD on the
    pooled calibration activations per layer).
  - Equal-input deduplication (peft hashes inputs to share SVD across QKV).

API stress-test: this variant requires data-driven init, so it implements
`group_init(model, targets, cfg, calibration_data)` to drive a single forward
pass on `calibration_data` with hooks that capture each target's input.

Identity at t=0: `lora_B = 0` -> delta = 0 -> y unchanged.

References:
  - peft EVA (full impl, with IncrementalPCA + redistribution):
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/eva.py
    (offline: docs/refs/peft_eva.py)
  - peft fine-tuning script demonstrating initialize_lora_eva_weights:
    https://github.com/huggingface/peft/blob/main/examples/eva_finetuning/eva_finetuning.py
    (offline: docs/refs/peft_eva_finetuning.py)
"""
import torch
from einops import einsum
from jaxtyping import Float
from torch import nn, Tensor as T
from typing import Iterable

from ..variant import register, ParamSpec

CalibrationBatch = dict | tuple | list | T
CalibrationData = Iterable[CalibrationBatch]


@register
class EVA:
    name = "eva"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        return {
            # A is trainable Parameter (peft semantics): EVA only changes the INIT.
            # peft copies SVD vectors into the LoRA A weight, which remains a regular
            # nn.Linear.weight Parameter (docs/refs/peft_eva.py:529).
            # On step 0 only B has nonzero grad (delta=0 since B=0); A starts moving
            # once B becomes nonzero, same gradient pattern as DeLoRA.
            "lora_A": ParamSpec((cfg.r, d_in), init="zeros", trainable=True),
            # B is zero-init -> identity at t=0.
            "lora_B": ParamSpec((d_out, cfg.r), init="zeros", trainable=True),
        }

    @staticmethod
    def init(layer: nn.Module, cfg) -> None:
        # No-op; group_init does the data-driven SVD across all targets at once.
        return

    @staticmethod
    def group_init(model: nn.Module, targets, cfg, calibration_data: CalibrationData | None) -> None:
        # adapter.load() passes _skip_group_init=True so this is only called on
        # the live attach path where calibration_data is required.
        if calibration_data is None:
            raise ValueError(
                "EVA requires calibration_data: an iterable of model inputs "
                "(dicts of kwargs to model.forward, tuples of positional args, "
                "or single tensors) used to estimate the per-layer input PCA. "
                "Pass via lora_lite.attach(model, cfg, calibration_data=batches)."
            )
        # Collect input activations per target via forward hooks.
        layers = {name: layer for name, layer, _ in targets}
        captured: dict[str, list[T]] = {n: [] for n in layers}

        def make_hook(name):
            def _h(module, args, kwargs):
                # signature: pre-forward, args[0] is the input tensor
                x = args[0].detach()
                captured[name].append(x.reshape(-1, x.shape[-1]).to(torch.float32).cpu())
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

        # SVD per target on pooled inputs; top-r right singular vectors -> A.
        for name, layer in layers.items():
            X = torch.cat(captured[name], dim=0)              # (N, d_in)
            if X.shape[0] < cfg.r:
                raise RuntimeError(
                    f"EVA at {name}: only {X.shape[0]} calibration tokens, need >= r={cfg.r}"
                )
            # full_matrices=False -> Vh shape (min(N,d_in), d_in); take top-r rows
            _, _, Vh = torch.linalg.svd(X, full_matrices=False)
            A = Vh[: cfg.r, :].to(layer.lora_A.dtype).to(layer.lora_A.device)
            with torch.no_grad():
                layer.lora_A.copy_(A)

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        cfg = layer._lora_cfg
        scale = cfg.alpha / cfg.r
        h = einsum(x, layer.lora_A, "... i, r i -> ... r")
        delta = einsum(h, layer.lora_B, "... r, o r -> ... o")
        return y + scale * delta
