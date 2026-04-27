"""EVA: Explained-Variance Adaptation. Paischer et al. 2024  https://arxiv.org/abs/2410.07170

LoRA forward `y + scale*(B@A@x)`; init A = top-r right singular vectors of the
layer-input distribution on a small calibration set (instead of kaiming).

Identity at t=0: B=0.

Stripped down: uniform per-layer rank, single full SVD on pooled inputs, no QKV
input dedup. (peft does rank redistribution + IncrementalPCA + hash dedup.)

Refs:
  - peft: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/eva.py
    (offline: docs/refs/peft_eva.py; example: docs/refs/peft_eva_finetuning.py)
"""
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import nn, Tensor as T
from typing import Iterable
from dataclasses import dataclass

from ..variant import register, ParamSpec
from ..config import AdapterConfig, register_config

CalibrationBatch = dict | tuple | list | T
CalibrationData = Iterable[CalibrationBatch]


@register_config
@dataclass
class EVAConfig(AdapterConfig):
    variant: str = "eva"


@register
class EVA:
    name = "eva"

    @staticmethod
    def param_specs(d_in, d_out, cfg):
        return dict(
            # A trainable per peft: EVA only changes the init.
            lora_A=ParamSpec((cfg.r, d_in), init="zeros"),
            lora_B=ParamSpec((d_out, cfg.r), init="zeros"),
        )

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
