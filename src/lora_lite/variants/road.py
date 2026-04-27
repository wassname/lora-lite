"""ROAD: Rotation ADaptation. https://arxiv.org/abs/2409.00119

ROAD applies a learned output-space block rotation/scaling after the frozen base
layer:

    y' = R y = R (W x + b)

This matches PEFT's unmerged forward path and fits lora-lite as a simple output
hook. We implement the three PEFT variants (`road_1`, `road_2`, `road_4`) and
skip merge/unmerge because this library keeps adapters as hooks.

Refs:
  - peft: https://github.com/huggingface/peft/blob/6030f9160ed2fc17220f6f41382a66f1257b6a93/src/peft/tuners/road/layer.py
"""
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import nn, Tensor as T

from ..config import AdapterConfig, register_config
from ..variant import ParamSpec, register

RoadVariant = Literal["road_1", "road_2", "road_4"]


@register_config
@dataclass
class RoadConfig(AdapterConfig):
    variant: str = "road"
    road_variant: RoadVariant = "road_1"
    group_size: int = 64


def _road_param_size(d_out: int, road_variant: str) -> int:
    if road_variant == "road_1":
        return d_out // 2
    if road_variant == "road_2":
        return d_out
    if road_variant == "road_4":
        return d_out * 2
    raise ValueError(f"road_variant must be 'road_1', 'road_2', or 'road_4', got {road_variant!r}")


def _validate_group_geometry(d_out: int, group_size: int) -> None:
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(f"ROAD group_size must be positive and even, got {group_size}")
    if d_out % group_size != 0:
        raise ValueError(f"ROAD d_out={d_out} must be divisible by group_size={group_size}")


def _prepare_cols(
    road_variant: str,
    group_size: int,
    road_theta: torch.Tensor,
    road_alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if road_variant == "road_1":
        # One θ/α per pair. Reuse it for both rows of each 2D rotation block.
        road_theta = road_theta.reshape(-1, group_size // 2).repeat_interleave(2, dim=0).flatten()
        road_alpha = road_alpha.reshape(-1, group_size // 2).repeat_interleave(2, dim=0).flatten()
        first_col = road_alpha * road_theta.cos()
        second_col = road_alpha * road_theta.sin()
    elif road_variant == "road_2":
        # One θ/α per output coordinate.
        first_col = road_alpha * road_theta.cos()
        second_col = road_alpha * road_theta.sin()
    elif road_variant == "road_4":
        # Independent θ/α for the first and second column contributions.
        road_theta = road_theta.reshape(-1, 2, group_size)
        road_alpha = road_alpha.reshape(-1, 2, group_size)
        first_col = road_alpha[:, 0, :].flatten() * road_theta[:, 0, :].cos().flatten()
        second_col = road_alpha[:, 1, :].flatten() * road_theta[:, 1, :].sin().flatten()
    else:
        raise ValueError(f"road_variant must be 'road_1', 'road_2', or 'road_4', got {road_variant!r}")
    return first_col, second_col


def _apply_road(
    road_variant: str,
    group_size: int,
    road_theta: torch.Tensor,
    road_alpha: torch.Tensor,
    y: Float[T, '*B o'],
) -> Float[T, '*B o']:
    first_col, second_col = _prepare_cols(road_variant, group_size, road_theta, road_alpha)
    y_grouped = y.reshape(-1, 2, group_size // 2)
    y1 = y_grouped[:, 0, :]
    y2 = y_grouped[:, 1, :]
    rotate_half_y = torch.stack((-y2, y1), dim=1).reshape(y.shape)
    return y * first_col + rotate_half_y * second_col


def _road_matrix(
    road_variant: str,
    group_size: int,
    road_theta: torch.Tensor,
    road_alpha: torch.Tensor,
) -> torch.Tensor:
    """Explicit PEFT merge matrix. Used for tests and small-debug inspection."""
    first_col, second_col = _prepare_cols(road_variant, group_size, road_theta, road_alpha)
    size = second_col.shape[0]
    output = torch.diag(first_col)
    swapped_second_col = second_col.reshape(-1, 2, group_size // 2)[:, [1, 0], :].flatten()
    rotated_diag_second_col = torch.diag(swapped_second_col).reshape(-1, 2, group_size // 2, size)[:, [1, 0], :, :]
    rotated_diag_second_col[:, 0, :, :] *= -1
    return output + rotated_diag_second_col.reshape(size, size)


@register
class ROAD:
    name = "road"

    @staticmethod
    def param_specs(d_in: int, d_out: int, cfg: RoadConfig) -> dict[str, ParamSpec]:
        _validate_group_geometry(d_out, cfg.group_size)
        size = _road_param_size(d_out, cfg.road_variant)
        return dict(
            lora_road_theta=ParamSpec((size,), init="near_zero"),
            lora_road_alpha=ParamSpec((size,), init="near_one"),
        )

    @staticmethod
    def init(layer: nn.Module, cfg: RoadConfig) -> None:
        return

    @staticmethod
    def forward(
        layer: nn.Module,
        x: Float[T, '*B i'],
        y: Float[T, '*B o'],
    ) -> Float[T, '*B o']:
        del x
        cfg = layer._lora_cfg
        y_cast = y.to(layer.lora_road_theta.dtype)
        return _apply_road(cfg.road_variant, cfg.group_size, layer.lora_road_theta, layer.lora_road_alpha, y_cast)
