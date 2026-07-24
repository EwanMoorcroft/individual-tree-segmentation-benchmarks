"""Pinned checkpoint-layout adaptations used by ForestFormer3D."""

from __future__ import annotations

from typing import TypeVar


TensorT = TypeVar("TensorT")


def checkpoint_tensor_for_runtime(name: str, tensor: TensorT) -> TensorT:
    """Map the archived RSKC sparse weights to spconv-v2's model layout."""
    if (
        (name.startswith("unet") or name.startswith("input_conv"))
        and name.endswith("weight")
        and tensor.ndim == 5
    ):
        return tensor.permute(4, 0, 1, 2, 3)
    return tensor
