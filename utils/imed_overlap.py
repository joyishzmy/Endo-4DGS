from typing import NamedTuple

import torch


class IMEDSoftOverlapWeight(NamedTuple):
    weight: torch.Tensor
    raw_visibility: torch.Tensor
    active: torch.Tensor


def build_imed_soft_overlap_weight(
    tool_mask,
    source_overlap_mask,
    visibility_threshold=1.0,
):
    """Build normalized RGB weights with an optional geometry-only gate.

    Frames whose raw source-view visibility is at least ``visibility_threshold``
    fall back exactly to the original tool mask. The visibility statistic uses
    only the projected source-depth mask and is independent of target RGB data.
    """
    if tool_mask.shape != source_overlap_mask.shape:
        raise ValueError(
            "tool_mask and source_overlap_mask must have identical shapes: "
            f"{tool_mask.shape} != {source_overlap_mask.shape}"
        )
    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError("visibility_threshold must be in [0, 1]")

    valid = tool_mask.float()
    source_overlap = source_overlap_mask.float()
    spatial_dims = tuple(range(1, valid.ndim))

    raw_visibility = source_overlap.mean(dim=spatial_dims, keepdim=True)
    active = raw_visibility < visibility_threshold

    overlap = source_overlap * valid
    valid_count = valid.sum(dim=spatial_dims, keepdim=True)
    overlap_ratio = overlap.sum(dim=spatial_dims, keepdim=True) / valid_count.clamp_min(1.0)
    raw_weight = valid * (1.0 + (1.0 - overlap_ratio) * overlap)
    raw_count = raw_weight.sum(dim=spatial_dims, keepdim=True)
    soft_weight = raw_weight * valid_count / raw_count.clamp_min(1.0)

    return IMEDSoftOverlapWeight(
        weight=torch.where(active, soft_weight, valid),
        raw_visibility=raw_visibility,
        active=active,
    )
