#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.imed_overlap import build_imed_soft_overlap_weight


tool_mask = torch.tensor(
    [
        [[1, 1], [1, 0]],
        [[1, 1], [1, 0]],
    ],
    dtype=torch.bool,
)
source_overlap = torch.tensor(
    [
        [[1, 1], [0, 0]],  # raw visibility = 0.50: soft weighting enabled
        [[1, 1], [1, 1]],  # raw visibility = 1.00: exact baseline fallback
    ],
    dtype=torch.bool,
)

result = build_imed_soft_overlap_weight(tool_mask, source_overlap, 0.90)
spatial_dims = (1, 2)

assert result.active.flatten().tolist() == [True, False]
assert torch.equal(result.weight[1], tool_mask[1].float())
assert torch.allclose(
    result.weight.sum(dim=spatial_dims),
    tool_mask.float().sum(dim=spatial_dims),
)
assert not torch.equal(result.weight[0], tool_mask[0].float())
assert torch.count_nonzero(result.weight[~tool_mask]) == 0

print("iMED soft-overlap gate tests passed")
