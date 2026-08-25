#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.graphics_utils import focal2fov, getProjectionMatrix


width, height = 640, 512
fx, fy = 520.0, 515.0
cx, cy = 282.75, 266.0
matrix = getProjectionMatrix(
    1.0,
    100.0,
    focal2fov(fx, width),
    focal2fov(fy, height),
    cx=cx,
    cy=cy,
    width=width,
    height=height,
)

for x, y, z in ((0.0, 0.0, 10.0), (1.5, -0.75, 8.0), (-2.0, 1.0, 12.0)):
    point = torch.tensor([x, y, z, 1.0])
    clip = matrix @ point
    ndc = clip[:2] / clip[3]
    pixel_x = ((ndc[0] + 1.0) * width - 1.0) / 2.0
    pixel_y = ((ndc[1] + 1.0) * height - 1.0) / 2.0
    expected_x = fx * x / z + cx
    expected_y = fy * y / z + cy
    assert math.isclose(float(pixel_x), expected_x, abs_tol=1e-4)
    assert math.isclose(float(pixel_y), expected_y, abs_tol=1e-4)

centered = getProjectionMatrix(
    1.0,
    100.0,
    focal2fov(fx, width),
    focal2fov(fy, height),
)
assert centered[0, 2] == 0 and centered[1, 2] == 0
print("iMED principal-point projection tests passed")
