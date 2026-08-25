#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.imed_stereo import project_left_depth_to_right


depth = np.full((3, 4), 2.0, dtype=np.float32)
valid = np.ones_like(depth, dtype=np.bool_)
K = np.asarray([[2.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
identity = np.eye(4)
projected, mask = project_left_depth_to_right(depth, valid, K, K, identity)
np.testing.assert_allclose(projected, depth)
assert mask.all()

# Two left points colliding in the right image must keep the nearer surface.
K_collapse = K.copy()
K_collapse[0, 0] = 0.01
depth_varying = depth.copy()
depth_varying[1, :] = [4.0, 3.0, 2.0, 1.0]
projected, mask = project_left_depth_to_right(depth_varying, valid, K, K_collapse, identity)
assert projected[1][mask[1]].min() == 1.0
print("iMED stereo geometry tests passed")
