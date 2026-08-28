#!/usr/bin/env python3

import sys
import json
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.imed_stereo import (
    CALIBRATION_FORMAT,
    build_stereo_photometric_weight,
    project_left_depth_to_right,
    resolve_stereo_calibration,
    use_stereo_auxiliary_for_stage,
)


depth = np.full((3, 4), 2.0, dtype=np.float32)
valid = np.ones_like(depth, dtype=np.bool_)
K = np.asarray([[2.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
identity = np.eye(4)
assert use_stereo_auxiliary_for_stage("coarse", fine_only=True) is False
assert use_stereo_auxiliary_for_stage("fine", fine_only=True) is True
assert use_stereo_auxiliary_for_stage("coarse", fine_only=False) is True
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

# Identity geometry and exposure-aligned images should retain high confidence,
# while a locally corrupted right observation must be downweighted.
left_rgb = np.linspace(0.1, 0.9, 36, dtype=np.float32).reshape(3, 4, 3)
_, projected_valid, source_indices = project_left_depth_to_right(
    depth, valid, K, K, identity, return_source_indices=True
)
right_rgb = np.clip(left_rgb * 1.2 + 0.05, 0.0, 1.0)
clean_weight = build_stereo_photometric_weight(
    left_rgb, right_rgb, source_indices, projected_valid, sigma=0.10
)
assert clean_weight[projected_valid].mean() > 0.95
corrupt_right = right_rgb.copy()
corrupt_right[1, 2] = 0.0
corrupt_weight = build_stereo_photometric_weight(
    left_rgb, corrupt_right, source_indices, projected_valid, sigma=0.10
)
assert corrupt_weight[1, 2] < clean_weight[1, 2]
assert np.all(corrupt_weight[~projected_valid] == 0)

# Calibration files retain float64 intrinsics, whereas the loader uses
# float32. This round-off must pass, but a genuinely different K must fail.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    sequence = root / "sequence"
    sequence.mkdir()
    calibration_dir = root / "calibration"
    calibration_dir.mkdir()
    stored_K = np.asarray([[1039.5010572364, 0.0, 565.5091746366], [0.0, 1040.0475664751, 532.1105243640], [0.0, 0.0, 1.0]])
    payload = {
        "format": CALIBRATION_FORMAT,
        "sequence": "sequence",
        "transform_R_from_L": np.eye(4).tolist(),
        "intrinsics": {"K2_L": stored_K.tolist()},
        "provenance": {"source_data_only": True, "used_endoscope1": False},
        "validation": {"passed": True},
    }
    path = calibration_dir / "sequence.json"
    path.write_text(json.dumps(payload))
    resolve_stereo_calibration(calibration_dir, sequence, {"K2_L": stored_K.astype(np.float32)})
    wrong_K = stored_K.astype(np.float32)
    wrong_K[0, 0] += 0.02
    try:
        resolve_stereo_calibration(calibration_dir, sequence, {"K2_L": wrong_K})
        raise AssertionError("Mismatched intrinsics were accepted")
    except ValueError as error:
        assert "do not match" in str(error)
print("iMED stereo geometry tests passed")
