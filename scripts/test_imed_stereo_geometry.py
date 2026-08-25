#!/usr/bin/env python3

import sys
import json
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.imed_stereo import (
    CALIBRATION_FORMAT,
    project_left_depth_to_right,
    resolve_stereo_calibration,
)


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
