"""Source-only stereo geometry helpers for the iMED challenge.

The functions in this module never inspect Endoscope 1 imagery.  The target
camera intrinsics/pose remain handled by the existing renderer and evaluator.
"""

import json
import os

import numpy as np


CALIBRATION_FORMAT = "imed_endoscope2_stereo_v1"


def resolve_stereo_calibration(calibration_dir, sequence_dir, expected_intrinsics=None):
    """Resolve and validate a per-sequence Endoscope-2 L-to-R calibration."""
    if not calibration_dir:
        raise ValueError("imed_stereo_calibration_dir is required when stereo training is enabled")
    sequence = os.path.basename(os.path.normpath(sequence_dir))
    path = os.path.join(os.path.abspath(calibration_dir), f"{sequence}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing stereo calibration: {path}. Run scripts/estimate_imed_stereo_extrinsics.py first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("format") != CALIBRATION_FORMAT:
        raise ValueError(f"Unsupported stereo calibration format in {path}")
    if payload.get("sequence") != sequence:
        raise ValueError(f"Calibration sequence mismatch in {path}")
    provenance = payload.get("provenance", {})
    if provenance.get("source_data_only") is not True or provenance.get("used_endoscope1") is not False:
        raise ValueError(f"Calibration does not certify source-only estimation: {path}")
    if payload.get("validation", {}).get("passed") is not True:
        raise ValueError(f"Stereo calibration did not pass validation: {path}")
    if expected_intrinsics is not None:
        stored_intrinsics = payload.get("intrinsics", {})
        for key, expected in expected_intrinsics.items():
            stored = np.asarray(stored_intrinsics.get(key), dtype=np.float64)
            # K.txt is parsed as float32 by the training loader while the
            # calibration JSON preserves float64 values. Allow only the
            # expected float32 round-off (about 1e-4 at focal length ~1000),
            # while still rejecting a calibration from another sequence.
            if stored.shape != (3, 3) or not np.allclose(stored, expected, rtol=1e-6, atol=1e-4):
                raise ValueError(f"Calibration intrinsics {key} do not match the sequence K.txt: {path}")

    transform = np.asarray(payload.get("transform_R_from_L"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid transform_R_from_L in {path}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"Invalid homogeneous transform in {path}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or np.linalg.det(rotation) < 0.99:
        raise ValueError(f"Non-rigid stereo transform in {path}")
    return transform.astype(np.float32), path


def project_left_depth_to_right(
    depth_l,
    valid_l,
    K_l,
    K_r,
    transform_r_from_l,
    return_source_indices=False,
):
    """Forward-project left depth into the synchronized right camera.

    A nearest-pixel z-buffer is deliberately used: only geometrically observed
    right pixels receive pseudo depth/supervision. Holes are kept invalid rather
    than inpainted, preventing invented target information from entering train.
    """
    depth_l = np.asarray(depth_l, dtype=np.float32)
    valid_l = np.asarray(valid_l, dtype=np.bool_)
    if depth_l.ndim != 2 or valid_l.shape != depth_l.shape:
        raise ValueError("depth_l and valid_l must be aligned HxW arrays")
    height, width = depth_l.shape
    valid = valid_l & np.isfinite(depth_l) & (depth_l > 0)
    flat = np.flatnonzero(valid.reshape(-1))
    depth_r = np.zeros((height, width), dtype=np.float32)
    source_indices_r = np.full((height, width), -1, dtype=np.int64)
    if flat.size == 0:
        result = (depth_r, depth_r.astype(np.bool_))
        return result + (source_indices_r,) if return_source_indices else result

    v_l, u_l = np.divmod(flat, width)
    z_l = depth_l.reshape(-1)[flat].astype(np.float64)
    K_l = np.asarray(K_l, dtype=np.float64)
    K_r = np.asarray(K_r, dtype=np.float64)
    transform = np.asarray(transform_r_from_l, dtype=np.float64)

    x_l = (u_l - K_l[0, 2]) * z_l / K_l[0, 0]
    y_l = (v_l - K_l[1, 2]) * z_l / K_l[1, 1]
    points_l = np.stack((x_l, y_l, z_l, np.ones_like(z_l)), axis=0)
    points_r = (transform @ points_l)[:3].T
    z_r = points_r[:, 2]
    in_front = np.isfinite(points_r).all(axis=1) & (z_r > 1e-6)

    u_r = np.full(z_r.shape, -1, dtype=np.int64)
    v_r = np.full(z_r.shape, -1, dtype=np.int64)
    u_r[in_front] = np.rint(K_r[0, 0] * points_r[in_front, 0] / z_r[in_front] + K_r[0, 2]).astype(np.int64)
    v_r[in_front] = np.rint(K_r[1, 1] * points_r[in_front, 1] / z_r[in_front] + K_r[1, 2]).astype(np.int64)
    inside = in_front & (u_r >= 0) & (u_r < width) & (v_r >= 0) & (v_r < height)
    if not inside.any():
        result = (depth_r, depth_r.astype(np.bool_))
        return result + (source_indices_r,) if return_source_indices else result

    target_flat = v_r[inside] * width + u_r[inside]
    z_values = z_r[inside].astype(np.float32)
    source_flat = flat[inside]
    # Sort by target pixel and then depth. The first entry for each target is
    # the z-buffer winner, and its source index lets us project synchronized
    # left RGB without inventing values in forward-warp holes.
    order = np.lexsort((z_values, target_flat))
    sorted_targets = target_flat[order]
    first = np.r_[True, sorted_targets[1:] != sorted_targets[:-1]]
    winners = order[first]
    selected_targets = target_flat[winners]
    depth_r.reshape(-1)[selected_targets] = z_values[winners]
    source_indices_r.reshape(-1)[selected_targets] = source_flat[winners]
    visible = source_indices_r >= 0
    result = (depth_r, visible)
    return result + (source_indices_r,) if return_source_indices else result


def build_stereo_photometric_weight(
    left_rgb,
    right_rgb,
    source_indices_r,
    valid_r,
    sigma=0.10,
):
    """Build a detached right-view confidence map from synchronized source RGB.

    A robust per-channel affine alignment removes global exposure/white-balance
    differences. Residual disagreement then receives a smooth exponential
    weight. Invalid/occluded forward-warp pixels remain exactly zero.
    """
    left_rgb = np.asarray(left_rgb, dtype=np.float32)
    right_rgb = np.asarray(right_rgb, dtype=np.float32)
    source_indices_r = np.asarray(source_indices_r, dtype=np.int64)
    valid_r = np.asarray(valid_r, dtype=np.bool_)
    if left_rgb.shape != right_rgb.shape or left_rgb.ndim != 3 or left_rgb.shape[2] != 3:
        raise ValueError("left_rgb and right_rgb must be aligned HxWx3 arrays")
    if source_indices_r.shape != left_rgb.shape[:2] or valid_r.shape != left_rgb.shape[:2]:
        raise ValueError("source_indices_r and valid_r must match the image shape")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")

    valid = valid_r & (source_indices_r >= 0)
    weight = np.zeros(valid.shape, dtype=np.float32)
    target_flat = np.flatnonzero(valid.reshape(-1))
    if target_flat.size == 0:
        return weight

    source_flat = source_indices_r.reshape(-1)[target_flat]
    projected = left_rgb.reshape(-1, 3)[source_flat].astype(np.float64)
    target = right_rgb.reshape(-1, 3)[target_flat].astype(np.float64)

    source_center = np.median(projected, axis=0)
    target_center = np.median(target, axis=0)
    source_scale = 1.4826 * np.median(np.abs(projected - source_center), axis=0)
    target_scale = 1.4826 * np.median(np.abs(target - target_center), axis=0)
    scale_ratio = np.clip(target_scale / np.maximum(source_scale, 1e-3), 0.5, 2.0)
    aligned = np.clip((projected - source_center) * scale_ratio + target_center, 0.0, 1.0)

    residual = np.mean(np.abs(aligned - target), axis=1)
    confidence = np.exp(-residual / float(sigma))
    weight.reshape(-1)[target_flat] = confidence.astype(np.float32)
    return weight
