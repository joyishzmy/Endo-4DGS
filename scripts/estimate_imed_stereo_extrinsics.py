#!/usr/bin/env python3
"""Estimate Endoscope-2 right-from-left extrinsics using source data only."""

import argparse
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


FORMAT = "imed_endoscope2_stereo_v1"


def parse_intrinsics(path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrices = {}
    index = 0
    while index < len(lines):
        if lines[index].startswith("# K"):
            key = lines[index][1:].strip().split()[0]
            matrices[key] = np.asarray(
                [[float(value) for value in lines[index + row].split()] for row in range(1, 4)],
                dtype=np.float64,
            )
            index += 4
        else:
            index += 1
    for key in ("K2_L", "K2_R"):
        if key not in matrices:
            raise ValueError(f"Missing {key} in {path}")
    return matrices


def frame_id(path):
    match = re.fullmatch(r"frame_(\d+)\.(?:png|npy)", path.name)
    if match is None:
        raise ValueError(f"Unexpected frame name: {path.name}")
    return int(match.group(1))


def indexed_files(directory, suffix):
    paths = sorted(directory.glob(f"*.{suffix}"))
    return {frame_id(path): path for path in paths}


def valid_tool_mask(path, shape):
    if path is None:
        return np.full(shape, 255, dtype=np.uint8)
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(raw == 0, 255, 0).astype(np.uint8)


def make_detector(max_features):
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02), cv2.NORM_L2, "SIFT"
    return cv2.ORB_create(nfeatures=max_features, fastThreshold=8), cv2.NORM_HAMMING, "ORB"


def collect_correspondences(left_path, right_path, depth_path, left_mask_path, right_mask_path, K_l, detector, norm):
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError(f"Could not read {left_path} or {right_path}")
    if left.shape != right.shape:
        raise ValueError("Synchronized Endoscope-2 L/R images have different shapes")
    depth = np.load(depth_path).astype(np.float64)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW: {depth_path}")
    scale_x = left.shape[1] / depth.shape[1]
    scale_y = left.shape[0] / depth.shape[0]
    if not np.isclose(scale_x, scale_y):
        raise ValueError("Anisotropic RGB/depth scaling is unsupported")

    mask_l = valid_tool_mask(left_mask_path, left.shape)
    mask_r = valid_tool_mask(right_mask_path, right.shape)
    kp_l, desc_l = detector.detectAndCompute(left, mask_l)
    kp_r, desc_r = detector.detectAndCompute(right, mask_r)
    if desc_l is None or desc_r is None or len(kp_l) < 8 or len(kp_r) < 8:
        return np.empty((0, 3)), np.empty((0, 2)), 0

    matcher = cv2.BFMatcher(norm)
    forward = matcher.knnMatch(desc_l, desc_r, k=2)
    reverse = matcher.knnMatch(desc_r, desc_l, k=2)
    reverse_best = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse
        if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance
    }
    matches = [
        pair[0]
        for pair in forward
        if len(pair) == 2
        and pair[0].distance < 0.78 * pair[1].distance
        and reverse_best.get(pair[0].trainIdx) == pair[0].queryIdx
    ]

    object_points, image_points = [], []
    for match in matches:
        u_l, v_l = kp_l[match.queryIdx].pt
        u_r, v_r = kp_r[match.trainIdx].pt
        depth_u = int(round(u_l / scale_x))
        depth_v = int(round(v_l / scale_y))
        if not (0 <= depth_u < depth.shape[1] and 0 <= depth_v < depth.shape[0]):
            continue
        z = float(depth[depth_v, depth_u])
        if not math.isfinite(z) or z <= 0:
            continue
        x = (u_l - K_l[0, 2]) * z / K_l[0, 0]
        y = (v_l - K_l[1, 2]) * z / K_l[1, 1]
        object_points.append((x, y, z))
        image_points.append((u_r, v_r))
    return np.asarray(object_points, dtype=np.float64), np.asarray(image_points, dtype=np.float64), len(matches)


def pose_matrix(rvec, tvec):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
    matrix[:3, 3] = np.asarray(tvec).reshape(3)
    return matrix


def rotation_error_degrees(a, b):
    relative = a[:3, :3] @ b[:3, :3].T
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="Path to one iMED sequence")
    parser.add_argument("--output-dir", default="calibration/imed")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--reprojection-threshold", type=float, default=2.0)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.60)
    parser.add_argument("--min-successful-frames", type=int, default=5)
    parser.add_argument("--max-rotation-std-deg", type=float, default=0.20)
    parser.add_argument("--max-translation-std", type=float, default=0.50)
    args = parser.parse_args()

    sequence_dir = Path(args.sequence).resolve()
    sequence_name = sequence_dir.name
    source = sequence_dir / "endoscope2"
    # Hard compliance guard: this script constructs source paths explicitly and
    # never enumerates or reads the Endoscope-1 directory.
    required = {
        "left": indexed_files(source / "L", "png"),
        "right": indexed_files(source / "R", "png"),
        "depth": indexed_files(source / "depthL", "npy"),
        "tool_l": indexed_files(source / "toolL", "png"),
        "tool_r": indexed_files(source / "toolR", "png"),
    }
    required_for_alignment = [required["left"], required["right"], required["depth"]]
    ids = sorted(set.intersection(*(set(files) for files in required_for_alignment)))
    if any(set(files) != set(ids) for files in required_for_alignment):
        raise RuntimeError("Endoscope-2 L/R/depth frame ids are not exactly aligned")
    for mask_key in ("tool_l", "tool_r"):
        if required[mask_key] and set(required[mask_key]) != set(ids):
            raise RuntimeError(f"Partial or misaligned source mask stream: endoscope2/{mask_key}")
    if len(ids) < args.min_successful_frames:
        raise RuntimeError(f"Only {len(ids)} complete Endoscope-2 stereo frames found")
    sample_indices = np.linspace(0, len(ids) - 1, min(args.max_frames, len(ids)), dtype=np.int64)
    selected_ids = [ids[int(index)] for index in np.unique(sample_indices)]

    intrinsics = parse_intrinsics(sequence_dir / "K.txt")
    K_l, K_r = intrinsics["K2_L"], intrinsics["K2_R"]
    detector, norm, detector_name = make_detector(args.max_features)
    frame_solutions = []
    pooled_object, pooled_image = [], []

    for current_id in selected_ids:
        obj, img, raw_matches = collect_correspondences(
            required["left"][current_id], required["right"][current_id],
            required["depth"][current_id], required["tool_l"].get(current_id),
            required["tool_r"].get(current_id), K_l, detector, norm,
        )
        if len(obj) < 8:
            continue
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K_r, None, iterationsCount=500,
            reprojectionError=args.reprojection_threshold, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 8:
            continue
        inlier_index = inliers.reshape(-1)
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                obj[inlier_index], img[inlier_index], K_r, None, rvec, tvec
            )
        transform = pose_matrix(rvec, tvec)
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K_r, None)
        errors = np.linalg.norm(projected.reshape(-1, 2) - img, axis=1)
        frame_solutions.append({
            "frame_id": current_id,
            "raw_matches": raw_matches,
            "correspondences": len(obj),
            "pnp_inliers": int((errors <= args.reprojection_threshold).sum()),
            "pnp_inlier_ratio": float((errors <= args.reprojection_threshold).mean()),
            "median_reprojection_px": float(np.median(errors[errors <= args.reprojection_threshold])),
            "transform": transform,
            "object_points": obj,
            "image_points": img,
        })
        pooled_object.append(obj[inlier_index])
        pooled_image.append(img[inlier_index])

    if len(frame_solutions) < args.min_successful_frames:
        raise RuntimeError(
            f"Only {len(frame_solutions)} frames produced valid source-only stereo poses; "
            f"need {args.min_successful_frames}"
        )

    rotations = Rotation.from_matrix(np.stack([item["transform"][:3, :3] for item in frame_solutions]))
    initial_rotation = rotations.mean().as_matrix()
    initial_translation = np.median(
        np.stack([item["transform"][:3, 3] for item in frame_solutions]), axis=0
    )
    initial_rvec = cv2.Rodrigues(initial_rotation)[0]
    all_object = np.concatenate(pooled_object, axis=0)
    all_image = np.concatenate(pooled_image, axis=0)
    if len(all_object) > 20000:
        keep = np.linspace(0, len(all_object) - 1, 20000, dtype=np.int64)
        all_object, all_image = all_object[keep], all_image[keep]
    ok, fixed_rvec, fixed_tvec = cv2.solvePnP(
        all_object, all_image, K_r, None,
        rvec=initial_rvec, tvec=initial_translation.reshape(3, 1),
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("Joint stereo pose refinement failed")
    fixed = pose_matrix(fixed_rvec, fixed_tvec)

    # Final acceptance is measured against the single fixed calibration, not
    # against independently fitted per-frame poses (which would be optimistic).
    for item in frame_solutions:
        projected, _ = cv2.projectPoints(
            item["object_points"], fixed_rvec, fixed_tvec, K_r, None
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - item["image_points"], axis=1)
        fixed_inliers = errors <= args.reprojection_threshold
        item["fixed_inliers"] = int(fixed_inliers.sum())
        item["fixed_inlier_ratio"] = float(fixed_inliers.mean())
        item["fixed_median_reprojection_px"] = (
            float(np.median(errors[fixed_inliers])) if fixed_inliers.any() else float("inf")
        )

    rotation_errors = np.asarray([rotation_error_degrees(item["transform"], fixed) for item in frame_solutions])
    translation_errors = np.asarray([
        np.linalg.norm(item["transform"][:3, 3] - fixed[:3, 3]) for item in frame_solutions
    ])
    ratios = np.asarray([item["fixed_inlier_ratio"] for item in frame_solutions])
    medians = np.asarray([item["fixed_median_reprojection_px"] for item in frame_solutions])
    validation = {
        "successful_frames": len(frame_solutions),
        "sampled_frames": len(selected_ids),
        "median_reprojection_px": float(np.median(medians)),
        "median_inlier_ratio": float(np.median(ratios)),
        "rotation_error_std_deg": float(rotation_errors.std()),
        "translation_error_std": float(translation_errors.std()),
        "thresholds": {
            "max_median_reprojection_px": args.reprojection_threshold,
            "min_median_inlier_ratio": args.min_inlier_ratio,
            "min_successful_frames": args.min_successful_frames,
            "max_rotation_error_std_deg": args.max_rotation_std_deg,
            "max_translation_error_std": args.max_translation_std,
        },
    }
    validation["passed"] = bool(
        validation["successful_frames"] >= args.min_successful_frames
        and validation["median_reprojection_px"] <= args.reprojection_threshold
        and validation["median_inlier_ratio"] >= args.min_inlier_ratio
        and validation["rotation_error_std_deg"] <= args.max_rotation_std_deg
        and validation["translation_error_std"] <= args.max_translation_std
    )
    payload = {
        "format": FORMAT,
        "sequence": sequence_name,
        "transform_convention": "X_R = transform_R_from_L @ X_L",
        "transform_R_from_L": fixed.tolist(),
        "intrinsics": {"K2_L": K_l.tolist(), "K2_R": K_r.tolist()},
        "provenance": {
            "source_data_only": True,
            "used_endoscope1": False,
            "inputs": ["endoscope2/L", "endoscope2/R", "endoscope2/depthL", "endoscope2/toolL", "endoscope2/toolR", "K2_L", "K2_R"],
            "detector": detector_name,
        },
        "validation": validation,
        "frames": [
            {
                key: value
                for key, value in item.items()
                if key not in {"transform", "object_points", "image_points"}
            }
            for item in frame_solutions
        ],
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sequence_name}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "validation": validation}, indent=2))
    if not validation["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
