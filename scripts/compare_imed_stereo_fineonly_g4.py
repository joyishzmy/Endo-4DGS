#!/usr/bin/env python3

import json
from pathlib import Path


SEQUENCES = ["session_004_scene_6_tool_3", "session_007_scene_5_tool_2"]
METHOD_PATHS = {
    "baseline_pp": Path("output/ablation/stereo_principal_v2/baseline_pp"),
    "g2v2_rgb": Path("output/ablation/stereo_principal_v2/g2v2_rgb"),
    "g3_photogate": Path("output/ablation/stereo_photogate_g3"),
    "g4_fineonly": Path("output/ablation/stereo_fineonly_g4"),
}
METRICS = ["PSNR", "SSIM", "LPIPS"]


def read(root, sequence):
    payload = json.loads((root / sequence / "results.json").read_text())
    return payload["ours_2500"]


values = {
    method: {sequence: read(root, sequence) for sequence in SEQUENCES}
    for method, root in METHOD_PATHS.items()
}
for sequence in SEQUENCES:
    print(f"\n===== {sequence} =====")
    baseline = values["baseline_pp"][sequence]
    for method in METHOD_PATHS:
        result = values[method][sequence]
        delta = {metric: result[metric] - baseline[metric] for metric in METRICS}
        print(
            f"{method:12s} PSNR={result['PSNR']:.8f} ({delta['PSNR']:+.8f})  "
            f"SSIM={result['SSIM']:.8f} ({delta['SSIM']:+.8f})  "
            f"LPIPS={result['LPIPS']:.8f} ({delta['LPIPS']:+.8f})"
        )

print("\n===== Two-sequence equal-weight mean delta vs baseline_pp =====")
for method in ("g2v2_rgb", "g3_photogate", "g4_fineonly"):
    deltas = {
        metric: sum(
            values[method][sequence][metric] - values["baseline_pp"][sequence][metric]
            for sequence in SEQUENCES
        ) / len(SEQUENCES)
        for metric in METRICS
    }
    print(
        f"{method:12s} dPSNR={deltas['PSNR']:+.8f} "
        f"dSSIM={deltas['SSIM']:+.8f} dLPIPS={deltas['LPIPS']:+.8f}"
    )
