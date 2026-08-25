#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path("output/ablation/stereo_rgbd_pilot_v1")
SEQUENCES = ["session_004_scene_6_tool_3", "session_007_scene_5_tool_2"]
METHODS = ["baseline", "g1_rgbd", "g2_stereo"]
METRICS = ["PSNR", "SSIM", "LPIPS"]


def read(method, sequence):
    payload = json.loads((ROOT / method / sequence / "results.json").read_text())
    return payload["ours_2500"]


values = {method: {sequence: read(method, sequence) for sequence in SEQUENCES} for method in METHODS}
for sequence in SEQUENCES:
    print(f"\n===== {sequence} =====")
    for method in METHODS:
        result = values[method][sequence]
        delta = {metric: result[metric] - values["baseline"][sequence][metric] for metric in METRICS}
        print(
            f"{method:9s} PSNR={result['PSNR']:.8f} ({delta['PSNR']:+.8f})  "
            f"SSIM={result['SSIM']:.8f} ({delta['SSIM']:+.8f})  "
            f"LPIPS={result['LPIPS']:.8f} ({delta['LPIPS']:+.8f})"
        )

print("\n===== Two-sequence equal-weight mean delta vs baseline =====")
for method in METHODS[1:]:
    deltas = {
        metric: sum(values[method][seq][metric] - values["baseline"][seq][metric] for seq in SEQUENCES) / len(SEQUENCES)
        for metric in METRICS
    }
    print(f"{method:9s} dPSNR={deltas['PSNR']:+.8f} dSSIM={deltas['SSIM']:+.8f} dLPIPS={deltas['LPIPS']:+.8f}")
