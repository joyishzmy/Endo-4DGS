#!/usr/bin/env python3

import json
import statistics
from pathlib import Path


STABILITY_ROOT = Path("output/ablation/softoverlap_gate090_seed_stability_v1")
SEED1_ROOT = Path("output/ablation/softoverlap_gate090_extension_v1")
SEQUENCES = (
    "session_005_scene_7_tool_1",
    "session_006_scene_7_tool_3",
)
METRICS = ("PSNR", "SSIM", "LPIPS")


def result_path(seed, method, sequence):
    if seed == 1:
        return SEED1_ROOT / method / sequence / "results.json"
    return STABILITY_ROOT / f"seed{seed}" / method / sequence / "results.json"


def read_result(seed, method, sequence):
    with result_path(seed, method, sequence).open() as handle:
        return json.load(handle)["ours_2500"]


for sequence in SEQUENCES:
    deltas = {metric: [] for metric in METRICS}
    print(f"\n===== {sequence} =====")
    for seed in (0, 1, 2):
        baseline = read_result(seed, "baseline", sequence)
        gated = read_result(seed, "gate090", sequence)
        values = {metric: gated[metric] - baseline[metric] for metric in METRICS}
        for metric in METRICS:
            deltas[metric].append(values[metric])
        print(
            f"seed={seed}: dPSNR={values['PSNR']:+.8f}  "
            f"dSSIM={values['SSIM']:+.8f}  dLPIPS={values['LPIPS']:+.8f}"
        )

    print("Three-seed delta summary:")
    for metric in METRICS:
        values = deltas[metric]
        print(
            f"{metric:5s}: mean={statistics.mean(values):+.8f}  "
            f"std={statistics.pstdev(values):.8f}"
        )
