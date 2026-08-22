#!/usr/bin/env python3

import json
from pathlib import Path
from statistics import fmean, pstdev


STABILITY_ROOT = Path("output/ablation/softoverlap_seed_stability_v1")
PAIRED_ROOT = Path("output/ablation/softoverlap_paired_v1")
REUSED_BASELINE_ROOT = Path("output/ablation/sourceoverlap_paired_v1/baseline")
SEQUENCES = (
    "session_005_scene_7_tool_2",
    "session_004_scene_2_tool_1",
)
METRICS = ("PSNR", "SSIM", "LPIPS")
SEEDS = (0, 1, 2)


def read_result(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["ours_2500"]


def result_path(seed, method, sequence):
    if seed != 0:
        return STABILITY_ROOT / f"seed{seed}" / method / sequence / "results.json"
    if method == "softoverlap":
        return PAIRED_ROOT / method / sequence / "results.json"
    current = PAIRED_ROOT / method / sequence / "results.json"
    reused = REUSED_BASELINE_ROOT / sequence / "results.json"
    return current if current.is_file() else reused


def main():
    for sequence in SEQUENCES:
        print(f"\n===== {sequence} =====")
        deltas = {metric: [] for metric in METRICS}
        for seed in SEEDS:
            baseline = read_result(result_path(seed, "baseline", sequence))
            soft = read_result(result_path(seed, "softoverlap", sequence))
            values = []
            for metric in METRICS:
                delta = soft[metric] - baseline[metric]
                deltas[metric].append(delta)
                values.append(f"d{metric}={delta:+.8f}")
            print(f"seed={seed}: " + "  ".join(values))

        print("Three-seed delta summary:")
        for metric in METRICS:
            print(
                f"{metric:5s}: mean={fmean(deltas[metric]):+.8f}  "
                f"std={pstdev(deltas[metric]):.8f}"
            )


if __name__ == "__main__":
    main()
