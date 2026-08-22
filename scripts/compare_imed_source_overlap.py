#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path("output/ablation/sourceoverlap_paired_v1")
SEQUENCES = (
    "session_004_scene_6_tool_3",
    "session_007_scene_11_tool_3",
    "session_004_scene_2_tool_1",
)
METRICS = ("PSNR", "SSIM", "LPIPS")


def load_result(method, sequence):
    path = ROOT / method / sequence / "results.json"
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    return result["ours_2500"]


def main():
    values = {method: {} for method in ("baseline", "sourceoverlap")}
    for sequence in SEQUENCES:
        for method in values:
            values[method][sequence] = load_result(method, sequence)

        print(f"\n===== {sequence} =====")
        for metric in METRICS:
            baseline = values["baseline"][sequence][metric]
            source_overlap = values["sourceoverlap"][sequence][metric]
            print(
                f"{metric:5s}: baseline={baseline:.8f}  "
                f"sourceoverlap={source_overlap:.8f}  "
                f"delta={source_overlap - baseline:+.8f}"
            )

    print("\n===== Three-sequence equal-weight mean =====")
    for metric in METRICS:
        baseline = sum(values["baseline"][s][metric] for s in SEQUENCES) / len(SEQUENCES)
        source_overlap = sum(values["sourceoverlap"][s][metric] for s in SEQUENCES) / len(SEQUENCES)
        print(
            f"{metric:5s}: baseline={baseline:.8f}  "
            f"sourceoverlap={source_overlap:.8f}  "
            f"delta={source_overlap - baseline:+.8f}"
        )


if __name__ == "__main__":
    main()
