#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path("output/ablation/softoverlap_paired_v1")
REUSED_BASELINE_ROOT = Path("output/ablation/sourceoverlap_paired_v1/baseline")
SEQUENCES = (
    "session_005_scene_7_tool_2",
    "session_004_scene_6_tool_3",
    "session_007_scene_11_tool_3",
    "session_004_scene_2_tool_1",
)
METRICS = ("PSNR", "SSIM", "LPIPS")


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["ours_2500"]


def load_baseline(sequence):
    current = ROOT / "baseline" / sequence / "results.json"
    reused = REUSED_BASELINE_ROOT / sequence / "results.json"
    return load_json(current if current.is_file() else reused)


def main():
    values = {"baseline": {}, "softoverlap": {}}
    for sequence in SEQUENCES:
        values["baseline"][sequence] = load_baseline(sequence)
        values["softoverlap"][sequence] = load_json(
            ROOT / "softoverlap" / sequence / "results.json"
        )

        print(f"\n===== {sequence} =====")
        for metric in METRICS:
            baseline = values["baseline"][sequence][metric]
            soft = values["softoverlap"][sequence][metric]
            print(
                f"{metric:5s}: baseline={baseline:.8f}  "
                f"softoverlap={soft:.8f}  delta={soft - baseline:+.8f}"
            )

    print("\n===== Four-sequence equal-weight mean =====")
    for metric in METRICS:
        baseline = sum(values["baseline"][s][metric] for s in SEQUENCES) / len(SEQUENCES)
        soft = sum(values["softoverlap"][s][metric] for s in SEQUENCES) / len(SEQUENCES)
        print(
            f"{metric:5s}: baseline={baseline:.8f}  "
            f"softoverlap={soft:.8f}  delta={soft - baseline:+.8f}"
        )


if __name__ == "__main__":
    main()
