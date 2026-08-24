#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path("output/ablation/softoverlap_gate090_extension_v1")
SEQUENCES = (
    "session_005_scene_7_tool_1",
    "session_007_scene_10_tool_1",
    "session_004_scene_6_tool_1",
    "session_004_scene_6_tool_2",
    "session_006_scene_7_tool_1",
    "session_006_scene_7_tool_3",
)
METRICS = ("PSNR", "SSIM", "LPIPS")


def read_result(method, sequence):
    path = ROOT / method / sequence / "results.json"
    with path.open() as handle:
        return json.load(handle)["ours_2500"]


deltas = {metric: [] for metric in METRICS}
for sequence in SEQUENCES:
    baseline = read_result("baseline", sequence)
    gated = read_result("gate090", sequence)
    print(f"\n===== {sequence} =====")
    for metric in METRICS:
        delta = gated[metric] - baseline[metric]
        deltas[metric].append(delta)
        print(
            f"{metric:5s}: baseline={baseline[metric]:.8f}  "
            f"gate090={gated[metric]:.8f}  delta={delta:+.8f}"
        )

print("\n===== Six-sequence equal-weight mean delta =====")
for metric in METRICS:
    values = deltas[metric]
    print(f"{metric:5s}: mean_delta={sum(values) / len(values):+.8f}")
