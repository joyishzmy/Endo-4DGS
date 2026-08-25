#!/usr/bin/env python3

import json
from pathlib import Path


REFERENCE_ROOT = Path("output/ablation/softoverlap_gate090_extension_v1")
ANNEAL_ROOT = Path("output/ablation/softoverlap_gate090_anneal500_v1")
SEQUENCES = (
    "session_005_scene_7_tool_1",
    "session_006_scene_7_tool_3",
)
METRICS = ("PSNR", "SSIM", "LPIPS")


def read_result(path):
    with path.open() as handle:
        return json.load(handle)["ours_2500"]


for sequence in SEQUENCES:
    baseline = read_result(REFERENCE_ROOT / "baseline" / sequence / "results.json")
    gated = read_result(REFERENCE_ROOT / "gate090" / sequence / "results.json")
    annealed = read_result(ANNEAL_ROOT / "anneal500" / sequence / "results.json")
    print(f"\n===== {sequence} / seed=1 =====")
    for metric in METRICS:
        print(
            f"{metric:5s}: baseline={baseline[metric]:.8f}  "
            f"gate090={gated[metric]:.8f}  anneal500={annealed[metric]:.8f}  "
            f"anneal-baseline={annealed[metric] - baseline[metric]:+.8f}"
        )
