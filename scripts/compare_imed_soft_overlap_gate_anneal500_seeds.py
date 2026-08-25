#!/usr/bin/env python3

import json
import statistics
from pathlib import Path


GATE_STABILITY_ROOT = Path("output/ablation/softoverlap_gate090_seed_stability_v1")
SEED1_REFERENCE_ROOT = Path("output/ablation/softoverlap_gate090_extension_v1")
ANNEAL_STABILITY_ROOT = Path(
    "output/ablation/softoverlap_gate090_anneal500_seed_stability_v1"
)
SEED1_ANNEAL_ROOT = Path("output/ablation/softoverlap_gate090_anneal500_v1")
SEQUENCES = (
    "session_005_scene_7_tool_1",
    "session_006_scene_7_tool_3",
)
METRICS = ("PSNR", "SSIM", "LPIPS")


def result_path(seed, method, sequence):
    if seed == 1:
        if method == "anneal500":
            return SEED1_ANNEAL_ROOT / method / sequence / "results.json"
        return SEED1_REFERENCE_ROOT / method / sequence / "results.json"
    if method == "anneal500":
        return (
            ANNEAL_STABILITY_ROOT
            / f"seed{seed}"
            / method
            / sequence
            / "results.json"
        )
    return GATE_STABILITY_ROOT / f"seed{seed}" / method / sequence / "results.json"


def read_result(seed, method, sequence):
    with result_path(seed, method, sequence).open() as handle:
        return json.load(handle)["ours_2500"]


for sequence in SEQUENCES:
    gate_deltas = {metric: [] for metric in METRICS}
    anneal_deltas = {metric: [] for metric in METRICS}
    print(f"\n===== {sequence} =====")
    for seed in (0, 1, 2):
        baseline = read_result(seed, "baseline", sequence)
        gated = read_result(seed, "gate090", sequence)
        annealed = read_result(seed, "anneal500", sequence)
        gate_values = {metric: gated[metric] - baseline[metric] for metric in METRICS}
        anneal_values = {
            metric: annealed[metric] - baseline[metric] for metric in METRICS
        }
        for metric in METRICS:
            gate_deltas[metric].append(gate_values[metric])
            anneal_deltas[metric].append(anneal_values[metric])
        print(
            f"seed={seed}: "
            f"gate(dP={gate_values['PSNR']:+.6f}, "
            f"dS={gate_values['SSIM']:+.6f}, dL={gate_values['LPIPS']:+.6f})  "
            f"anneal(dP={anneal_values['PSNR']:+.6f}, "
            f"dS={anneal_values['SSIM']:+.6f}, dL={anneal_values['LPIPS']:+.6f})"
        )

    print("Three-seed mean +/- population std:")
    for metric in METRICS:
        gate = gate_deltas[metric]
        anneal = anneal_deltas[metric]
        print(
            f"{metric:5s}: gate={statistics.mean(gate):+.8f} +/- "
            f"{statistics.pstdev(gate):.8f}  "
            f"anneal={statistics.mean(anneal):+.8f} +/- "
            f"{statistics.pstdev(anneal):.8f}"
        )
