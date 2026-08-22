#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="/home/login/Documents/ZMY/imed4dgs"
PYTHON_BIN="/home/login/Documents/ZMY/conda-envs/imed4dgs/bin/python"
DATA_ROOT="${REPO_ROOT}/data/imed"
OUT_ROOT="${REPO_ROOT}/output/ablation/sourceoverlap_paired_v1"
PORT="${IMED_PORT:-6012}"

cd "${REPO_ROOT}" || exit 1

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}"
    exit 1
fi

gpu_util=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
gpu_mem=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if (( gpu_util > 10 || gpu_mem > 1000 )); then
    echo "GPU0 is busy: utilization=${gpu_util}%, memory=${gpu_mem} MiB. Retry later."
    exit 2
fi

if ss -ltnH "sport = :${PORT}" | grep -q .; then
    echo "Port ${PORT} is occupied. Retry with: IMED_PORT=6013 $0"
    exit 2
fi

methods=(baseline sourceoverlap)
configs=(
    arguments/imed_extent10_smooth002_iter2500.py
    arguments/imed_extent10_smooth002_sourceoverlap_iter2500.py
)
seqs=(
    session_004_scene_6_tool_3
    session_007_scene_11_tool_3
    session_004_scene_2_tool_1
)

mkdir -p "${OUT_ROOT}"

for method_idx in "${!methods[@]}"; do
    method="${methods[$method_idx]}"
    config="${configs[$method_idx]}"

    for seq in "${seqs[@]}"; do
        source="${DATA_ROOT}/${seq}"
        output="${OUT_ROOT}/${method}/${seq}"
        log="${OUT_ROOT}/${method}_${seq}.log"

        if [[ -s "${output}/results.json" ]]; then
            echo "Complete result exists, skipping: ${method} / ${seq}"
            continue
        fi
        if [[ -e "${output}" ]]; then
            echo "Incomplete output exists; preserving it and stopping: ${output}"
            exit 3
        fi

        echo "===== Training: ${method} / ${seq} ====="
        CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" train.py \
            -s "${source}" \
            -m "${output}" \
            --configs "${config}" \
            --port "${PORT}" \
            --save_iterations 2000 2500 \
            2>&1 | tee "${log}"
        status=${PIPESTATUS[0]}
        if (( status != 0 )); then
            echo "Training failed with exit code ${status}; output was preserved: ${output}"
            exit "${status}"
        fi

        echo "===== Rendering: ${method} / ${seq} ====="
        CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" render.py \
            -m "${output}" \
            --iteration 2500 \
            --skip_train \
            --skip_video \
            2>&1 | tee -a "${log}"
        status=${PIPESTATUS[0]}
        if (( status != 0 )); then
            echo "Rendering failed with exit code ${status}; output was preserved: ${output}"
            exit "${status}"
        fi

        echo "===== Metrics: ${method} / ${seq} ====="
        CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" metrics.py \
            -m "${output}" \
            2>&1 | tee -a "${log}"
        status=${PIPESTATUS[0]}
        if (( status != 0 )); then
            echo "Metric evaluation failed with exit code ${status}; output was preserved: ${output}"
            exit "${status}"
        fi
    done
done

echo "All paired source-overlap experiments completed."
