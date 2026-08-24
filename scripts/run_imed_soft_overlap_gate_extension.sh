#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="/home/login/Documents/ZMY/imed4dgs"
PYTHON_BIN="/home/login/Documents/ZMY/conda-envs/imed4dgs/bin/python"
DATA_ROOT="${REPO_ROOT}/data/imed"
OUT_ROOT="${REPO_ROOT}/output/ablation/softoverlap_gate090_extension_v1"
PORT="${IMED_PORT:-6026}"
SEED=1

cd "${REPO_ROOT}" || exit 1

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}"
    exit 1
fi

if ss -ltnH "sport = :${PORT}" | grep -q .; then
    echo "Port ${PORT} is occupied. Retry with: IMED_PORT=6027 $0"
    exit 2
fi

seqs=(
    session_005_scene_7_tool_1
    session_007_scene_10_tool_1
    session_004_scene_6_tool_1
    session_004_scene_6_tool_2
    session_006_scene_7_tool_1
    session_006_scene_7_tool_3
)
methods=(baseline gate090)
configs=(
    arguments/imed_extent10_smooth002_iter2500.py
    arguments/imed_extent10_smooth002_softoverlap_gate090_iter2500.py
)

wait_for_gpu0() {
    local gpu_util gpu_mem
    while true; do
        gpu_util=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
        gpu_mem=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        if (( gpu_util <= 10 && gpu_mem <= 1000 )); then
            return 0
        fi
        echo "GPU0 is busy: utilization=${gpu_util}%, memory=${gpu_mem} MiB. Retrying in 30 s."
        sleep 30
    done
}

run_model() {
    local method="$1"
    local config="$2"
    local seq="$3"
    local output="${OUT_ROOT}/${method}/${seq}"
    local log="${OUT_ROOT}/${method}_${seq}.log"
    local status

    if [[ -s "${output}/results.json" ]]; then
        echo "Complete result exists, skipping: ${method} / ${seq}"
        return 0
    fi
    if [[ -e "${output}" ]]; then
        echo "Incomplete output exists; preserving it and stopping: ${output}"
        return 3
    fi
    wait_for_gpu0

    echo "===== Training: seed=${SEED} / ${method} / ${seq} ====="
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" train.py \
        -s "${DATA_ROOT}/${seq}" \
        -m "${output}" \
        --configs "${config}" \
        --seed "${SEED}" \
        --port "${PORT}" \
        --save_iterations 2000 2500 \
        2>&1 | tee "${log}"
    status=${PIPESTATUS[0]}
    if (( status != 0 )); then
        echo "Training failed with exit code ${status}; output was preserved: ${output}"
        return "${status}"
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
        return "${status}"
    fi

    echo "===== Metrics: ${method} / ${seq} ====="
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" metrics.py \
        -m "${output}" \
        2>&1 | tee -a "${log}"
    status=${PIPESTATUS[0]}
    if (( status != 0 )); then
        echo "Metric evaluation failed with exit code ${status}; output was preserved: ${output}"
        return "${status}"
    fi
}

mkdir -p "${OUT_ROOT}"

for method_idx in "${!methods[@]}"; do
    method="${methods[$method_idx]}"
    config="${configs[$method_idx]}"
    for seq in "${seqs[@]}"; do
        run_model "${method}" "${config}" "${seq}" || exit $?
    done
done

"${PYTHON_BIN}" scripts/compare_imed_soft_overlap_gate_extension.py
echo "All six-sequence gate090 confirmation experiments completed."
