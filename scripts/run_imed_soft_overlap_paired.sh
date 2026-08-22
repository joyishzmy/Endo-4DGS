#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="/home/login/Documents/ZMY/imed4dgs"
PYTHON_BIN="/home/login/Documents/ZMY/conda-envs/imed4dgs/bin/python"
DATA_ROOT="${REPO_ROOT}/data/imed"
OUT_ROOT="${REPO_ROOT}/output/ablation/softoverlap_paired_v1"
REUSED_BASELINE_ROOT="${REPO_ROOT}/output/ablation/sourceoverlap_paired_v1/baseline"
PORT="${IMED_PORT:-6012}"

cd "${REPO_ROOT}" || exit 1

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}"
    exit 1
fi

check_gpu0() {
    local gpu_util gpu_mem
    gpu_util=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    gpu_mem=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( gpu_util > 10 || gpu_mem > 1000 )); then
        echo "GPU0 is busy: utilization=${gpu_util}%, memory=${gpu_mem} MiB. Retry later."
        return 1
    fi
}

if ss -ltnH "sport = :${PORT}" | grep -q .; then
    echo "Port ${PORT} is occupied. Retry with: IMED_PORT=6013 $0"
    exit 2
fi

seqs=(
    session_005_scene_7_tool_2
    session_004_scene_6_tool_3
    session_007_scene_11_tool_3
    session_004_scene_2_tool_1
)

run_model() {
    local method="$1"
    local config="$2"
    local seq="$3"
    local source="${DATA_ROOT}/${seq}"
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
    if ! check_gpu0; then
        return 2
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

for seq in "${seqs[@]}"; do
    if [[ -s "${OUT_ROOT}/baseline/${seq}/results.json" ]]; then
        echo "Complete baseline exists in the current experiment: ${seq}"
    elif [[ -s "${REUSED_BASELINE_ROOT}/${seq}/results.json" ]]; then
        echo "Reusing the compliant paired baseline from sourceoverlap_paired_v1: ${seq}"
    else
        run_model baseline arguments/imed_extent10_smooth002_iter2500.py "${seq}" || exit $?
    fi
done

for seq in "${seqs[@]}"; do
    run_model softoverlap arguments/imed_extent10_smooth002_softoverlap_iter2500.py "${seq}" || exit $?
done

echo "All stratified soft-overlap experiments completed."
