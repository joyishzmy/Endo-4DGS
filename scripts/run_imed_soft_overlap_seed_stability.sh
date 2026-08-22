#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="/home/login/Documents/ZMY/imed4dgs"
PYTHON_BIN="/home/login/Documents/ZMY/conda-envs/imed4dgs/bin/python"
DATA_ROOT="${REPO_ROOT}/data/imed"
OUT_ROOT="${REPO_ROOT}/output/ablation/softoverlap_seed_stability_v1"
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
    session_004_scene_2_tool_1
)
methods=(baseline softoverlap)
configs=(
    arguments/imed_extent10_smooth002_iter2500.py
    arguments/imed_extent10_smooth002_softoverlap_iter2500.py
)

run_model() {
    local seed="$1"
    local method="$2"
    local config="$3"
    local seq="$4"
    local output="${OUT_ROOT}/seed${seed}/${method}/${seq}"
    local log="${OUT_ROOT}/seed${seed}_${method}_${seq}.log"
    local status

    if [[ -s "${output}/results.json" ]]; then
        echo "Complete result exists, skipping: seed=${seed} / ${method} / ${seq}"
        return 0
    fi
    if [[ -e "${output}" ]]; then
        echo "Incomplete output exists; preserving it and stopping: ${output}"
        return 3
    fi
    if ! check_gpu0; then
        return 2
    fi

    echo "===== Training: seed=${seed} / ${method} / ${seq} ====="
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" train.py \
        -s "${DATA_ROOT}/${seq}" \
        -m "${output}" \
        --configs "${config}" \
        --seed "${seed}" \
        --port "${PORT}" \
        --save_iterations 2000 2500 \
        2>&1 | tee "${log}"
    status=${PIPESTATUS[0]}
    if (( status != 0 )); then
        echo "Training failed with exit code ${status}; output was preserved: ${output}"
        return "${status}"
    fi

    echo "===== Rendering: seed=${seed} / ${method} / ${seq} ====="
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

    echo "===== Metrics: seed=${seed} / ${method} / ${seq} ====="
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
echo "Reusing the completed seed=0 paired results from softoverlap_paired_v1."

for seed in 1 2; do
    for method_idx in "${!methods[@]}"; do
        method="${methods[$method_idx]}"
        config="${configs[$method_idx]}"
        for seq in "${seqs[@]}"; do
            run_model "${seed}" "${method}" "${config}" "${seq}" || exit $?
        done
    done
done

echo "All seed-stability experiments completed."
