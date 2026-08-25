#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${IMED_PYTHON_BIN:-/home/login/Documents/ZMY/conda-envs/imed4dgs/bin/python}"
DATA_ROOT="${IMED_DATA_ROOT:-${REPO_ROOT}/data/imed}"
CALIBRATION_ROOT="${REPO_ROOT}/calibration/imed"
OUT_ROOT="${REPO_ROOT}/output/ablation/stereo_principal_v2"
PORT="${IMED_PORT:-6042}"
SEED="${IMED_SEED:-1}"

cd "${REPO_ROOT}" || exit 1

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}"
    exit 1
fi

seqs=(session_004_scene_6_tool_3 session_007_scene_5_tool_2)
methods=(baseline_pp g1_pp g2v2_rgb)
configs=(
    arguments/imed_extent10_principal_baseline_iter2500.py
    arguments/imed_extent10_principal_rgbd_g1_iter2500.py
    arguments/imed_extent10_principal_stereo_rgb_g2v2_iter2500.py
)

wait_for_gpu0() {
    local gpu_util gpu_mem
    while true; do
        gpu_util=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
        gpu_mem=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        if (( gpu_util <= 10 && gpu_mem <= 1000 )); then
            return 0
        fi
        echo "GPU0 busy: utilization=${gpu_util}%, memory=${gpu_mem} MiB; retrying in 30 s."
        sleep 30
    done
}

run_model() {
    local method="$1" config="$2" seq="$3"
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
        -s "${DATA_ROOT}/${seq}" -m "${output}" \
        --configs "${config}" --seed "${SEED}" --port "${PORT}" \
        --save_iterations 2000 2500 2>&1 | tee "${log}"
    status=${PIPESTATUS[0]}
    (( status == 0 )) || return "${status}"

    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" render.py \
        -m "${output}" --iteration 2500 --skip_train --skip_video \
        2>&1 | tee -a "${log}"
    status=${PIPESTATUS[0]}
    (( status == 0 )) || return "${status}"

    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" metrics.py -m "${output}" 2>&1 | tee -a "${log}"
    status=${PIPESTATUS[0]}
    (( status == 0 )) || return "${status}"
}

mkdir -p "${CALIBRATION_ROOT}" "${OUT_ROOT}"
"${PYTHON_BIN}" scripts/test_imed_stereo_geometry.py || exit $?
"${PYTHON_BIN}" scripts/test_imed_principal_projection.py || exit $?

for seq in "${seqs[@]}"; do
    calibration="${CALIBRATION_ROOT}/${seq}.json"
    if [[ ! -s "${calibration}" ]]; then
        echo "Missing calibration: ${calibration}"
        exit 2
    fi
done

for method_index in "${!methods[@]}"; do
    for seq in "${seqs[@]}"; do
        run_model "${methods[$method_index]}" "${configs[$method_index]}" "${seq}" || {
            status=$?
            echo "Experiment failed (${status}); output/log preserved."
            exit "${status}"
        }
    done
done

"${PYTHON_BIN}" scripts/compare_imed_stereo_principal_v2.py
echo "Principal-point/RGB-only stereo v2 pilot completed."
