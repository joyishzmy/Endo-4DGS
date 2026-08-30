#!/usr/bin/env bash
# Match the official iMED NVS evaluator contract locally.
# Usage: ./scripts/local_test.sh <image> <input_dir> <output_dir>

set -euo pipefail

IMAGE="${1:?Usage: $0 <image> <input_dir> <output_dir>}"
INPUT="${2:?Missing input_dir}"
OUTPUT="${3:?Missing output_dir}"

test -d "$INPUT"
mkdir -p "$OUTPUT"

timeout 10800 docker run --rm \
  --gpus '"device=0"' \
  --network=none \
  --memory=120g \
  --pids-limit=1024 \
  --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e IMED_SEED=1 \
  -v "$(realpath "$INPUT")":/input:ro \
  -v "$(realpath "$OUTPUT")":/output \
  "$IMAGE"

python3 imed_nvs_submission/validate_output.py "$INPUT" "$OUTPUT"
