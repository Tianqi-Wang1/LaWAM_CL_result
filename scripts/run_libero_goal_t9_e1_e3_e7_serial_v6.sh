#!/usr/bin/env bash
set -euo pipefail

export TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
export POLICY_GPU="${POLICY_GPU:-4}"
export EVAL_GPU="${EVAL_GPU:-5}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_TRIALS="${NUM_TRIALS:-50}"
export VLM_LORA_RANK=32
export VLM_LORA_ALPHA=32

echo "================ SERIAL 1/3: E1 ================"
bash scripts/run_libero_goal_t9_e1_v6.sh

echo "================ SERIAL 2/3: E3 ================"
export FLOW_LORA_RANK=32
export FLOW_LORA_ALPHA=32
bash scripts/run_libero_goal_t9_e3_v6.sh

echo "================ SERIAL 3/3: E7 ================"
export CONDITIONING_BOTTLENECK=128
bash scripts/run_libero_goal_t9_e7_v6.sh

echo "[OK] E1/E3/E7 serial protocol complete."
