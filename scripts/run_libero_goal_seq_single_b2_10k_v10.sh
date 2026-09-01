#!/usr/bin/env bash
set -euo pipefail
MODE=b2 MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}" NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-600}" bash scripts/run_libero_goal_seq_single_strategy_v10.sh
