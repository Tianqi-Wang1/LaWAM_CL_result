#!/usr/bin/env bash
set -euo pipefail
MODE=e1 VLM_LORA_RANK="${VLM_LORA_RANK:-32}" VLM_LORA_ALPHA="${VLM_LORA_ALPHA:-32}" \
  bash scripts/run_libero_goal_t9_e1_e3_e7_variant_v6.sh
