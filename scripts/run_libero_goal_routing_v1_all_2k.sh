#!/usr/bin/env bash
set -euo pipefail
echo "[WARN] Routing-V1 protocol changed: Base=10K, experts=2K."
echo "[WARN] Forwarding to scripts/run_libero_goal_routing_v1_all_base10k_expert2k.sh"
exec bash scripts/run_libero_goal_routing_v1_all_base10k_expert2k.sh "$@"
