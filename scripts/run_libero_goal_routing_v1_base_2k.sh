#!/usr/bin/env bash
set -euo pipefail
echo "[WARN] Routing-V1 Base protocol has been updated: Base is now 10K, not 2K."
echo "[WARN] Forwarding to scripts/run_libero_goal_routing_v1_base_10k.sh"
exec bash scripts/run_libero_goal_routing_v1_base_10k.sh "$@"
