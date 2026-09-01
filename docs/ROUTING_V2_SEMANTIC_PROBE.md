# Routing-V2 Semantic AE Bank Passive Probe

This probe validates only the first-stage semantic retrieval of Routing-V2.

- Candidate bank: T6–T9 Semantic AEs (CL-only).
- Execution remains oracle/task-ID: T6 rollout executes Skill T6, etc.
- Semantic ranking never changes the action-producing checkpoint.
- Every action-chunk policy call computes the common V2 anchor:
  Base VLM (task text-LoRA disabled) + Base act/flow queries (task query residual ignored).
- The same `H_act_base` is scored by all four Semantic AEs.
- Lower reconstruction MSE is ranked better.

Primary metrics are chunk-level Top-1 accuracy and Top-2 recall. The probe also saves
Top-1 confusion, GT-rank histograms, mean reconstruction-error matrix, and expert
selection frequency to diagnose AE score bias.

Recommended sequence:

```bash
# wiring smoke
NUM_TRIALS=2 EVAL_WORKERS=4 \
  bash scripts/eval_libero_goal_routing_v2_semantic_bank.sh

# useful first diagnostic
NUM_TRIALS=10 EVAL_WORKERS=8 \
  bash scripts/eval_libero_goal_routing_v2_semantic_bank.sh
```
