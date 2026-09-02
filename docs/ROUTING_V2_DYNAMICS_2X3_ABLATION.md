# Routing-V2: 2x3 Future-Dynamics Ablation

This package trains only new Dynamics routing memories. Existing V2 Base, T6-T9 Skill checkpoints,
and the existing Semantic AE bank are reused unchanged.

## Factor A: WM source used only for routing imagination

- `task`: `WM_base + WM-LoRA_k` (current V2 behavior)
- `base`: shared `WM_base`; task WM LoRA is temporarily disabled only while producing the routing future

The selected skill's normal execution still uses its complete task-specific Skill Path, including WM LoRA.

## Factor B: Dynamics verifier input

- `hdhz`: `[h_t, Delta h, z]`, reconstruct `Delta h + z` (current baseline)
- `hdh`: `[h_t, Delta h]`, reconstruct `Delta h` only
- `dh`: `[Delta h]`, reconstruct `Delta h` only

This gives six variants:

| ID | Variant | Routing WM | Dynamics input |
|---|---|---|---|
| A1 | `taskwm_hdhz` | Task WM | `[h, Delta h, z]` |
| A2 | `taskwm_hdh` | Task WM | `[h, Delta h]` |
| A3 | `taskwm_dh` | Task WM | `[Delta h]` |
| B1 | `basewm_hdhz` | Shared Base WM | `[h, Delta h, z]` |
| B2 | `basewm_hdh` | Shared Base WM | `[h, Delta h]` |
| B3 | `basewm_dh` | Shared Base WM | `[Delta h]` |

A1 is the already-trained Routing-V2 Dynamics baseline. It is extracted into the uniform variant
layout but is **not retrained**. Therefore only five new memories are trained per task.

## Training control

- Existing Skill Path: frozen
- Existing Semantic AE: reused; not retrained
- New Dynamics AE only: trainable
- GT/predicted transition mixture: 0.5 / 0.5
- Default: 1000 steps, same budget as the original V2 memory
- Eval loss: every 100 steps by default so 1000-step convergence can be checked before extending budget

For Base-WM variants, `z_k` remains task-specific (VLM-LoRA/query/QFormer-LoRA), but `h_pred` is
computed with the task LaWM LoRA residual scaling set to zero. This creates a common dynamics
coordinate system without changing task execution.

## Storage

Only `dynamics_ae.pt` plus metadata/config are kept for each new variant. The temporary full policy
checkpoint produced by the existing trainer is deleted after extraction.

Layout:

```text
routing_v2/task6/routing_memory_variants/
  taskwm_hdhz/dynamics_ae.pt   # extracted existing baseline
  taskwm_hdh/dynamics_ae.pt
  taskwm_dh/dynamics_ae.pt
  basewm_hdhz/dynamics_ae.pt
  basewm_hdh/dynamics_ae.pt
  basewm_dh/dynamics_ae.pt
```

## Smoke

Validates one Task-WM/no-z path and the proposed Shared-WM/no-z path:

```bash
MAX_TRAIN_STEPS=20 bash scripts/run_libero_goal_routing_v2_dynamics_ablation_smoke.sh
```

## Formal training: all six variants / T6-T9

A1 is reference-only; the other five variants are trained for 1000 steps on all four tasks.

```bash
MAX_TRAIN_STEPS=1000 \
TRAIN_GPUS=4,5,6,7 \
PER_DEVICE_BATCH_SIZE=64 \
GRADIENT_ACCUMULATION_STEPS=1 \
TRAIN_EVAL_INTERVAL=100 \
TRAIN_EVAL_BATCHES=20 \
  bash scripts/run_libero_goal_routing_v2_dynamics_ablation_all.sh
```

Do not run six full 500-episode closed-loop evaluations immediately. First run the same passive
Semantic-Top2 Dynamics probe for all six variants under fixed fusion hyperparameters
`delta=0.2, lambda_max=0.5, gamma=2`; select the best one or two, then rerun the formal CL1-CL4
closed-loop evaluation.
