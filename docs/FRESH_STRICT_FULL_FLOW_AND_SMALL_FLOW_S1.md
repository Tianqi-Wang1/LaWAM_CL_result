# Fresh Strict Full-Flow T9 and Small Flow S1

This implementation provides two isolated Goal/T9 experiments. Both use the
Formal Goal Base checkpoint, Base normalization statistics, the original Flow
Matching objective, and the existing LIBERO closed-loop evaluator.

## What changed

- `ConditionalFlowMatchingConfig` now records `expert_variant` and optionally
  accepts an explicit `ffn_dim`.
- `expert_variant=small_s1` is validated as exactly:
  `hidden_dim=512`, `num_layers=6`, `attention_heads=8`, `ffn_dim=2048`.
- The original full architecture and old checkpoints remain compatible because
  the new fields default to `expert_variant=full` and `ffn_dim=None`.
- `trainer.strict_finetune_init=true` fails before training when a requested
  checkpoint tensor is missing, unexpected, or shape-incompatible.
- `trainer.load_pretrained_policy_flow=false` now skips both Flow state-dict
  aliases:
  - `policy_backend.flow.*`
  - `policy_action_head.*`
- The global experiment seed is applied before model construction, so fresh
  Small Flow initialization is reproducible and identical across DDP ranks.

The alias fix is essential. Without it, same-shaped tensors under
`policy_action_head.*` could silently initialize part of a supposedly fresh
Small Flow from the Base Flow.

## Experiment 1: Fresh Strict Full-Flow T9

Run:

```bash
TRAIN_GPUS=4,5,6,7 \
POLICY_GPU=4 \
EVAL_GPU=5 \
bash scripts/run_libero_goal_full_flow_t9_retrain_v1.sh
```

Protocol:

- Task: `libero_goal`, T9 only.
- Initialization: the complete 306M Flow is loaded from Formal Base.
- Trainable parameters: `policy_backend.flow.*` only.
- Capacity: 1024 hidden, 16 layers, 16 heads, FFN 4096.
- Training: 2000 steps by default.
- Evaluation: 50 closed-loop trials by default.
- Non-Flow losses have zero training weight; the optimization objective is the
  original Flow Matching loss only.

The script aborts if normalization changes, the merged config is wrong, any
frozen upstream tensor changes, or no canonical Flow tensor changes.

## Experiment 2: Small Flow S1 T9

Run only after the Full-Flow upper bound is normal:

```bash
TRAIN_GPUS=4,5,6,7 \
POLICY_GPU=4 \
EVAL_GPU=5 \
bash scripts/run_libero_goal_small_flow_s1_t9_v1.sh
```

Protocol:

- Task: `libero_goal`, T9 only.
- Initialization: VLM, VLM-to-LAM QFormer, LAM/LaWM, queries, and every other
  upstream tensor are loaded exactly from Formal Base.
- Flow initialization: fresh random initialization; no Base Flow tensor or
  Flow alias is loaded.
- Trainable parameters: the complete Small Flow only.
- Capacity: 512 hidden, 6 layers, 8 heads, FFN 2048.
- Inputs, masks, action chunking, time sampling, FM target, velocity decoder,
  inference integration, and evaluator are unchanged from the Full Flow.
- Training: 2000 steps by default.
- Evaluation: 50 closed-loop trials by default.

The script additionally checks that the final Small Flow has fewer parameters
and different tensor shapes than the Base Flow while every non-Flow tensor is
bitwise identical to Formal Base.

## Useful overrides

Both scripts accept environment overrides, for example:

```bash
MAX_TRAIN_STEPS=200 \
NUM_TRIALS=5 \
PER_DEVICE_BATCH_SIZE=16 \
TRAIN_GPUS=4,5 \
POLICY_GPU=4 \
EVAL_GPU=5 \
bash scripts/run_libero_goal_small_flow_s1_t9_v1.sh
```

Main variables:

| Variable | Default |
|---|---:|
| `MAX_TRAIN_STEPS` | 2000 |
| `NUM_WARMUP_STEPS` | 120 |
| `ACTION_LR` | 1e-4 |
| `PER_DEVICE_BATCH_SIZE` | 64 |
| `GRADIENT_ACCUMULATION_STEPS` | 1 |
| `NUM_TRIALS` | 50 |
| `SEED` | 2026 |
| `REPEATED_DIFFUSION_STEPS` | 2 |

Use `BASE_RUN=/absolute/path/to/formal_base_run` if automatic Base discovery
does not select the intended run.

## Recommended execution order

1. Run `python scripts/inspect_small_flow_s1.py --base-checkpoint <BASE_CKPT>`.
2. Run the strict Full-Flow experiment.
3. Confirm its T9 SR and inspect the `[full-flow-check]` line.
4. Run Small Flow S1 without changing the data, seed, FM protocol, or evaluator.
5. Compare `per_task_summary.csv` and the final training losses.

Suggested S1 gate:

- proceed to the latent proposal stage when T9 SR is at least 0.85, preferably
  near 0.90, and reasonably close to the Fresh Full-Flow upper bound;
- if S1 is substantially worse, first test a larger `expert_variant=custom`
  capacity. Do not add latent/world auxiliary losses to repair basic action
  learning.
