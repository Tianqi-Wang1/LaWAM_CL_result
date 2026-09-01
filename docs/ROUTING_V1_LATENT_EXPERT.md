# Routing-V1: Expert Latent Prediction for LaWAM Continual Learning

## Goal

## Formal first-round training budget

The current protocol intentionally uses different optimization budgets:

- **Base T0-T5: 10,000 optimizer steps**, 600 warmup steps, global batch 256. The retained VLM, LA queries / VLM-to-LAM QFormer, LaWM decoder, full Action Expert, and the new z* head are trainable. The Stage-1 IDM teacher and DINO feature extractor remain frozen.
- **Fresh B1/B2 task experts T6-T9: 2,000 optimizer steps each**, 120 warmup steps. Every task starts from the same new 10K latent-enabled Base. Shared VLM, LA/action queries, VLM-to-LAM QFormer, and LaWM are frozen according to B1/B2; the task-specific Action Expert components and z* head remain trainable (B2 additionally trains Text-LoRA-r32).

Use `scripts/run_libero_goal_routing_v1_base_10k.sh` for the Base and `scripts/run_libero_goal_routing_v1_all_base10k_expert2k.sh` for the complete first-round pipeline.

This first routing-stage experiment **does not change inference routing**. Evaluation still supplies the correct task ID. The goal is only to verify that the B1/B2 task expert can learn an auxiliary latent action `z*` without damaging action success rate.

The LaWAM teacher/online quantities are:

- `z_GT`: Stage-1 IDM latent action inferred from the real transition `(h_t, h_GT)`.
- `z_hat`: existing VLM/LA-query/QFormer policy-prior prediction.
- `h_GT`: real future DINO feature.
- `h_hat = LaWM(h_t, z_hat)`: existing online LaWM subgoal.
- `z*`: new Action-Expert latent prediction from the final DiT action-token hidden states.
- `h* = LaWM(h_t, z*)`: world-model consequence of the expert latent.

Routing-V1 adds

```
L_z* = MSE(z*, z_GT)
L_h* = MSE(LaWM(h_t, z*), h_GT)
```

with defaults `lambda_z = lambda_h = 0.1`, matching the scale of the original LaWAM latent/world auxiliary terms.

## Expert latent head

The head is attached to the same final DiT representation used by the action decoder:

```
final DiT output
    -> valid action-token mask-mean pooling
    -> LayerNorm
    -> Linear(hidden_dim -> head_hidden_dim)
    -> GELU
    -> Linear(head_hidden_dim -> LAM code_dim)
    -> z*
```

Default head hidden dimension is 1024. The output dimension is automatically aligned to `lam.code_dim` at model construction time.

Important V1 limitation: training DiT hidden states are conditioned on noisy flow action tokens, so `z*` may depend on flow time/noise. This is intentional for the first structural validation. Routing-time deterministic extraction/noise-stability will be studied after the single-task performance check.

## Freeze protocols

### New Base (T0-T5)

Start from the released LaWAM pretrain. Train for 2k steps in the quick round.

Trainable:

- retained Qwen3-VL backbone (first 16 layers used by LaWAM), including vision/text/embedding/merger;
- latent-action query and flow-action query;
- VLM-to-LAM QFormer/query aggregation;
- full LaWM decoder;
- full Action Expert;
- new expert latent head.

Frozen by the existing LaWAM implementation:

- Stage-1 IDM teacher encoder;
- DINO feature extractor.

### B1 task expert (fresh T6/T7/T8/T9 from the new Base)

Trainable:

- Dense DiT layers 12-15;
- conditioning adapters (`enc_vlm` + all AdaLN), bottleneck 128;
- nonlinear Attention/FFN adapters in DiT layers 0-11, bottleneck 128;
- expert latent head.

Frozen:

- shared VLM;
- latent/action queries;
- VLM-to-LAM QFormer;
- LaWM;
- all other original Base parameters.

### B2 task expert

Same as B1 plus VLM text LoRA rank 32 / alpha 32. Original shared VLM weights remain frozen; only LoRA tensors are task-specific/trainable.

## Logged diagnostics

In addition to the two new losses, training/validation logs expose:

- `diag_zstar_to_zhat = MSE(z*, z_hat)`
- `diag_zhat_to_zgt = MSE(z_hat, z_GT)`
- `diag_hstar_to_hhat = MSE(h*, h_hat)`

Together with `loss_expert_latent` and `loss_expert_world`, these are the first measurements for deciding later whether routing supervision should be GT-anchored, online-aligned, or dual-anchor.

## Quick 2k commands

### 1. New Base only

```bash
bash scripts/run_libero_goal_routing_v1_base_2k.sh
```

For a quick train-only smoke without 50-trial evaluation:

```bash
DO_EVAL=false MAX_TRAIN_STEPS=2000 \
  bash scripts/run_libero_goal_routing_v1_base_2k.sh
```

### 2. B1 on one task first (recommended T9 smoke)

```bash
TASKS="9" bash scripts/run_libero_goal_routing_v1_b1_2k.sh
```

### 3. B2 on one task

```bash
TASKS="9" bash scripts/run_libero_goal_routing_v1_b2_2k.sh
```

### 4. All fresh T6-T9 B1

```bash
bash scripts/run_libero_goal_routing_v1_b1_2k.sh
```

### 5. All fresh T6-T9 B2

```bash
bash scripts/run_libero_goal_routing_v1_b2_2k.sh
```

### 6. Base + B1 + B2 end-to-end

```bash
bash scripts/run_libero_goal_routing_v1_all_2k.sh
```

Each T6-T9 run is **fresh from the same new Routing-V1 Base**, not sequential checkpoint chaining. Evaluation still uses the provided task ID and current standard LIBERO evaluator; no automatic router is enabled.

## Useful overrides

```bash
# Train one task without closed-loop eval
DO_EVAL=false MODE=b1 TASK_ID=9 \
  bash scripts/run_libero_goal_routing_v1_seq_single_variant_2k.sh

# Change auxiliary weights
LATENT_WEIGHT=0.1 WORLD_WEIGHT=0.1 TASKS="9" \
  bash scripts/run_libero_goal_routing_v1_b1_2k.sh

# Later 10k confirmation (same script, no code change)
MAX_TRAIN_STEPS=10000 NUM_WARMUP_STEPS=600 TASKS="6 7 8 9" \
  bash scripts/run_libero_goal_routing_v1_b1_2k.sh
```

For a clean 10k result directory, it is better to set a separate `ROUTING_ROOT` rather than mixing 2k/10k runs.
