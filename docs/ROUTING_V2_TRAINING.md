# Routing-V2 training: continual WAM skill paths + two routing memories

## Scope of this package

This package implements **training only** for Routing-V2. Evaluation remains an oracle/task-ID
sanity check. No automatic routing code is introduced here.

The wall-clock chain is:

```
Base(T0-T5, 10k)
  -> T6 skill + T6 memories
  -> T7 skill + T7 memories
  -> T8 skill + T8 memories
  -> T9 skill + T9 memories
  -> evaluate Base/T6/T7/T8/T9 with provided task IDs
```

T6-T9 are executed serially but are **not checkpoint-chained**. Every task skill is initialized
fresh from the same V2 Base. This is required by the expandable-skill interpretation: old paths
stay frozen and a new task owns a new delta path.

## Clean V2 Base

V2 Base is retrained from the released LaWAM pretrain using the original LaWAM objective only:

```
L = L_action + 0.1 L_distill + 0.1 L_wm
```

Routing-V1 `z*` and its auxiliary losses are disabled. Base VLM, Base act/flow queries,
VLM-to-LAM QFormer, full LaWM decoder and full Action Expert are trainable. Stage-1 IDM/DINO
remain the existing frozen teacher/features.

## Task-specific V2 skill path

For task k, starting fresh from the same V2 Base:

- Action side: existing B2 (`last4 dense + conditioning-r128 + nonlinear adapters in blocks 0-11 r128`).
- VLM: text-only LoRA rank 32 (same B2 design).
- Queries: Base `act_query` and `flow_action_query` are kept frozen. Two zero-initialized task residuals
  are added: `q_k = q_base + delta_q_k`.
- VLM-to-LAM/QFormer: recursive **explicit-Linear** LoRA rank 32.
- LaWM decoder: recursive **explicit-Linear** LoRA rank 32.
- Routing-V1 expert `z*` head: disabled.

`nn.MultiheadAttention` stores packed Q/K/V matrices as raw parameters; the first V2 version
intentionally leaves those frozen and LoRA-adapts its explicit `out_proj`, plus FFNs, AdaLN
modulation linears, input/output projections, etc. Runtime logs print the exact target list and
parameter count instead of assuming an architecture count.

Knowledge insulation is preserved because `detach_future_feature=true`: action-flow loss cannot
rewrite LaWM through the predicted future branch. LaWM-LoRA is driven by the original world loss;
QFormer/latent path is aligned by the original distillation/world objectives.

## Stable semantic anchor

The future semantic router must compare every task in one coordinate system. During memory
training, semantic features are therefore extracted with:

```
Base VLM weights (all VLM LoRA residual scaling temporarily set to zero)
+ Base act_query
+ Base flow_action_query
-> H_act_base [B, Q, D_vlm]
```

Task query deltas are deliberately bypassed. QFormer/LaWM task adapters occur downstream and
cannot change `H_act_base`.

## Semantic AE (coarse retrieval memory)

File: `starVLA/model/framework/latent_world/routing_v2/autoencoders.py`

```
H_act_base [B,Q,D]
 -> feature LayerNorm (functional)
 -> Linear(D, 128)
 -> GELU
 -> Linear(128, D)
 -> token-wise reconstruction error
```

No mean pooling is used before the AE; the eight LaWAM latent-action-query tokens retain their
query structure. At D=2048 the AE has about 0.526M parameters.

## Spatial Dynamics AE (future verification memory)

The task-specific skill path is frozen first. On task-k data we construct both:

```
GT:        (h_t, z_GT, h_GT)
predicted: (h_t, z_k,  h_hat_k)
```

For every DINO spatial token p the AE sees:

```
[h_t,p ; (h_future,p - h_t,p) ; z]
```

The current state is conditioning context. The AE reconstructs only normalized `Delta h` and `z`,
so shared current-state reconstruction cannot dominate the routing score.

Architecture (defaults):

```
[B,256, 2*D_vision + D_z]
 -> Linear(...,192) + learned 2D-token position table
 -> TransformerEncoder x2 (hidden 192, 6 heads, FFN 768)
 -> spatial Linear(192,D_vision)  reconstruct Delta h
 -> mean-pool + Linear(192,D_z)   reconstruct z
```

At D_vision=768, D_z=32, N=256 this is about 1.39M parameters.

Memory training uses a 50/50 mixture of GT and the *correct skill's predicted* transitions, reducing
the train/test gap that would arise from learning only ideal GT transitions.

## Memory phase isolation

The skill checkpoint is loaded, all VLM/query/QFormer/LaWM/Action parameters are frozen, and only:

```
policy_backend.routing_v2_memory.*
```

can receive gradients. The full policy is used only to generate detached training targets/features.
After the memory phase, `scripts/extract_routing_v2_memory.py` saves only the small AE weights and
the temporary full policy checkpoint is deleted by default to save storage.

## Commands

### Full wiring smoke

All training happens first; all task-ID evaluation happens last:

```bash
BASE_STEPS=20 SKILL_STEPS=20 MEMORY_STEPS=20 NUM_TRIALS=1 \
  bash scripts/run_libero_goal_routing_v2_all.sh
```

or simply:

```bash
bash scripts/run_libero_goal_routing_v2_smoke.sh
```

### Formal first round

```bash
BASE_STEPS=10000 SKILL_STEPS=2000 MEMORY_STEPS=1000 NUM_TRIALS=50 \
  bash scripts/run_libero_goal_routing_v2_all.sh
```

This is equivalent to the defaults:

```bash
bash scripts/run_libero_goal_routing_v2_all.sh
```

### Train-only components

```bash
MAX_TRAIN_STEPS=10000 bash scripts/run_libero_goal_routing_v2_base.sh
TASK_ID=6 MAX_TRAIN_STEPS=2000 bash scripts/run_libero_goal_routing_v2_skill_task.sh
TASK_ID=6 MAX_TRAIN_STEPS=1000 bash scripts/run_libero_goal_routing_v2_memory_task.sh
```

### Task-ID evaluation only

```bash
NUM_TRIALS=50 bash scripts/eval_libero_goal_routing_v2_taskid_all.sh
```

## First logs to audit

Skill startup must print:

- `[RoutingV2][SKILL][PARAMS] ...`
- `[RoutingV2][SKILL][LORA] qformer_targets=...`
- `[RoutingV2][SKILL][LORA] lawm_targets=...`
- two query-delta tensors, both zero-initialized before optimization.

Memory startup must print:

- `[RoutingV2][MEMORY] instantiated: ...`
- `[RoutingV2][MEMORY][PARAMS] ... all skill params FROZEN`

Memory train logs expose:

- `train_loss_routing_v2_semantic`
- `train_loss_routing_v2_dynamics`
- `train_loss_routing_v2_dynamics_gt`
- `train_loss_routing_v2_dynamics_pred`
- GT/predicted `Delta h` and `z` reconstruction diagnostics.

No Routing-V1 `loss_expert_latent/loss_expert_world` should appear in V2 policy training.
