# Routing-V1: No-Task-ID Evaluation

This document describes the first task-agnostic routing evaluator for the latent-enabled B1/B2 experts.

## 1. Candidate score

The expert-side latent is always read from the **final denoising forward, final DiT hidden**, using the same trained pooling/projection head as Routing-V1 training.

For candidate expert `k`:

- latent distance: `d_z = 1 - cosine(z_k, z*_k)`
- world distance: token-wise `1 - cosine(h_k, h*_k)`, averaged over world tokens
- combined score: `S_k = alpha * d_z + (1-alpha) * d_h`

The candidate with the **lowest** score is selected, and the already-generated action chunk from that candidate is executed.

Available score modes:

- `latent`: `S=d_z`
- `world`: `S=d_h`
- `combined`: weighted score (default `alpha=0.5`)

All candidate experts receive the **same initial Flow noise tensor** for each action chunk.

## 2. B1 vs B2

### B1

VLM / latent-action query / QFormer / LaWM are shared and frozen. The evaluator computes shared `z` and `h` once per action chunk and only swaps the task-specific Flow/Action Expert tensors:

`shared z,h -> E_k -> a_k,z*_k -> LaWM(z*_k)=h*_k`

### B2

Each candidate's **Text-LoRA is activated** together with its Action Expert. Thus every candidate has its own online branch:

`VLM + TextLoRA_k -> z_k -> LaWM -> h_k`

and

`ActionExpert_k -> a_k,z*_k -> LaWM -> h*_k`

Routing therefore measures branch-wise semantic/motor/world self-consistency.

## 3. Two distinct CL protocols

These are separate closed-loop evaluations, not two statistics from one rollout.

### `cl_only`

Base is completely excluded from routing competition.

- CL1: candidates `{T6}`, evaluate `T6`
- CL2: candidates `{T6,T7}`, evaluate `T6,T7`
- CL3: candidates `{T6,T7,T8}`, evaluate `T6,T7,T8`
- CL4: candidates `{T6,T7,T8,T9}`, evaluate `T6,T7,T8,T9`

Produces a 4x4 lifecycle SR matrix and REGEN-style FWT/NBT/AUC.

### `base_inclusive`

- Base row: Base only, evaluate `T0..T5`
- CL1: `{Base,T6}`, evaluate `T0..T6`
- CL2: `{Base,T6,T7}`, evaluate `T0..T7`
- CL3: `{Base,T6,T7,T8}`, evaluate `T0..T8`
- CL4: `{Base,T6,T7,T8,T9}`, evaluate `T0..T9`

Produces a 5x10 SR matrix. Metrics use acquisition-stage-aware lifecycles because T0..T5 are jointly acquired at Base. The metrics file reports both `FWT_all_acquisition` and `FWT_CL_tasks` to make the convention explicit.

## 4. Task-ID leakage protection

LIBERO sends `routing_gt_task_id` only for diagnostics. `RoutingV1ExpertBankPolicy` removes all routing metadata **before** `build_infer_batch`. The GT task ID is used only after scores and the selected expert are computed, for routing accuracy/confusion logs.

The first debug requests print a `[RoutingV1][CHECK]` message confirming this stripping.

## 5. Startup/runtime checks

The routing server audits and logs:

- B1 cannot contain Text-LoRA task tensors.
- B2 task experts must contain nonzero Text-LoRA tensors.
- Base emulation has zero Text-LoRA and zero B1/B2 side adapters.
- candidate task-specific parameter categories and counts.
- B1 shared VLM/QFormer/LaWM is computed exactly once per request.
- B1 `z/h` are exactly identical across candidates (otherwise runtime error).
- B2 candidate `z/h` deltas are printed to verify Text-LoRA branch differences.
- same Flow noise tensor is used by all candidates in the chunk.
- action/z/h/z*/h* shapes and finite-value checks.
- per-candidate `d_z`, `d_h`, score and selected expert.

## 6. Raw routing diagnostics

Every routed action chunk writes `routing_chunks.jsonl`, including:

- CL stage/context
- episode/chunk index
- GT task ID (diagnostic only)
- expected expert
- selected expert
- per-candidate `d_z`, `d_h`, score
- routing correctness

The summarizer generates:

- `routing_summary.json`
- `routing_per_task.csv`
- `routing_per_episode.csv`
- `routing_confusion.csv`

It additionally reports expert switching rate and correct-expert score margin.

Protocol-level evaluation concatenates all CL-stage routing logs (with stage context preserved) and generates `routing_summary_all_stages/`.

## 7. Main commands

### Recommended first smoke

B1, CL-only, world-only, 2 trials/task:

```bash
bash scripts/eval_libero_goal_routing_v1_b1_world_smoke.sh
```

B2 distinct Text-LoRA path:

```bash
bash scripts/eval_libero_goal_routing_v1_b2_world_smoke.sh
```

Full wiring smoke for B1/B2 and CL-only/Base-inclusive (1 trial/task by default):

```bash
bash scripts/eval_libero_goal_routing_v1_protocol_smoke.sh
```

### Formal one configuration

```bash
MODE=b1 \
PROTOCOL=cl_only \
SCORE_MODE=world \
NUM_TRIALS=50 \
bash scripts/eval_libero_goal_routing_v1_bank.sh
```

B2 combined:

```bash
MODE=b2 \
PROTOCOL=base_inclusive \
SCORE_MODE=combined \
ALPHA=0.5 \
NUM_TRIALS=50 \
bash scripts/eval_libero_goal_routing_v1_bank.sh
```

## 8. Output root

Outputs are stored under:

`results/eval_runs/lawam_cl/libero_goal/routing_v1_no_taskid/<mode>/<protocol>/<score>_a<alpha>/<timestamp>/`

Important top-level files:

- `sr_matrix_cl_only.csv` or `sr_matrix_base_inclusive.csv`
- `metrics/metrics.json`
- `metrics/task_metrics.csv`
- `routing_chunks_all_stages.jsonl`
- `routing_summary_all_stages/routing_summary.json`
- `manifest.csv`
- `PROTOCOL.txt`
