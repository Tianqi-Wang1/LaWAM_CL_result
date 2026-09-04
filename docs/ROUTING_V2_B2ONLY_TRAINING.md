# Routing-V2 simplified B2-only training protocol

## 1. Relation to the original LaWAM training flow

LaWAM has two pretraining stages before LIBERO post-training:

1. **Stage 1 - latent action/world model pretraining.** Frozen visual features from the current
   and future frames are passed to the inverse-dynamics encoder to infer the teacher latent action
   `z_GT`. The decoder predicts the future visual feature and is retained as LaWM. The IDM encoder
   is later used only as a frozen teacher.
2. **Stage 2 - LaWAM policy pretraining.** The VLM policy prior predicts `z_hat` from the current
   observation and instruction. LaWM expands `z_hat` into a predicted future visual feature, and
   the Alternate-DiT action expert generates the action chunk. Training combines action flow
   matching, latent-action distillation, and future-feature supervision. Knowledge insulation
   prevents action loss from overwriting LaWM through the predicted-future branch.

This experiment starts from the released Stage-2 LaWAM pretrain checkpoint. It does **not** rerun
Stage 1 or Stage 2 pretraining. It first performs a clean LIBERO Goal Base post-training on T0-T5.

## 2. Experiment definition

### Base

The Base checkpoint is trained on T0-T5 for 10,000 optimizer steps with the original LaWAM
objective:

```text
L = L_action + 0.1 L_distill + 0.1 L_world
```

VLM, Base queries, QFormer, LaWM, and the full action expert are trainable. Stage-1 DINO/IDM
remain frozen. Routing memories and all Routing-V1 `z*` losses are disabled.

Only `final_model/pytorch_model.pt` is retained for the large Base policy.

### Four independent CL skills

T6, T7, T8, and T9 are each trained for 10,000 optimizer steps. They are run serially for GPU
convenience, but every task starts independently from the same Base checkpoint:

```text
Base -> T6
Base -> T7
Base -> T8
Base -> T9
```

They are not checkpoint-chained. This preserves one isolated expert per new task.

The only task-specific trainable parameters are:

```text
VLM Text-LoRA rank 32
+ Action-B2
    - original DiT blocks 12-15: dense fine-tuning
    - conditioning adapters: bottleneck 128
    - nonlinear attention/FFN adapters in DiT blocks 0-11: bottleneck 128
```

The following modules stay shared and frozen:

```text
act_query
flow_action_query
VLM-to-LAM QFormer
LaWM
```

There are no query residuals, QFormer-LoRA, LaWM-LoRA, or Routing-V1 `z*` head. The expected
task-specific parameter count is 90,583,040 per task, approximately 3.55% of a 2.555B LaWAM.

Only `final_model/pytorch_model.pt` is retained for each large task policy. A strict post-training
audit rejects any unexpected upstream change or adapter.

## 3. Routing-memory training

After all task skills are complete, each task trains one Semantic AE and one Dynamics AE jointly
for a single continuous 5,000-step optimization trajectory. Compact snapshots are extracted at
1,000, 2,000, and 5,000 steps. Training three independent AE runs is deliberately avoided so the
three checkpoints are directly comparable.

### Semantic AE input

```text
observation + instruction
  -> shared Base VLM (task Text-LoRA temporarily disabled)
  -> shared Base act/flow queries
  -> H_act_base
  -> task Semantic AE
```

This retains a common semantic coordinate system across all tasks.

### Dynamics AE input

```text
observation + instruction                    observation frames
  -> task VLM Text-LoRA                        -> frozen DINO/LAM encoder
  -> shared frozen queries                     -> shared h_t
  -> shared frozen QFormer                           |
  -> task-conditioned z_k                            |
             |                                       |
             +--------> shared frozen Base-WM <------+
                              -> h_hat_k
                              -> [h_t, h_hat_k - h_t]
                              -> task Dynamics AE
```

The verifier uses `input_mode=hdh`: direct `z` is not reconstructed or scored. The Dynamics loss
keeps the existing 50/50 mixture of real/teacher transitions and correct-skill predicted
transitions. `h_t` itself is a shared DINO/LAM visual feature; task Text-LoRA affects the verifier
through `z_k`, and therefore through `h_hat_k` and `Delta h_k`. Both AEs are task-local memories;
no task-pair relationship is used.

During this phase, every policy/skill parameter is frozen. After the three compact snapshots are
extracted, all temporary full-policy memory checkpoints are deleted.

Output layout per task:

```text
task6/routing_memory/
  step_1000/
    routing_memory.pt
    semantic_ae.pt
    dynamics_ae.pt
    metadata.json
    memory_train_config.yaml
  step_2000/
    ...
  step_5000/
    ...
  routing_memory.pt   # compatibility copy of step 5000
  semantic_ae.pt      # compatibility copy of step 5000
  dynamics_ae.pt      # compatibility copy of step 5000
  manifest.json
  latest_step.txt
```

## 4. Files to upload

Copy these files to the same relative paths in the server repository:

```text
starVLA/model/framework/latent_world/runtime/freeze_policy.py
starVLA/config/training/train_libero.yaml
scripts/audit_routing_v2_b2only_skill.py
scripts/extract_routing_v2_memory_snapshots.py
scripts/run_libero_goal_routing_v2_b2only_skill_task.sh
scripts/run_libero_goal_routing_v2_b2only_memory_task.sh
scripts/run_libero_goal_routing_v2_b2only_all.sh
scripts/eval_libero_goal_routing_v2_taskid_all.sh
docs/ROUTING_V2_B2ONLY_TRAINING.md
```

The existing Base script below is reused and does not need replacement if the server checkout
already matches this repository:

```text
scripts/run_libero_goal_routing_v2_base.sh
```

## 5. Recommended commands

Run from the LaWAM repository root.

### Static syntax check after upload

```bash
python -m py_compile \
  starVLA/model/framework/latent_world/runtime/freeze_policy.py \
  scripts/audit_routing_v2_b2only_skill.py \
  scripts/extract_routing_v2_memory_snapshots.py

bash -n scripts/run_libero_goal_routing_v2_b2only_skill_task.sh
bash -n scripts/run_libero_goal_routing_v2_b2only_memory_task.sh
bash -n scripts/run_libero_goal_routing_v2_b2only_all.sh
```

### Full formal experiment in one command

```bash
nohup env \
  TRAIN_GPUS=4,5,6,7 \
  POLICY_GPU=4 \
  EVAL_GPU=5 \
  BASE_STEPS=10000 \
  SKILL_STEPS=10000 \
  MEMORY_STEPS=5000 \
  MEMORY_SNAPSHOT_STEPS=1000,2000,5000 \
  NUM_TRIALS=50 \
  bash scripts/run_libero_goal_routing_v2_b2only_all.sh \
  > routing_v2_b2only_formal_launcher.log 2>&1 &
```

The default output root is:

```text
/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2_b2only
```

### Train Base and four skills first, without AEs

```bash
TRAIN_GPUS=4,5,6,7 \
RUN_TASKID_EVAL=true \
SKIP_MEMORIES=true \
bash scripts/run_libero_goal_routing_v2_b2only_all.sh
```

### Resume with the AEs after checking task-ID performance

```bash
TRAIN_GPUS=4,5,6,7 \
SKIP_BASE=true \
SKIP_SKILLS=true \
RUN_TASKID_EVAL=false \
MEMORY_STEPS=5000 \
MEMORY_SNAPSHOT_STEPS=1000,2000,5000 \
bash scripts/run_libero_goal_routing_v2_b2only_all.sh
```

### Run one skill or one task's AEs

```bash
TASK_ID=6 MAX_TRAIN_STEPS=10000 \
  bash scripts/run_libero_goal_routing_v2_b2only_skill_task.sh

TASK_ID=6 MAX_TRAIN_STEPS=5000 MEMORY_SNAPSHOT_STEPS=1000,2000,5000 \
  bash scripts/run_libero_goal_routing_v2_b2only_memory_task.sh
```

### Short wiring smoke

The strict parameter audit expects the adapters to have received updates, so use at least several
optimizer steps:

```bash
V2_ROOT=/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2_b2only_smoke \
BASE_STEPS=20 \
BASE_WARMUP_STEPS=2 \
SKILL_STEPS=20 \
SKILL_WARMUP_STEPS=2 \
MEMORY_STEPS=20 \
MEMORY_WARMUP_STEPS=2 \
MEMORY_SNAPSHOT_STEPS=10,20 \
MEMORY_SAVE_INTERVAL=10 \
NUM_TRIALS=1 \
bash scripts/run_libero_goal_routing_v2_b2only_all.sh
```

## 6. First log lines to check

Skill training must report the validated B2 isolation line:

```text
[V8-PARTIAL-DENSE+COND+NONLINEAR] ... with_text_lora=True
```

The post-training audit must finish with:

```text
[OK] Simplified Routing-V2 B2-only skill audit passed
```

The memory phase must report:

```text
[RoutingV2][MEMORY][PARAMS] ... skill_path=b2_only ...
skill_struct_qformer_lora=0
skill_struct_lawm_lora=0
```

and extraction must report successful outputs for all requested steps.
