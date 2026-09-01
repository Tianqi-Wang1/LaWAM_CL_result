# Routing-V2 closed-loop CL evaluation

This overlay turns the already validated passive Routing-V2 evidence into real skill selection.

## Protocol

CL-only incremental bank:

- CL1: bank `{T6}`, evaluate T6.
- CL2: bank `{T6,T7}`, evaluate T6,T7.
- CL3: bank `{T6,T7,T8}`, evaluate T6,T7,T8.
- CL4: bank `{T6,T7,T8,T9}`, evaluate T6,T7,T8,T9.

Each true LIBERO task is launched separately only so the server can log `gt_task_id` diagnostically. The GT ID never enters the examples, Semantic score, confidence gate, Dynamics score, fusion, or action selection.

Per action chunk:

1. Base VLM + Base queries produce the common Semantic anchor.
2. Candidate Semantic AEs rank the currently learned skills.
3. `C_sem=(e2-e1)/(e2+eps)`.
4. If `C_sem >= delta`, use Semantic Top-1.
5. If `C_sem < delta`, only Semantic Top-2 skills run their task-specific upstream WAM paths.
6. Pair-normalize Semantic and Dynamics reconstruction errors.
7. `lambda_dyn=lambda_max*((delta-C_sem)/delta)^gamma`.
8. Fuse `S=(1-lambda_dyn)*Sem + lambda_dyn*Dyn` and select the lower-energy task.
9. The selected **full Skill Path**, including its task-specific action expert, generates the real action chunk.

Current diagnostic hyperparameters are `delta=0.20`, `lambda_max=0.50`, `gamma=2.0`. They should eventually be frozen on a held-out routing validation split before a paper-final test.

## Smoke

```bash
NUM_TRIALS=1 \
EVAL_WORKERS=4 \
POLICY_GPU=4 \
EVAL_GPU=5 \
DEBUG_DECISIONS=8 \
bash scripts/eval_libero_goal_routing_v2_closed_loop_cl.sh
```

## Formal 50-trial evaluation

```bash
NUM_TRIALS=50 \
EVAL_WORKERS=8 \
POLICY_GPU=4 \
EVAL_GPU=5 \
SAVE_VIDEOS=False \
GATE_THRESHOLD=0.20 \
LAMBDA_MAX=0.50 \
FUSION_GAMMA=2.0 \
DEBUG_DECISIONS=4 \
bash scripts/eval_libero_goal_routing_v2_closed_loop_cl.sh
```

## Outputs

The run lives under:

`results/eval_runs/lawam_cl/libero_goal/routing_v2_closed_loop_cl/<stamp>/`

Important outputs:

- `sr_matrix_cl_only.csv`: 4x4 lower-triangular CL SR matrix.
- `metrics/metrics.json`: FWT/acquisition, final BWT, lifecycle BWT, REGEN-style NBT, AUC, final seen SR.
- `metrics/task_metrics.csv`: per-task lifecycle metrics.
- `routing_summary_all_stages.csv`: chunk routing accuracy and gate usage per CL stage.
- `CL*/routing_summary/`: per-stage routing summaries/confusions.
- `CL*/T*/routing_chunks.jsonl`: every online routing decision.

### Metric note

The existing project has historically called the mean diagonal/acquisition SR `FWT`. This overlay preserves that key for compatibility. Strict classical FWT would require evaluating each task *before* it is learned, which this CL-only protocol does not do. `BWT_final` is the standard final-minus-acquisition backward transfer over old CL tasks T6-T8; positive is improvement, negative is forgetting.
