# Routing-V2: Confidence-aware Semantic + Dynamics Fusion

This is an **offline diagnostic**. It reads the existing passive Dynamics probe logs (`T*/dynamics_probe.jsonl`) and does not run LIBERO, load a policy, or use a GPU.

## Motivation

Existing results show:

- Semantic Top-1 is strong (~94%).
- Dynamics hard reranking can recover many Semantic errors, but also damages many already-correct Semantic decisions.
- Dynamics is beneficial mainly when Semantic confidence is very low.

We therefore test two safer fusion families inside a Semantic-confidence gate.

### Confidence gate

`C_sem = (e2-e1)/(e2+eps)` where lower means more ambiguous.

Primary diagnostic gate:

`C_sem < 0.2`

Outside the gate, Semantic Top-1 is always kept.

### Pairwise score normalization

For the Semantic Top-2 candidates `(a,b)`, separately normalize Semantic and Dynamics reconstruction errors:

`sem_share(k) = e_sem(k)/(e_sem(a)+e_sem(b)+eps)`

`dyn_share(k) = e_dyn(k)/(e_dyn(a)+e_dyn(b)+eps)`

Lower is better. This avoids directly adding two reconstruction losses with unrelated numeric scales.

### Fixed fusion

Inside the gate:

`S_k = (1-lambda)*sem_share(k) + lambda*dyn_share(k)`

### Confidence-adaptive fusion

Inside the gate:

`lambda_dyn(C) = lambda_max * ((delta-C)/delta)^gamma`

clipped to `[0, lambda_max]`.

Then:

`S_k = (1-lambda_dyn)*sem_share(k) + lambda_dyn*dyn_share(k)`

This makes Dynamics evidence weak near the gate boundary and strongest only when Semantic is extremely ambiguous.

## Run

```bash
cd /home/jincai_guo/tianqi/CVPR2027/LaWAM

DYN_ROOT=/home/jincai_guo/tianqi/CVPR2027/LaWAM/results/eval_runs/lawam_cl/libero_goal/routing_v2_dynamics_probe/20260901_102613

python scripts/analyze_routing_v2_confidence_fusion.py \
  --root "${DYN_ROOT}" \
  --primary-gate 0.20
```

No GPU is needed.

## Outputs

Under `${DYN_ROOT}/confidence_fusion/`:

- `REPORT.md`
- `fusion_summary.json`
- `fusion_sweep.csv`
- `selected_variants_summary.csv`
- `fusion_per_task.csv`
- `fusion_confidence_deciles.csv`
- `adaptive_primary_best_decisions.csv`
- `adaptive_primary_best_confusion.csv`

## Main interpretation

Prioritize:

1. `hybrid_accuracy`
2. `recovered`
3. `damaged`
4. `net_corrections = recovered - damaged`
5. per-task behavior, especially T8

A useful fusion should reduce damage substantially relative to hard Dynamics reranking while preserving a meaningful fraction of its Semantic-error recovery.

## Important protocol note

The script sweeps parameters using GT labels from the passive probe. Those **best** settings are diagnostic only. For a final paper result, select and freeze `delta`, `lambda_max` / `lambda`, and `gamma` on a held-out routing validation split, then evaluate once on the final test set.
