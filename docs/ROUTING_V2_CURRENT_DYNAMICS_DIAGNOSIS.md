# Routing-V2 current Dynamics diagnosis

This overlay adds **offline/passive analysis only**. It does not change training, policy inference, routing, checkpoints, or evaluation rollout code.

## Purpose

Use the existing `T6..T9/dynamics_probe.jsonl` logs to answer:

1. Is Dynamics useful on every task, or only selected tasks?
2. Is its gain concentrated in low Semantic-confidence chunks?
3. At what low-confidence prefix does Dynamics stop producing positive net corrections?
4. Does the best confidence gate have a broad useful region, or is it a narrow test-set optimum?

The script reports `Recovery`, `Damage`, and `Net Corrections`; do not judge the Dynamics verifier only by its overall Top-2 accuracy.

## Install

Copy `scripts/diagnose_routing_v2_dynamics.py` to the repository `scripts/` directory.

## Run on the current passive probe

```bash
cd /home/jincai_guo/tianqi/CVPR2027/LaWAM

DYN_ROOT=/home/jincai_guo/tianqi/CVPR2027/LaWAM/results/eval_runs/lawam_cl/libero_goal/routing_v2_dynamics_probe/20260901_102613

python scripts/diagnose_routing_v2_dynamics.py \
  --root "${DYN_ROOT}"
```

If `latest_dynamics_probe_run.txt` exists, use:

```bash
DYN_ROOT=$(cat /home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2/latest_dynamics_probe_run.txt)
python scripts/diagnose_routing_v2_dynamics.py --root "${DYN_ROOT}"
```

## Outputs

Under `${DYN_ROOT}/dynamics_diagnosis/`:

- `REPORT.md`: compact human-readable diagnosis.
- `dynamics_task_diagnosis.csv`: per-task Semantic/Dynamics/Recovery/Damage.
- `dynamics_confidence_deciles.csv`: equal-count confidence bins; bin 0 is most ambiguous.
- `dynamics_low_confidence_prefix.csv`: cumulative gating at 1%, 2%, ..., 100% lowest confidence.
- `dynamics_threshold_sweep_detailed.csv`: detailed confidence-threshold sweep.
- `dynamics_confusion.csv`: Dynamics winner confusion matrix.
- `dynamics_diagnosis.json`: machine-readable headline summary.

## What to look for

The desired pattern for confidence-gated WAM verification is:

- lowest-confidence bins: `Dynamics Accuracy > Semantic Accuracy`, positive `Net Corrections`;
- medium/high-confidence bins: Semantic becomes stronger and Dynamics becomes harmful;
- the cumulative low-confidence prefix reaches a maximum hybrid accuracy at a small activation rate.

If the result has this shape, the next ablation should compare `[h, Δh, z]` against `[h, Δh]` while keeping all Skill checkpoints fixed.

The reported best threshold is diagnostic only because it uses passive-test GT labels. Select/freeze the final threshold on a held-out routing validation split.
