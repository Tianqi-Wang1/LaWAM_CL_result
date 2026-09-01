# Routing-V1 B2 Temporal Stabilization

This update keeps **one routing decision per action chunk** and focuses on B2.
It does not lock an expert for an episode. Instead, it stabilizes the B2 branch-wise
self-consistency score across successive chunks of the same episode.

## Instantaneous B2 score

For candidate expert k at chunk t:

- `dz[k] = 1 - cos(z_k, z*_k)`
- `dh[k] = 1 - cos(h_k, h*_k)`

With `SCORE_NORMALIZATION=candidate_mean`:

- `dzN[k] = dz[k] / mean_j dz[j]`
- `dhN[k] = dh[k] / mean_j dh[j]`

The recommended first score is:

`S[k,t] = 0.8 * dzN[k,t] + 0.2 * dhN[k,t]`.

## Temporal stabilization

EMA mode:

`Sema[k,t] = beta * Sema[k,t-1] + (1-beta) * S[k,t]`.

A fresh `argmin_k Sema[k,t]` is still computed at every action chunk.
The state is keyed only by `routing_episode_id`, never by task ID. If temporal
routing is enabled and no episode ID is supplied, evaluation aborts instead of
risking state leakage across episodes.

Optional EMA+hysteresis mode switches from the previous expert to a new proposal
only when:

`Sema[current] - Sema[new] >= margin`.

This suppresses score jitter but still permits online switching.

## New diagnostics

Each chunk log now records:

- raw `dz`, `dh`
- normalized/used `dz_used`, `dh_used`
- instantaneous `score`
- `temporal_score`
- `instant_selected_expert`
- final `selected_expert`
- previous/proposed expert and hysteresis blocking

The routing summarizer additionally reports instantaneous vs final routing accuracy
and instantaneous vs final switch rate.

## Recommended first experiments

1. Offline sweep on the existing B2 oracle-state probe log:

```bash
python scripts/analyze_routing_v1_temporal_offline.py \
  --log /path/to/B2/oracle_state_probe/.../routing_chunks.jsonl \
  --output-dir /tmp/b2_temporal_sweep \
  --alpha 0.8 \
  --normalization candidate_mean
```

2. CL4 closed-loop smoke (1 trial/task by default):

```bash
bash scripts/eval_libero_goal_routing_v1_b2_temporal_smoke.sh
```

It compares:

- normalized 0.8/0.2 fusion, no temporal memory
- EMA beta=0.3
- EMA beta=0.3 + hysteresis margin=0.08

3. After selecting a temporal policy, run the full CL-only lifecycle:

```bash
NUM_TRIALS=10 \
TEMPORAL_MODE=ema_hysteresis \
TEMPORAL_BETA=0.3 \
TEMPORAL_MARGIN=0.08 \
bash scripts/eval_libero_goal_routing_v1_b2_temporal_clonly.sh
```

For the formal run, change `NUM_TRIALS=50` and usually `EVAL_WORKERS=16`.
