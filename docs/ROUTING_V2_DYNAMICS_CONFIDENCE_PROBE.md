# Routing-V2 passive Dynamics verification + Semantic confidence

This stage never allows routing to control the robot. The provided task-ID Skill
Path remains the action-producing policy for every rollout.

## Stage 1 confidence

For Semantic AE reconstruction errors `e1 <= e2 <= ...`, define:

- absolute margin: `e2 - e1`
- relative margin: `(e2-e1)/(e1+eps)`
- **normalized gap (primary): `(e2-e1)/(e2+eps)`**

The normalized gap is bounded and scale-friendly. Larger means Semantic Top-1
is separated from Top-2; smaller means ambiguity. The intended gate is:

`if normalized_gap < delta: run Dynamics verification; else trust Semantic Top-1`.

`analyze_routing_v2_semantic_confidence.py` can be run immediately on the
existing Semantic probe without any GPU rollout. It reports how much of the
Semantic error set is captured for each threshold and how much computation is
activated. Thresholds using GT correctness are diagnostic only; final `delta`
must be frozen on a held-out routing validation split.

## Stage 2 passive Dynamics verification

Semantic Top-2 candidates are imagined independently with their task-specific
Routing-V2 upstream path:

`VLM Text-LoRA_k + query delta_k + QFormer-LoRA_k + LaWM-LoRA_k`.

The candidate gives `z_k` and predicted future `h_k`. Its own Dynamics AE scores
`(h_t, h_k, z_k)`. The lower-error candidate is the Dynamics winner. The action
expert is never swapped by this probe, and the GT upstream state is restored
before oracle action generation.

Primary diagnostics:

- Dynamics Top-2 accuracy
- RecoveryRate = P(Dynamics correct | Semantic Top-1 wrong)
- DamageRate = P(Dynamics wrong | Semantic Top-1 correct)
- confidence-decile Dynamics gain
- gated hybrid accuracy vs activation-rate sweep

The ideal pattern is that Dynamics gains are concentrated in low Semantic
confidence bins, so the final router can call Dynamics only when needed.
