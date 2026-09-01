from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path

from deployment.model_server.routing_v1_policy import RoutingV1ExpertBankPolicy
from deployment.model_server.server_policy import build_policy_server_metadata
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def _parse_expert_specs(values: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--expert must be TASK=PATH, got {value!r}")
        task_raw, path = value.split("=", 1)
        task = int(task_raw.lstrip("tT"))
        if task not in {6, 7, 8, 9}:
            raise ValueError(f"Expert task must be 6..9, got {task}")
        if task in out:
            raise ValueError(f"Duplicate expert task {task}")
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        out[task] = str(resolved)
    return out


def main(args: argparse.Namespace) -> None:
    expert_ckpts = _parse_expert_specs(args.expert)
    candidates = [x.strip().lower() for x in args.candidates.split(",") if x.strip()]
    base_ckpt = Path(args.base_ckpt).expanduser().resolve()
    if not base_ckpt.is_file():
        raise FileNotFoundError(base_ckpt)

    policy = RoutingV1ExpertBankPolicy(
        base_checkpoint=base_ckpt,
        expert_checkpoints=expert_ckpts,
        candidate_labels=candidates,
        mode=args.mode,
        score_mode=args.score_mode,
        alpha=args.alpha,
        score_normalization=args.score_normalization,
        temporal_mode=args.temporal_mode,
        temporal_beta=args.temporal_beta,
        temporal_margin=args.temporal_margin,
        execution_mode=args.execution_mode,
        device="cuda",
        use_bf16=bool(args.use_bf16),
        routing_log_path=args.routing_log,
        context_label=args.context_label,
        require_gt_diagnostics=bool(args.require_gt_diagnostics),
        debug_requests=args.debug_requests,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )

    hostname = socket.gethostname()
    logging.info("Creating Routing-V1 server (host=%s, ip=%s)", hostname, socket.gethostbyname(hostname))
    metadata = build_policy_server_metadata(
        policy.template_policy,
        ckpt_path=policy.template_checkpoint,
        server_type="lawam_routing_v1",
        env="generic",
        supported_eval_envs=["libero"],
        extra_metadata={
            "routing_v1": True,
            "routing_mode": args.mode,
            "routing_score_mode": args.score_mode,
            "routing_alpha": float(args.alpha),
            "routing_score_normalization": args.score_normalization,
            "routing_temporal_mode": args.temporal_mode,
            "routing_temporal_beta": float(args.temporal_beta),
            "routing_temporal_margin": float(args.temporal_margin),
            "routing_execution_mode": args.execution_mode,
            "routing_candidates": candidates,
            "routing_context_label": args.context_label,
            "routing_require_gt_diagnostics": bool(args.require_gt_diagnostics),
            "routing_base_ckpt": str(base_ckpt),
        },
    )
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )
    logging.info("Routing-V1 websocket server ready on 0.0.0.0:%d", args.port)
    server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--expert", action="append", default=[], help="Repeated TASK=PATH, e.g. 6=/.../pytorch_model.pt")
    p.add_argument("--candidates", required=True, help="Comma-separated labels: base,t6,t7,t8,t9")
    p.add_argument("--mode", choices=["b1", "b2"], required=True)
    p.add_argument("--score_mode", choices=["latent", "world", "combined"], default="world")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument(
        "--score_normalization",
        choices=["none", "candidate_mean"],
        default="none",
        help="Optional per-chunk candidate-wise normalization before score fusion.",
    )
    p.add_argument(
        "--temporal_mode",
        choices=["none", "ema", "ema_hysteresis"],
        default="none",
        help="Per-episode temporal stabilization while still making one routing decision per action chunk.",
    )
    p.add_argument("--temporal_beta", type=float, default=0.0, help="EMA coefficient in [0,1).")
    p.add_argument("--temporal_margin", type=float, default=0.0, help="Switch margin for ema_hysteresis.")
    p.add_argument(
        "--execution_mode",
        choices=["route", "oracle_execute"],
        default="route",
        help=(
            "route executes the score-selected expert; oracle_execute computes scores task-agnostically "
            "but executes the GT expert after scoring, for clean-state routing diagnostics only."
        ),
    )
    p.add_argument("--routing_log", type=str, default=None)
    p.add_argument("--context_label", type=str, default=None, help="Stage/run label written to routing diagnostics.")
    p.add_argument("--require_gt_diagnostics", action="store_true", help="Require GT task metadata for evaluation checks; never used for routing selection.")
    p.add_argument("--debug_requests", type=int, default=5)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--port", type=int, default=10093)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--idle_timeout", type=int, default=1800)
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
