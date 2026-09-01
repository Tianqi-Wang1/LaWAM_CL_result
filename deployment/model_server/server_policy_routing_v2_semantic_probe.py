#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import socket
from pathlib import Path
from typing import Dict, Sequence

import torch

from deployment.model_server.server_policy import (
    build_policy_server_metadata,
    load_policy_from_checkpoint,
)
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.latent_world.routing_v2.autoencoders import (
    SemanticTokenAutoencoder,
)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_memory_spec(spec: str) -> tuple[int, Path]:
    if "=" not in spec:
        raise ValueError(
            f"Invalid --semantic-memory {spec!r}; expected TASK_ID=/path/to/routing_memory.pt"
        )
    task_s, path_s = spec.split("=", 1)
    task_id = int(task_s)
    path = Path(path_s).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Semantic memory not found for T{task_id}: {path}")
    return task_id, path


def _load_semantic_ae(path: Path, *, device: torch.device) -> SemanticTokenAutoencoder:
    state = _torch_load(path)
    if not isinstance(state, dict):
        raise RuntimeError(f"Routing memory must be a state dict: {path}")

    prefix = "semantic."
    sem_state = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix) and torch.is_tensor(value)
    }
    if not sem_state:
        raise RuntimeError(f"No {prefix!r} tensors found in {path}")

    encoder_weight = sem_state.get("encoder.0.weight")
    if encoder_weight is None or encoder_weight.ndim != 2:
        raise RuntimeError(
            f"Cannot infer Semantic AE dimensions from {path}: missing encoder.0.weight"
        )
    bottleneck_dim = int(encoder_weight.shape[0])
    input_dim = int(encoder_weight.shape[1])
    ae = SemanticTokenAutoencoder(
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim,
    )
    result = ae.load_state_dict(sem_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Semantic AE load mismatch for {path}: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return ae.to(device=device, dtype=torch.float32).eval()


class SemanticBankProbePolicy:
    """Passive Routing-V2 Semantic AE probe.

    Routing scores are computed from Base-VLM + Base-query H_act and are never
    used to choose the action-producing skill.  The supplied checkpoint remains
    the oracle/task-ID skill for the entire rollout.
    """

    def __init__(
        self,
        *,
        policy,
        semantic_bank: Dict[int, SemanticTokenAutoencoder],
        gt_task_id: int,
        output_jsonl: Path,
        debug_decisions: int = 8,
    ) -> None:
        self.policy = policy
        self.backend = policy.policy_backend
        self.batch_builder = policy.policy_infer_batch_builder
        self.gt_task_id = int(gt_task_id)
        self.tasks = sorted(int(x) for x in semantic_bank)
        if self.gt_task_id not in self.tasks:
            raise ValueError(
                f"GT task T{self.gt_task_id} is not in Semantic AE bank {self.tasks}."
            )
        self.semantic_bank = semantic_bank
        self.output_jsonl = Path(output_jsonl).expanduser().resolve()
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_jsonl.open("a", encoding="utf-8", buffering=1)
        self._decision_id = 0
        self.debug_decisions = max(0, int(debug_decisions))

        # Hard checks that this is the exact common anchor defined for V2.
        anchor_fn = getattr(self.backend, "_run_routing_v2_base_anchor_hact", None)
        if anchor_fn is None:
            raise RuntimeError(
                "Checkpoint/backend does not expose _run_routing_v2_base_anchor_hact; "
                "install the Routing-V2 training code first."
            )
        logging.info(
            "[RoutingV2][SEMANTIC-PROBE] GT/action skill=T%d, bank=%s, "
            "routing is PASSIVE and cannot control actions.",
            self.gt_task_id,
            self.tasks,
        )
        logging.info(
            "[RoutingV2][SEMANTIC-PROBE] Anchor = Base VLM with task VLM-LoRA "
            "temporarily disabled + immutable Base act/flow queries."
        )
        logging.info(
            "[RoutingV2][SEMANTIC-PROBE] JSONL = %s",
            self.output_jsonl,
        )

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass

    @torch.inference_mode()
    def _score_semantic_bank(self, examples: Sequence[dict]) -> None:
        if len(examples) == 0:
            return
        batch = self.batch_builder.build_infer_batch(examples)
        anchor_hact = self.backend._run_routing_v2_base_anchor_hact(
            prepared_batch=batch
        ).detach()

        per_task_errors = []
        for task_id in self.tasks:
            out = self.semantic_bank[task_id](anchor_hact)
            err = out["per_sample_error"].detach().float()
            if err.ndim != 1 or int(err.shape[0]) != len(examples):
                raise RuntimeError(
                    f"T{task_id} Semantic AE returned invalid per-sample error "
                    f"shape={tuple(err.shape)}, batch={len(examples)}"
                )
            per_task_errors.append(err)

        errors = torch.stack(per_task_errors, dim=1)  # [B, K], lower is better.
        order = torch.argsort(errors, dim=1, descending=False)
        gt_col = self.tasks.index(self.gt_task_id)

        for sample_idx, example in enumerate(examples):
            ranked_cols = order[sample_idx].detach().cpu().tolist()
            ranked_tasks = [self.tasks[int(col)] for col in ranked_cols]
            gt_rank = int(ranked_tasks.index(self.gt_task_id) + 1)
            top1 = int(ranked_tasks[0])
            top2 = [int(x) for x in ranked_tasks[: min(2, len(ranked_tasks))]]
            score_dict = {
                str(task_id): float(errors[sample_idx, col].item())
                for col, task_id in enumerate(self.tasks)
            }
            row = {
                "decision_id": int(self._decision_id),
                "sample_index": int(sample_idx),
                "gt_task_id": int(self.gt_task_id),
                "instruction": str(example.get("lang", "")),
                "bank_tasks": self.tasks,
                "errors": score_dict,
                "ranked_tasks": ranked_tasks,
                "top1_task": top1,
                "top2_tasks": top2,
                "gt_rank": gt_rank,
                "top1_correct": bool(top1 == self.gt_task_id),
                "top2_correct": bool(self.gt_task_id in top2),
                "gt_error": float(errors[sample_idx, gt_col].item()),
            }
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            if self._decision_id < self.debug_decisions:
                score_text = ", ".join(
                    f"T{t}={score_dict[str(t)]:.6f}" for t in self.tasks
                )
                logging.info(
                    "[RoutingV2][SEMANTIC][decision=%d sample=%d GT=T%d] "
                    "%s -> rank=%s top1=T%d top2=%s gt_rank=%d",
                    self._decision_id,
                    sample_idx,
                    self.gt_task_id,
                    score_text,
                    ranked_tasks,
                    top1,
                    top2,
                    gt_rank,
                )
            self._decision_id += 1

    @torch.inference_mode()
    def predict_action(
        self,
        examples: Sequence[dict],
        return_intermediates: bool = False,
        **kwargs,
    ):
        # Critically, probing happens before action generation but its output is
        # discarded. The underlying oracle/task-ID skill checkpoint produces the
        # action exactly as in the existing task-ID evaluation.
        self._score_semantic_bank(examples)
        return self.policy.predict_action(
            examples=examples,
            return_intermediates=bool(return_intermediates),
            **kwargs,
        )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--port", type=int, default=10093)
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--idle_timeout", type=int, default=1800)
    ap.add_argument("--gt-task-id", type=int, required=True)
    ap.add_argument(
        "--semantic-memory",
        action="append",
        required=True,
        help="Repeat as TASK_ID=/path/to/routing_memory.pt",
    )
    ap.add_argument("--probe-output", required=True)
    ap.add_argument("--debug-decisions", type=int, default=8)
    return ap


def main(args) -> None:
    policy = load_policy_from_checkpoint(
        args.ckpt_path,
        use_bf16=bool(args.use_bf16),
        device="cuda",
    )
    device = next(policy.parameters()).device

    bank: Dict[int, SemanticTokenAutoencoder] = {}
    memory_paths = {}
    for spec in args.semantic_memory:
        task_id, path = _parse_memory_spec(spec)
        if task_id in bank:
            raise ValueError(f"Duplicate Semantic AE memory for T{task_id}")
        bank[task_id] = _load_semantic_ae(path, device=device)
        memory_paths[str(task_id)] = str(path)
        logging.info(
            "[RoutingV2][SEMANTIC-PROBE] loaded T%d Semantic AE from %s",
            task_id,
            path,
        )

    wrapper = SemanticBankProbePolicy(
        policy=policy,
        semantic_bank=bank,
        gt_task_id=int(args.gt_task_id),
        output_jsonl=Path(args.probe_output),
        debug_decisions=int(args.debug_decisions),
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    logging.info("Creating semantic-probe server (host=%s ip=%s)", hostname, local_ip)
    metadata = build_policy_server_metadata(
        policy,
        ckpt_path=args.ckpt_path,
        server_type="routing_v2_semantic_probe",
        env="generic",
        supported_eval_envs=["libero"],
        extra_metadata={
            "routing_v2_probe": "semantic_ae_bank",
            "routing_v2_probe_passive": True,
            "routing_v2_gt_task_id_diagnostics_only": int(args.gt_task_id),
            "routing_v2_semantic_bank": sorted(bank),
            "routing_v2_memory_paths": memory_paths,
        },
    )
    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=int(args.port),
        idle_timeout=int(args.idle_timeout),
        metadata=metadata,
    )
    logging.info("Semantic-probe server ready on 0.0.0.0:%d", int(args.port))
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
