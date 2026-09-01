#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import socket
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from deployment.model_server.server_policy import (
    build_policy_server_metadata,
    load_policy_from_checkpoint,
)
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.latent_world.routing_v2.autoencoders import (
    SemanticTokenAutoencoder,
    SpatialDynamicsAutoencoder,
)


_DENSE_BLOCK_RE = re.compile(r"^policy_backend\.flow\.DiT\.transformer_blocks\.(\d+)\.")


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _unwrap_state(obj):
    if not isinstance(obj, dict):
        raise RuntimeError("Checkpoint object is not a state dict")
    for wrapper in ("state_dict", "model", "module"):
        inner = obj.get(wrapper)
        if isinstance(inner, dict) and inner and any(torch.is_tensor(v) for v in inner.values()):
            return inner
    return obj


def _parse_spec(spec: str, label: str) -> tuple[int, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --{label} {spec!r}; expected TASK_ID=/path")
    task_s, path_s = spec.split("=", 1)
    task_id = int(task_s)
    path = Path(path_s).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} for T{task_id}: {path}")
    return task_id, path


def _load_memory(path: Path, *, device: torch.device, dynamics_heads: int, z_weight: float):
    state = _unwrap_state(_torch_load(path))
    sem_state = {
        k[len("semantic."):]: v
        for k, v in state.items()
        if k.startswith("semantic.") and torch.is_tensor(v)
    }
    dyn_state = {
        k[len("dynamics."):]: v
        for k, v in state.items()
        if k.startswith("dynamics.") and torch.is_tensor(v)
    }
    if not sem_state or not dyn_state:
        raise RuntimeError(f"Missing semantic/dynamics tensors in {path}")

    ew = sem_state["encoder.0.weight"]
    sem = SemanticTokenAutoencoder(
        input_dim=int(ew.shape[1]),
        bottleneck_dim=int(ew.shape[0]),
    )
    sem.load_state_dict(sem_state, strict=True)

    input_w = dyn_state["input_proj.weight"]
    pos = dyn_state["pos_embed"]
    delta_w = dyn_state["delta_decoder.weight"]
    z_w = dyn_state["z_decoder.1.weight"]
    hidden = int(input_w.shape[0])
    vision = int(delta_w.shape[0])
    latent = int(z_w.shape[0])
    ntokens = int(pos.shape[1])
    if int(input_w.shape[1]) != 2 * vision + latent:
        raise RuntimeError(f"Cannot infer Dynamics AE geometry from {path}")
    layer_ids = []
    for key in dyn_state:
        if key.startswith("encoder.layers."):
            try:
                layer_ids.append(int(key.split(".")[2]))
            except Exception:
                pass
    nlayers = max(layer_ids) + 1 if layer_ids else 0
    ffn = int(dyn_state["encoder.layers.0.linear1.weight"].shape[0])
    dyn = SpatialDynamicsAutoencoder(
        vision_dim=vision,
        latent_dim=latent,
        num_tokens=ntokens,
        hidden_dim=hidden,
        num_layers=nlayers,
        num_heads=int(dynamics_heads),
        ffn_dim=ffn,
        z_loss_weight=float(z_weight),
    )
    dyn.load_state_dict(dyn_state, strict=True)
    return (
        sem.to(device=device, dtype=torch.float32).eval(),
        dyn.to(device=device, dtype=torch.float32).eval(),
    )


def _is_lora(key: str, prefix: str) -> bool:
    return key.startswith(prefix) and (key.endswith(".lora_A") or key.endswith(".lora_B"))


def _is_dense_last4(key: str) -> bool:
    m = _DENSE_BLOCK_RE.match(key)
    return m is not None and int(m.group(1)) in {12, 13, 14, 15}


def _is_flow_side_adapter(key: str) -> bool:
    return key.startswith("policy_backend.flow.") and (
        ".conditioning_adapter." in key
        or ".attn_nonlinear_adapter." in key
        or ".ffn_nonlinear_adapter." in key
    )


def _is_task_specific_param(key: str) -> bool:
    if key in {
        "policy_backend.routing_v2_act_query_delta",
        "policy_backend.routing_v2_flow_query_delta",
    }:
        return True
    if _is_lora(key, "policy_backend.vlm."):
        return True
    if _is_lora(key, "policy_backend.vlm_to_lam."):
        return True
    if _is_lora(key, "policy_backend.lam.decoder."):
        return True
    if _is_dense_last4(key) or _is_flow_side_adapter(key):
        return True
    return False


def _param_group(key: str) -> str:
    if key.startswith("policy_backend.vlm."):
        return "vlm_text_lora"
    if key.startswith("policy_backend.vlm_to_lam."):
        return "qformer_lora"
    if key.startswith("policy_backend.lam.decoder."):
        return "lawm_lora"
    if key in {
        "policy_backend.routing_v2_act_query_delta",
        "policy_backend.routing_v2_flow_query_delta",
    }:
        return "query_delta"
    if _is_dense_last4(key):
        return "flow_dense_last4"
    if ".conditioning_adapter." in key:
        return "flow_conditioning"
    if ".attn_nonlinear_adapter." in key or ".ffn_nonlinear_adapter." in key:
        return "flow_nonlinear"
    return "other"


def _checkpoint_tensor(state: Mapping[str, torch.Tensor], key: str):
    if key in state:
        return state[key]
    if key.startswith("policy_backend.flow."):
        alias = "policy_action_head." + key[len("policy_backend.flow."):]
        if alias in state:
            return state[alias]
    if key.startswith("policy_backend.vlm."):
        alias = "policy_vlm_adapter.model." + key[len("policy_backend.vlm."):]
        if alias in state:
            return state[alias]
    return None


class RoutingV2ClosedLoopPolicy:
    """Closed-loop task-agnostic Routing-V2 policy over a CL skill bank.

    Selection never reads GT/task ID.  `gt_task_id` is accepted only because the
    evaluation script runs one LIBERO task at a time and we want chunk-level
    diagnostics.  The decision path is strictly:

      Base anchor -> Semantic AE bank -> confidence gate -> optional Top-2
      Dynamics AE -> confidence-adaptive fusion -> selected Skill Path -> action.
    """

    def __init__(
        self,
        *,
        policy,
        skill_paths: Mapping[int, Path],
        semantic_bank,
        dynamics_bank,
        gt_task_id: int | None,
        stage: str,
        routing_log: Path,
        gate_threshold: float,
        lambda_max: float,
        gamma: float,
        debug_decisions: int = 8,
    ) -> None:
        self.policy = policy
        self.backend = policy.policy_backend
        self.batch_builder = policy.policy_infer_batch_builder
        self.tasks = sorted(int(t) for t in skill_paths)
        if not self.tasks:
            raise ValueError("Routing-V2 requires at least one candidate skill")
        if set(self.tasks) != set(semantic_bank) or set(self.tasks) != set(dynamics_bank):
            raise ValueError("Skill/Semantic/Dynamics banks do not match")
        self.semantic_bank = semantic_bank
        self.dynamics_bank = dynamics_bank
        self.gt_task_id = None if gt_task_id is None else int(gt_task_id)
        self.stage = str(stage)
        self.gate_threshold = float(gate_threshold)
        self.lambda_max = float(lambda_max)
        self.gamma = float(gamma)
        if not 0.0 <= self.lambda_max <= 1.0:
            raise ValueError("lambda_max must be in [0,1]")
        if self.gate_threshold < 0.0:
            raise ValueError("gate_threshold must be >= 0")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be > 0")
        self.debug_decisions = max(0, int(debug_decisions))
        self._decision_id = 0
        self.params = dict(policy.named_parameters(remove_duplicate=True))
        self.task_param_names = [name for name in self.params if _is_task_specific_param(name)]
        if not self.task_param_names:
            raise RuntimeError("No Routing-V2 task-specific parameters found in template model")

        self.task_overrides: dict[int, dict[str, torch.Tensor]] = {}
        for task in self.tasks:
            state = _unwrap_state(_torch_load(Path(skill_paths[task])))
            override: dict[str, torch.Tensor] = {}
            missing = []
            groups = defaultdict(int)
            for name in self.task_param_names:
                src = _checkpoint_tensor(state, name)
                if src is None:
                    missing.append(name)
                    continue
                p = self.params[name]
                if tuple(src.shape) != tuple(p.shape):
                    raise RuntimeError(
                        f"T{task} shape mismatch for {name}: ckpt={tuple(src.shape)} model={tuple(p.shape)}"
                    )
                tensor = src.to(device=p.device, dtype=p.dtype).detach()
                override[name] = tensor
                groups[_param_group(name)] += int(tensor.numel())
            if missing:
                raise RuntimeError(f"T{task} missing task-specific tensors: {missing[:20]}")
            self.task_overrides[task] = override
            logging.info(
                "[RoutingV2][BANK] T%d tensors=%d params=%s groups=%s",
                task,
                len(override),
                f"{sum(v.numel() for v in override.values()):,}",
                dict(sorted(groups.items())),
            )
            del state

        self.routing_log = Path(routing_log).expanduser().resolve()
        self.routing_log.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.routing_log.open("a", encoding="utf-8", buffering=1)
        self._apply_task(self.tasks[0])

        logging.info("=" * 72)
        logging.info("[RoutingV2] CLOSED-LOOP router ready")
        logging.info("  stage             : %s", self.stage)
        logging.info("  candidates        : %s", self.tasks)
        logging.info("  gt task           : %s (DIAGNOSTICS ONLY)", self.gt_task_id)
        logging.info("  gate              : C_sem < %.6f", self.gate_threshold)
        logging.info("  adaptive fusion   : lambda_max=%.4f gamma=%.4f", self.lambda_max, self.gamma)
        logging.info("  task params       : %s", f"{sum(self.params[n].numel() for n in self.task_param_names):,}")
        logging.info("  routing log       : %s", self.routing_log)
        logging.info("=" * 72)

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass

    @torch.no_grad()
    def _apply_task(self, task_id: int) -> None:
        override = self.task_overrides[int(task_id)]
        for name, src in override.items():
            self.params[name].copy_(src)

    @staticmethod
    def _confidence(e1: float, e2: float, eps: float = 1e-12) -> float:
        return float((e2 - e1) / max(abs(e2), eps))

    def _lambda_dyn(self, confidence: float) -> float:
        if len(self.tasks) < 2 or confidence >= self.gate_threshold:
            return 0.0
        if self.gate_threshold <= 0.0:
            return 0.0
        ratio = (self.gate_threshold - confidence) / self.gate_threshold
        ratio = max(0.0, min(1.0, ratio))
        return float(self.lambda_max * (ratio ** self.gamma))

    @staticmethod
    def _pair_share(value: float, other: float, eps: float = 1e-12) -> float:
        return float(value / max(value + other, eps))

    @torch.inference_mode()
    def _select_tasks(self, examples: Sequence[dict]) -> tuple[list[int], list[dict]]:
        batch = self.batch_builder.build_infer_batch(examples)
        anchor = self.backend._run_routing_v2_base_anchor_hact(prepared_batch=batch).detach()
        sem_cols = [
            self.semantic_bank[t](anchor)["per_sample_error"].detach().float()
            for t in self.tasks
        ]
        sem_errors = torch.stack(sem_cols, dim=1)  # [B,K]
        sem_order = torch.argsort(sem_errors, dim=1)

        # Determine which samples need WAM verification.  CL1 is intentionally
        # trivial: the single available skill is selected without Dynamics.
        sample_meta: list[dict] = []
        needed_dyn_tasks: set[int] = set()
        for i in range(len(examples)):
            ranked = [self.tasks[int(j)] for j in sem_order[i].cpu().tolist()]
            top1 = int(ranked[0])
            top2 = ranked[:2]
            if len(top2) >= 2:
                e1 = float(sem_errors[i, self.tasks.index(top2[0])].item())
                e2 = float(sem_errors[i, self.tasks.index(top2[1])].item())
                conf = self._confidence(e1, e2)
            else:
                e1 = float(sem_errors[i, self.tasks.index(top1)].item())
                e2 = e1
                conf = float("inf")
            gate = bool(len(top2) >= 2 and conf < self.gate_threshold)
            if gate:
                needed_dyn_tasks.update(int(t) for t in top2)
            sample_meta.append(
                {
                    "semantic_ranked": ranked,
                    "semantic_top1": top1,
                    "semantic_top2": top2,
                    "semantic_e1": e1,
                    "semantic_e2": e2,
                    "semantic_confidence": conf,
                    "gate_active": gate,
                }
            )

        dyn_per_task: dict[int, torch.Tensor] = {}
        for task in sorted(needed_dyn_tasks):
            self._apply_task(task)
            shared = self.backend._run_shared_encoding_infer(
                prepared_batch=batch,
                source=f"RoutingV2.closed_loop.dynamic.T{task}",
                lam_features_with_no_grad=False,
            )
            dyn_per_task[task] = self.dynamics_bank[task](
                h_t=shared.h_t,
                h_future=shared.h_t1_pred,
                z=shared.pred_action_emb,
            )["per_sample_error"].detach().float()

        selected: list[int] = []
        rows: list[dict] = []
        for i, meta in enumerate(sample_meta):
            top1 = int(meta["semantic_top1"])
            top2 = [int(t) for t in meta["semantic_top2"]]
            conf = float(meta["semantic_confidence"])
            gate = bool(meta["gate_active"])
            lam = self._lambda_dyn(conf) if gate else 0.0
            sem_dict = {
                str(task): float(sem_errors[i, j].item())
                for j, task in enumerate(self.tasks)
            }
            dyn_dict: dict[str, float] = {}
            fused_dict: dict[str, float] = {}
            if not gate:
                winner = top1
            else:
                a, b = top2
                sa, sb = sem_dict[str(a)], sem_dict[str(b)]
                da = float(dyn_per_task[a][i].item())
                db = float(dyn_per_task[b][i].item())
                dyn_dict = {str(a): da, str(b): db}
                sem_share = {
                    a: self._pair_share(sa, sb),
                    b: self._pair_share(sb, sa),
                }
                dyn_share = {
                    a: self._pair_share(da, db),
                    b: self._pair_share(db, da),
                }
                fused = {
                    task: (1.0 - lam) * sem_share[task] + lam * dyn_share[task]
                    for task in (a, b)
                }
                fused_dict = {str(task): float(fused[task]) for task in (a, b)}
                winner = int(min((a, b), key=lambda task: fused[task]))
            selected.append(winner)
            row = {
                "decision_id": int(self._decision_id + i),
                "stage": self.stage,
                "sample_index": int(i),
                "gt_task_id": self.gt_task_id,
                "candidate_tasks": self.tasks,
                "semantic_errors": sem_dict,
                "semantic_ranked_tasks": meta["semantic_ranked"],
                "semantic_top1_task": top1,
                "semantic_top2_tasks": top2,
                "semantic_confidence": None if not np.isfinite(conf) else conf,
                "gate_threshold": self.gate_threshold,
                "gate_active": gate,
                "lambda_dyn": lam,
                "dynamics_errors_top2": dyn_dict,
                "fused_scores_top2": fused_dict,
                "selected_task": int(winner),
                "semantic_top1_correct": None if self.gt_task_id is None else bool(top1 == self.gt_task_id),
                "routing_correct": None if self.gt_task_id is None else bool(winner == self.gt_task_id),
            }
            rows.append(row)
        self._decision_id += len(examples)
        return selected, rows

    @torch.inference_mode()
    def predict_action(self, examples: Sequence[dict], return_intermediates: bool = False, **kwargs):
        if not examples:
            raise ValueError("Routing-V2 closed-loop predict_action requires non-empty examples")
        if kwargs:
            logging.debug("[RoutingV2] ignored predict kwargs: %s", sorted(kwargs))
        if return_intermediates:
            logging.warning("[RoutingV2] return_intermediates requested; closed-loop router returns actions only")

        # IMPORTANT: GT/task ID is not injected into `examples`; it is only a
        # server-side diagnostic constant used after selection for logging.
        selected, rows = self._select_tasks(examples)

        # Generate actions only for the actually selected Skill Paths.  Grouping
        # samples by winner avoids running all action experts for every sample.
        out_actions: list[np.ndarray | None] = [None] * len(examples)
        by_task: dict[int, list[int]] = defaultdict(list)
        for i, task in enumerate(selected):
            by_task[int(task)].append(i)
        for task, indices in sorted(by_task.items()):
            self._apply_task(task)
            sub_examples = [examples[i] for i in indices]
            result = self.policy.predict_action(examples=sub_examples, return_intermediates=False)
            arr = np.asarray(result["normalized_actions"], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None, ...]
            if int(arr.shape[0]) != len(indices):
                raise RuntimeError(
                    f"T{task} action batch mismatch: got {arr.shape[0]} expected {len(indices)}"
                )
            for j, original_idx in enumerate(indices):
                out_actions[original_idx] = arr[j]

        if any(x is None for x in out_actions):
            raise RuntimeError("Routing-V2 failed to produce actions for every sample")
        stacked = np.stack([x for x in out_actions if x is not None], axis=0).astype(np.float32, copy=False)

        for row in rows:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if int(row["decision_id"]) < self.debug_decisions:
                logging.info(
                    "[RoutingV2][ROUTE][%s decision=%d GT=%s] sem1=T%d top2=%s C=%s gate=%s lambda=%.4f dyn=%s fused=%s -> T%d correct=%s",
                    self.stage,
                    row["decision_id"],
                    row["gt_task_id"],
                    row["semantic_top1_task"],
                    row["semantic_top2_tasks"],
                    "NA" if row["semantic_confidence"] is None else f"{row['semantic_confidence']:.6f}",
                    row["gate_active"],
                    row["lambda_dyn"],
                    row["dynamics_errors_top2"],
                    row["fused_scores_top2"],
                    row["selected_task"],
                    row["routing_correct"],
                )

        return {"normalized_actions": stacked}


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True, help="Template skill checkpoint; first candidate is recommended")
    ap.add_argument("--port", type=int, default=10093)
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--idle_timeout", type=int, default=1800)
    ap.add_argument("--candidate-skill", action="append", required=True, help="TASK_ID=/path/to/skill/pytorch_model.pt")
    ap.add_argument("--routing-memory", action="append", required=True, help="TASK_ID=/path/to/routing_memory.pt")
    ap.add_argument("--gt-task-id", type=int, default=None, help="Diagnostics only; NEVER used by routing")
    ap.add_argument("--stage", default="CL")
    ap.add_argument("--routing-log", required=True)
    ap.add_argument("--gate-threshold", type=float, default=0.20)
    ap.add_argument("--lambda-max", type=float, default=0.50)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--debug-decisions", type=int, default=8)
    ap.add_argument("--dynamics-heads", type=int, default=6)
    ap.add_argument("--dynamics-z-weight", type=float, default=0.5)
    return ap


def main(args) -> None:
    skill_paths: dict[int, Path] = {}
    for spec in args.candidate_skill:
        task, path = _parse_spec(spec, "candidate-skill")
        if task in skill_paths:
            raise ValueError(f"Duplicate candidate skill T{task}")
        skill_paths[task] = path
    if not skill_paths:
        raise RuntimeError("Empty candidate skill bank")

    template_task = sorted(skill_paths)[0]
    template_path = Path(args.ckpt_path).expanduser().resolve()
    if template_path != skill_paths[template_task]:
        logging.warning(
            "[RoutingV2] template=%s while first candidate T%d=%s; this is allowed only if architectures match",
            template_path,
            template_task,
            skill_paths[template_task],
        )
    policy = load_policy_from_checkpoint(str(template_path), use_bf16=bool(args.use_bf16), device="cuda")
    device = next(policy.parameters()).device

    sem_bank = {}
    dyn_bank = {}
    memory_paths = {}
    for spec in args.routing_memory:
        task, path = _parse_spec(spec, "routing-memory")
        if task in sem_bank:
            raise ValueError(f"Duplicate routing memory T{task}")
        sem, dyn = _load_memory(
            path,
            device=device,
            dynamics_heads=int(args.dynamics_heads),
            z_weight=float(args.dynamics_z_weight),
        )
        sem_bank[task] = sem
        dyn_bank[task] = dyn
        memory_paths[str(task)] = str(path)
    if set(skill_paths) != set(sem_bank):
        raise RuntimeError(
            f"Candidate/memory mismatch skills={sorted(skill_paths)} memories={sorted(sem_bank)}"
        )

    wrapper = RoutingV2ClosedLoopPolicy(
        policy=policy,
        skill_paths=skill_paths,
        semantic_bank=sem_bank,
        dynamics_bank=dyn_bank,
        gt_task_id=args.gt_task_id,
        stage=str(args.stage),
        routing_log=Path(args.routing_log),
        gate_threshold=float(args.gate_threshold),
        lambda_max=float(args.lambda_max),
        gamma=float(args.gamma),
        debug_decisions=int(args.debug_decisions),
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    metadata = build_policy_server_metadata(
        policy,
        ckpt_path=str(template_path),
        server_type="routing_v2_closed_loop",
        env="generic",
        supported_eval_envs=["libero"],
        extra_metadata={
            "routing_v2_closed_loop": True,
            "routing_v2_stage": str(args.stage),
            "routing_v2_candidates": sorted(skill_paths),
            "routing_v2_gt_task_id_diagnostics_only": args.gt_task_id,
            "routing_v2_gate_threshold": float(args.gate_threshold),
            "routing_v2_lambda_max": float(args.lambda_max),
            "routing_v2_gamma": float(args.gamma),
            "routing_v2_memory_paths": memory_paths,
        },
    )
    logging.info("Creating Routing-V2 closed-loop server host=%s ip=%s", hostname, local_ip)
    WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=int(args.port),
        idle_timeout=int(args.idle_timeout),
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
