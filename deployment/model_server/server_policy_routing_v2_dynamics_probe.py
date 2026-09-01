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
    SpatialDynamicsAutoencoder,
)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    state = _torch_load(path)
    if not isinstance(state, dict):
        raise RuntimeError(f"Routing memory must be a state dict: {path}")

    sem_state = {k[len("semantic."):]: v for k, v in state.items() if k.startswith("semantic.") and torch.is_tensor(v)}
    dyn_state = {k[len("dynamics."):]: v for k, v in state.items() if k.startswith("dynamics.") and torch.is_tensor(v)}
    if not sem_state or not dyn_state:
        raise RuntimeError(f"Missing semantic/dynamics tensors in {path}")

    ew = sem_state.get("encoder.0.weight")
    sem = SemanticTokenAutoencoder(input_dim=int(ew.shape[1]), bottleneck_dim=int(ew.shape[0]))
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
    for k in dyn_state:
        if k.startswith("encoder.layers."):
            try:
                layer_ids.append(int(k.split(".")[2]))
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
    return sem.to(device=device, dtype=torch.float32).eval(), dyn.to(device=device, dtype=torch.float32).eval()


def _is_upstream_delta_key(k: str) -> bool:
    if k in {"policy_backend.routing_v2_act_query_delta", "policy_backend.routing_v2_flow_query_delta"}:
        return True
    if k.startswith("policy_backend.vlm.") and (k.endswith(".lora_A") or k.endswith(".lora_B")):
        return True
    if k.startswith("policy_backend.vlm_to_lam.") and (k.endswith(".lora_A") or k.endswith(".lora_B")):
        return True
    if k.startswith("policy_backend.lam.decoder.") and (k.endswith(".lora_A") or k.endswith(".lora_B")):
        return True
    return False


def _load_upstream_delta(path: Path, *, policy, device: torch.device) -> Dict[str, torch.Tensor]:
    state = _torch_load(path)
    if not isinstance(state, dict):
        raise RuntimeError(f"Upstream delta must be a state dict: {path}")
    params = dict(policy.named_parameters())
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not torch.is_tensor(v) or not _is_upstream_delta_key(k):
            continue
        if k not in params:
            raise KeyError(f"Delta key {k} from {path} not present in loaded policy")
        p = params[k]
        if tuple(p.shape) != tuple(v.shape):
            raise RuntimeError(f"Shape mismatch {k}: policy={tuple(p.shape)} delta={tuple(v.shape)}")
        out[k] = v.to(device=device, dtype=p.dtype)
    expected = [k for k in params if _is_upstream_delta_key(k)]
    missing = sorted(set(expected) - set(out))
    extra = sorted(set(out) - set(expected))
    if missing or extra:
        raise RuntimeError(f"Upstream delta mismatch for {path}: missing={missing[:20]} extra={extra[:20]}")
    return out


class PassiveDynamicsProbePolicy:
    """Semantic Top-2 + Dynamics-AE passive verifier.

    Every rollout still executes the provided GT/task-ID skill.  Semantic and
    dynamics predictions are diagnostics only.  Candidate upstream task deltas
    are swapped temporarily to obtain z_k and h^pred_k, then the GT upstream
    delta is restored before oracle action generation.
    """

    def __init__(self, *, policy, semantic_bank, dynamics_bank, delta_bank, gt_task_id: int, output_jsonl: Path, debug_decisions: int = 8):
        self.policy = policy
        self.backend = policy.policy_backend
        self.batch_builder = policy.policy_infer_batch_builder
        self.gt_task_id = int(gt_task_id)
        self.tasks = sorted(int(t) for t in semantic_bank)
        if self.gt_task_id not in self.tasks:
            raise ValueError(f"GT T{self.gt_task_id} not in bank {self.tasks}")
        if set(self.tasks) != set(dynamics_bank) or set(self.tasks) != set(delta_bank):
            raise ValueError("Semantic/Dynamics/Delta task banks do not match")
        self.semantic_bank = semantic_bank
        self.dynamics_bank = dynamics_bank
        self.delta_bank = delta_bank
        self.params = dict(policy.named_parameters())
        self.output_jsonl = Path(output_jsonl).expanduser().resolve()
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_jsonl.open("a", encoding="utf-8", buffering=1)
        self._decision_id = 0
        self.debug_decisions = max(0, int(debug_decisions))
        if getattr(self.backend, "_run_routing_v2_base_anchor_hact", None) is None:
            raise RuntimeError("Routing-V2 Base anchor API missing")
        if getattr(self.backend, "_run_shared_encoding_infer", None) is None:
            raise RuntimeError("Routing-V2 shared inference API missing")
        self._apply_delta(self.gt_task_id)
        logging.info("[RoutingV2][DYN-PROBE] passive GT/action skill=T%d bank=%s", self.gt_task_id, self.tasks)
        logging.info("[RoutingV2][DYN-PROBE] Semantic Top-2 -> candidate skill imagination -> task Dynamics AE; NEVER controls action")
        logging.info("[RoutingV2][DYN-PROBE] JSONL=%s", self.output_jsonl)

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass

    @torch.no_grad()
    def _apply_delta(self, task_id: int) -> None:
        for k, src in self.delta_bank[int(task_id)].items():
            self.params[k].copy_(src)

    @staticmethod
    def _sem_conf(e1: float, e2: float, eps: float = 1e-12) -> dict:
        gap = e2 - e1
        return {
            "semantic_abs_margin": gap,
            "semantic_relative_margin": gap / max(abs(e1), eps),
            "semantic_normalized_gap": gap / max(abs(e2), eps),
            "semantic_error_ratio": e2 / max(abs(e1), eps),
        }

    @torch.inference_mode()
    def _probe(self, examples: Sequence[dict]) -> None:
        if not examples:
            return
        batch = self.batch_builder.build_infer_batch(examples)

        # Stage 1: common Base anchor and Semantic AE bank.
        anchor = self.backend._run_routing_v2_base_anchor_hact(prepared_batch=batch).detach()
        sem_cols = []
        for t in self.tasks:
            sem_cols.append(self.semantic_bank[t](anchor)["per_sample_error"].detach().float())
        sem_errors = torch.stack(sem_cols, dim=1)  # [B,K]
        sem_order = torch.argsort(sem_errors, dim=1)

        # Top-2 can differ per batch sample. Evaluate each candidate task once for
        # the whole batch, then use the sample-specific score only when it is in Top-2.
        needed = sorted({self.tasks[int(c)] for row in sem_order[:, :2].cpu().tolist() for c in row})
        dyn_per_task: Dict[int, torch.Tensor] = {}
        for t in needed:
            self._apply_delta(t)
            shared = self.backend._run_shared_encoding_infer(
                prepared_batch=batch,
                source=f"RoutingV2.passive_dynamics.T{t}",
                lam_features_with_no_grad=False,
            )
            dyn = self.dynamics_bank[t](
                h_t=shared.h_t,
                h_future=shared.h_t1_pred,
                z=shared.pred_action_emb,
            )["per_sample_error"].detach().float()
            dyn_per_task[t] = dyn

        # Restore oracle upstream path before action generation.
        self._apply_delta(self.gt_task_id)

        for i, ex in enumerate(examples):
            ranked = [self.tasks[int(c)] for c in sem_order[i].cpu().tolist()]
            top2 = ranked[:2]
            sdict = {str(t): float(sem_errors[i, j].item()) for j, t in enumerate(self.tasks)}
            e1, e2 = sdict[str(top2[0])], sdict[str(top2[1])]
            ddict = {str(t): float(dyn_per_task[t][i].item()) for t in top2}
            dyn_ranked = sorted(top2, key=lambda t: ddict[str(t)])
            dyn_winner = int(dyn_ranked[0])
            sem_top1 = int(top2[0])
            sem_top1_correct = sem_top1 == self.gt_task_id
            sem_top2_correct = self.gt_task_id in top2
            dyn_correct = dyn_winner == self.gt_task_id
            row = {
                "decision_id": int(self._decision_id),
                "sample_index": int(i),
                "gt_task_id": int(self.gt_task_id),
                "instruction": str(ex.get("lang", "")),
                "bank_tasks": self.tasks,
                "semantic_errors": sdict,
                "semantic_ranked_tasks": ranked,
                "semantic_top1_task": sem_top1,
                "semantic_top2_tasks": top2,
                "semantic_gt_rank": int(ranked.index(self.gt_task_id) + 1),
                "semantic_top1_correct": bool(sem_top1_correct),
                "semantic_top2_correct": bool(sem_top2_correct),
                **self._sem_conf(e1, e2),
                "dynamics_errors_top2": ddict,
                "dynamics_ranked_top2": dyn_ranked,
                "dynamics_winner_task": dyn_winner,
                "dynamics_correct": bool(dyn_correct),
                "semantic_error_recovered": bool((not sem_top1_correct) and dyn_correct),
                "semantic_correct_damaged": bool(sem_top1_correct and (not dyn_correct)),
            }
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if self._decision_id < self.debug_decisions:
                logging.info(
                    "[RoutingV2][DYN][decision=%d sample=%d GT=T%d] sem=%s top2=%s conf=%.6f dyn=%s -> sem=T%d dyn=T%d recovered=%s damaged=%s",
                    self._decision_id, i, self.gt_task_id,
                    ",".join(f"T{t}:{sdict[str(t)]:.5f}" for t in self.tasks),
                    top2, row["semantic_normalized_gap"],
                    ",".join(f"T{t}:{ddict[str(t)]:.5f}" for t in top2),
                    sem_top1, dyn_winner, row["semantic_error_recovered"], row["semantic_correct_damaged"],
                )
            self._decision_id += 1

    @torch.inference_mode()
    def predict_action(self, examples: Sequence[dict], return_intermediates: bool = False, **kwargs):
        self._probe(examples)
        # GT upstream delta is restored inside _probe; the originally loaded GT
        # action expert was never swapped. Thus action execution is oracle/task-ID.
        return self.policy.predict_action(
            examples=examples,
            return_intermediates=bool(return_intermediates),
            **kwargs,
        )


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--port", type=int, default=10093)
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--idle_timeout", type=int, default=1800)
    ap.add_argument("--gt-task-id", type=int, required=True)
    ap.add_argument("--routing-memory", action="append", required=True, help="TASK_ID=/path/to/routing_memory.pt")
    ap.add_argument("--upstream-delta", action="append", required=True, help="TASK_ID=/path/to/routing_upstream_delta.pt")
    ap.add_argument("--probe-output", required=True)
    ap.add_argument("--debug-decisions", type=int, default=8)
    ap.add_argument("--dynamics-heads", type=int, default=6)
    ap.add_argument("--dynamics-z-weight", type=float, default=0.5)
    return ap


def main(args) -> None:
    policy = load_policy_from_checkpoint(args.ckpt_path, use_bf16=bool(args.use_bf16), device="cuda")
    device = next(policy.parameters()).device

    sem_bank = {}; dyn_bank = {}; mem_paths = {}
    for spec in args.routing_memory:
        t, p = _parse_spec(spec, "routing-memory")
        if t in sem_bank:
            raise ValueError(f"Duplicate memory T{t}")
        sem, dyn = _load_memory(p, device=device, dynamics_heads=args.dynamics_heads, z_weight=args.dynamics_z_weight)
        sem_bank[t] = sem; dyn_bank[t] = dyn; mem_paths[str(t)] = str(p)
        logging.info("[RoutingV2][DYN-PROBE] loaded T%d routing memory %s", t, p)

    delta_bank = {}; delta_paths = {}
    for spec in args.upstream_delta:
        t, p = _parse_spec(spec, "upstream-delta")
        if t in delta_bank:
            raise ValueError(f"Duplicate upstream delta T{t}")
        delta_bank[t] = _load_upstream_delta(p, policy=policy, device=device)
        delta_paths[str(t)] = str(p)
        logging.info("[RoutingV2][DYN-PROBE] loaded T%d upstream delta to GPU (%d tensors)", t, len(delta_bank[t]))

    wrapper = PassiveDynamicsProbePolicy(
        policy=policy,
        semantic_bank=sem_bank,
        dynamics_bank=dyn_bank,
        delta_bank=delta_bank,
        gt_task_id=int(args.gt_task_id),
        output_jsonl=Path(args.probe_output),
        debug_decisions=int(args.debug_decisions),
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    metadata = build_policy_server_metadata(
        policy,
        ckpt_path=args.ckpt_path,
        server_type="routing_v2_passive_dynamics_probe",
        env="generic",
        supported_eval_envs=["libero"],
        extra_metadata={
            "routing_v2_probe": "semantic_top2_dynamics_ae",
            "routing_v2_probe_passive": True,
            "routing_v2_gt_task_id_diagnostics_only": int(args.gt_task_id),
            "routing_v2_bank": sorted(sem_bank),
            "routing_v2_memory_paths": mem_paths,
            "routing_v2_upstream_delta_paths": delta_paths,
        },
    )
    logging.info("Creating passive dynamics-probe server host=%s ip=%s", hostname, local_ip)
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
