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
from starVLA.model.framework.latent_world.routing_v2 import temporarily_disable_lora
from starVLA.model.framework.latent_world.routing_v2.autoencoders import (
    SemanticTokenAutoencoder,
    SpatialDynamicsAutoencoder,
)


VARIANT_SPECS = {
    "taskwm_hdhz": ("task", "hdhz"),
    "taskwm_hdh": ("task", "hdh"),
    "taskwm_dh": ("task", "dh"),
    "basewm_hdhz": ("base", "hdhz"),
    "basewm_hdh": ("base", "hdh"),
    "basewm_dh": ("base", "dh"),
}
VARIANTS = tuple(VARIANT_SPECS)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_task_path(spec: str, label: str) -> tuple[int, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --{label} {spec!r}; expected TASK_ID=/path")
    task_s, path_s = spec.split("=", 1)
    task_id = int(task_s)
    path = Path(path_s).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} for T{task_id}: {path}")
    return task_id, path


def _parse_variant_path(spec: str) -> tuple[str, int, Path]:
    # Syntax: VARIANT:TASK_ID=/path
    if "=" not in spec or ":" not in spec.split("=", 1)[0]:
        raise ValueError(
            f"Invalid --dynamics-memory {spec!r}; expected VARIANT:TASK_ID=/path"
        )
    lhs, path_s = spec.split("=", 1)
    variant, task_s = lhs.split(":", 1)
    variant = variant.strip()
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant={variant!r}; expected one of {VARIANTS}")
    task_id = int(task_s)
    path = Path(path_s).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Dynamics memory {variant}/T{task_id}: {path}")
    return variant, task_id, path


def _load_semantic(path: Path, *, device: torch.device) -> SemanticTokenAutoencoder:
    state = _torch_load(path)
    if not isinstance(state, dict):
        raise RuntimeError(f"Routing memory must be a state dict: {path}")
    sem_state = {
        k[len("semantic."):]: v
        for k, v in state.items()
        if isinstance(k, str) and k.startswith("semantic.") and torch.is_tensor(v)
    }
    if not sem_state:
        # Support full model state dict too.
        prefix = "policy_backend.routing_v2_memory.semantic."
        sem_state = {
            k[len(prefix):]: v
            for k, v in state.items()
            if isinstance(k, str) and k.startswith(prefix) and torch.is_tensor(v)
        }
    if not sem_state:
        raise RuntimeError(f"Missing Semantic AE tensors in {path}")
    ew = sem_state.get("encoder.0.weight")
    if ew is None:
        raise RuntimeError(f"Cannot infer Semantic AE geometry from {path}")
    sem = SemanticTokenAutoencoder(input_dim=int(ew.shape[1]), bottleneck_dim=int(ew.shape[0]))
    sem.load_state_dict(sem_state, strict=True)
    return sem.to(device=device, dtype=torch.float32).eval()


def _infer_num_layers(state: dict[str, torch.Tensor]) -> int:
    ids = []
    for k in state:
        if k.startswith("encoder.layers."):
            try:
                ids.append(int(k.split(".")[2]))
            except Exception:
                pass
    return max(ids) + 1 if ids else 0


def _load_dynamics(
    path: Path,
    *,
    device: torch.device,
    input_mode: str,
    latent_dim: int,
    dynamics_heads: int,
    z_weight: float,
) -> SpatialDynamicsAutoencoder:
    state = _torch_load(path)
    if not isinstance(state, dict):
        raise RuntimeError(f"Dynamics memory must be a state dict: {path}")

    # The 2x3 extraction script saves a plain SpatialDynamicsAutoencoder state dict.
    if "input_proj.weight" not in state:
        prefixes = (
            "policy_backend.routing_v2_memory.dynamics.",
            "routing_v2_memory.dynamics.",
            "dynamics.",
        )
        found = None
        for prefix in prefixes:
            tmp = {
                k[len(prefix):]: v
                for k, v in state.items()
                if isinstance(k, str) and k.startswith(prefix) and torch.is_tensor(v)
            }
            if tmp:
                found = tmp
                break
        if found is None:
            raise RuntimeError(f"Cannot find Dynamics-AE tensors in {path}")
        state = found
    else:
        state = {k: v for k, v in state.items() if torch.is_tensor(v)}

    iw = state["input_proj.weight"]
    pos = state["pos_embed"]
    dw = state["delta_decoder.weight"]
    hidden = int(iw.shape[0])
    vision = int(dw.shape[0])
    ntokens = int(pos.shape[1])
    ffn = int(state["encoder.layers.0.linear1.weight"].shape[0])
    nlayers = _infer_num_layers(state)

    expected_in = vision
    if input_mode in {"hdhz", "hdh"}:
        expected_in += vision
    if input_mode == "hdhz":
        zw = state.get("z_decoder.1.weight")
        if zw is None:
            raise RuntimeError(f"{input_mode} memory missing z_decoder in {path}")
        inferred_latent = int(zw.shape[0])
        if inferred_latent != int(latent_dim):
            raise RuntimeError(
                f"latent dim mismatch in {path}: memory={inferred_latent} policy={latent_dim}"
            )
        expected_in += latent_dim
    else:
        if any(k.startswith("z_decoder.") for k in state):
            raise RuntimeError(f"{input_mode} memory unexpectedly contains z_decoder in {path}")
    if int(iw.shape[1]) != expected_in:
        raise RuntimeError(
            f"Dynamics geometry mismatch for mode={input_mode} in {path}: "
            f"input_proj={tuple(iw.shape)}, expected input={expected_in}"
        )

    dyn = SpatialDynamicsAutoencoder(
        vision_dim=vision,
        latent_dim=int(latent_dim),
        num_tokens=ntokens,
        hidden_dim=hidden,
        num_layers=nlayers,
        num_heads=int(dynamics_heads),
        ffn_dim=ffn,
        z_loss_weight=float(z_weight),
        input_mode=input_mode,
    )
    dyn.load_state_dict(state, strict=True)
    return dyn.to(device=device, dtype=torch.float32).eval()


def _is_upstream_delta_key(k: str) -> bool:
    if k in {
        "policy_backend.routing_v2_act_query_delta",
        "policy_backend.routing_v2_flow_query_delta",
    }:
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
        raise RuntimeError(
            f"Upstream delta mismatch for {path}: missing={missing[:20]} extra={extra[:20]}"
        )
    return out


class Passive2x3DynamicsProbePolicy:
    """Paired passive evaluation for all six future-dynamics variants.

    The robot always executes the provided oracle/GT skill.  Every variant sees
    exactly the same chunks, Semantic Top-2 candidates, and task-specific latent
    actions.  For each Top-2 candidate we compute task-WM and shared-Base-WM
    futures once, then score all three representation modes for each future.
    None of these scores can affect the executed action.
    """

    def __init__(
        self,
        *,
        policy,
        semantic_bank,
        dynamics_banks,
        delta_bank,
        gt_task_id: int,
        output_jsonl: Path,
        debug_decisions: int = 8,
    ):
        self.policy = policy
        self.backend = policy.policy_backend
        self.batch_builder = policy.policy_infer_batch_builder
        self.gt_task_id = int(gt_task_id)
        self.tasks = sorted(int(t) for t in semantic_bank)
        if self.gt_task_id not in self.tasks:
            raise ValueError(f"GT T{self.gt_task_id} not in bank {self.tasks}")
        if set(self.tasks) != set(delta_bank):
            raise ValueError("Semantic/Delta task banks do not match")
        if set(dynamics_banks) != set(VARIANTS):
            raise ValueError(f"Expected Dynamics variants {VARIANTS}, got {sorted(dynamics_banks)}")
        for v in VARIANTS:
            if set(dynamics_banks[v]) != set(self.tasks):
                raise ValueError(f"Dynamics bank {v} tasks do not match Semantic bank")

        self.semantic_bank = semantic_bank
        self.dynamics_banks = dynamics_banks
        self.delta_bank = delta_bank
        self.params = dict(policy.named_parameters())
        self.output_jsonl = Path(output_jsonl).expanduser().resolve()
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_jsonl.open("a", encoding="utf-8", buffering=1)
        self._decision_id = 0
        self.debug_decisions = max(0, int(debug_decisions))

        for name in (
            "_run_routing_v2_base_anchor_hact",
            "_run_shared_encoding_infer",
            "_decode_future_tokens_strict_single_query",
        ):
            if getattr(self.backend, name, None) is None:
                raise RuntimeError(f"Routing-V2 required API missing: {name}")

        self._apply_delta(self.gt_task_id)
        logging.info(
            "[RoutingV2][2x3-PROBE] passive oracle/action skill=T%d bank=%s variants=%s",
            self.gt_task_id,
            self.tasks,
            list(VARIANTS),
        )
        logging.info(
            "[RoutingV2][2x3-PROBE] ALL six variants share the exact same chunks and Semantic Top-2; NEVER control action"
        )
        logging.info("[RoutingV2][2x3-PROBE] JSONL=%s", self.output_jsonl)

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

        # Stage 1: one common Base semantic anchor for all six variants.
        anchor = self.backend._run_routing_v2_base_anchor_hact(prepared_batch=batch).detach()
        sem_cols = []
        for t in self.tasks:
            sem_cols.append(self.semantic_bank[t](anchor)["per_sample_error"].detach().float())
        sem_errors = torch.stack(sem_cols, dim=1)
        sem_order = torch.argsort(sem_errors, dim=1)
        needed = sorted(
            {self.tasks[int(c)] for row in sem_order[:, :2].cpu().tolist() for c in row}
        )

        # Cache each candidate's two counterfactual future sources once.
        # z_k and h_t are identical for Task-WM and Base-WM; only the decoder
        # LoRA contribution is switched off for Base-WM future prediction.
        candidate_cache: dict[int, dict[str, torch.Tensor]] = {}
        for t in needed:
            self._apply_delta(t)
            shared = self.backend._run_shared_encoding_infer(
                prepared_batch=batch,
                source=f"RoutingV2.2x3_probe.T{t}",
                lam_features_with_no_grad=False,
            )
            h_t = shared.h_t.detach()
            z = shared.pred_action_emb.detach()
            h_task = shared.h_t1_pred.detach()
            # `_run_shared_encoding_infer` computes the normal Task-WM future
            # inside LaWAM's bf16 autocast region.  This explicit Base-WM
            # decode happens outside that region.  In bf16 server mode the
            # candidate latent z can therefore remain float32 while decoder
            # weights are bfloat16, which makes LoRALinear/F.linear fail with
            # `mat1 and mat2 must have the same dtype`.  Re-enter the decoder
            # precision context here so Base-WM and Task-WM use the same
            # inference precision contract.
            decoder_param = next(self.backend.lam.decoder.parameters())
            decoder_dtype = decoder_param.dtype
            with temporarily_disable_lora(self.backend.lam.decoder):
                if h_t.is_cuda and decoder_dtype in (torch.float16, torch.bfloat16):
                    with torch.autocast(device_type="cuda", dtype=decoder_dtype):
                        h_base = self.backend._decode_future_tokens_strict_single_query(
                            h_t=h_t,
                            pred_action_emb=z,
                            source=f"RoutingV2.2x3_probe.baseWM.T{t}",
                        ).detach()
                else:
                    h_base = self.backend._decode_future_tokens_strict_single_query(
                        h_t=h_t.to(dtype=decoder_dtype),
                        pred_action_emb=z.to(dtype=decoder_dtype),
                        source=f"RoutingV2.2x3_probe.baseWM.T{t}",
                    ).detach()
            if self._decision_id == 0:
                logging.info(
                    "[RoutingV2][2x3-PROBE][DTYPE] T%d h_t=%s z=%s decoder=%s h_task=%s h_base=%s",
                    t, h_t.dtype, z.dtype, decoder_dtype, h_task.dtype, h_base.dtype,
                )
            candidate_cache[t] = {
                "h_t": h_t,
                "z": z,
                "task": h_task,
                "base": h_base,
            }

        # Score all six variants. Each Dynamics AE only consumes the future
        # source and representation it was trained for.
        dyn_per_variant: dict[str, dict[int, torch.Tensor]] = {
            v: {} for v in VARIANTS
        }
        for v, (wm_source, input_mode) in VARIANT_SPECS.items():
            for t in needed:
                cc = candidate_cache[t]
                out = self.dynamics_banks[v][t](
                    h_t=cc["h_t"],
                    h_future=cc[wm_source],
                    z=cc["z"] if input_mode == "hdhz" else None,
                )
                dyn_per_variant[v][t] = out["per_sample_error"].detach().float()

        # Restore oracle upstream path before real action generation.
        self._apply_delta(self.gt_task_id)

        for i, ex in enumerate(examples):
            ranked = [self.tasks[int(c)] for c in sem_order[i].cpu().tolist()]
            top2 = ranked[:2]
            sdict = {
                str(t): float(sem_errors[i, j].item())
                for j, t in enumerate(self.tasks)
            }
            e1, e2 = sdict[str(top2[0])], sdict[str(top2[1])]
            sem_top1 = int(top2[0])
            sem_top1_correct = sem_top1 == self.gt_task_id
            sem_top2_correct = self.gt_task_id in top2

            variant_rows = {}
            for v in VARIANTS:
                ddict = {
                    str(t): float(dyn_per_variant[v][t][i].item()) for t in top2
                }
                dyn_ranked = sorted(top2, key=lambda t: ddict[str(t)])
                winner = int(dyn_ranked[0])
                correct = winner == self.gt_task_id
                variant_rows[v] = {
                    "wm_source": VARIANT_SPECS[v][0],
                    "input_mode": VARIANT_SPECS[v][1],
                    "dynamics_errors_top2": ddict,
                    "dynamics_ranked_top2": dyn_ranked,
                    "dynamics_winner_task": winner,
                    "dynamics_correct": bool(correct),
                    "semantic_error_recovered": bool((not sem_top1_correct) and correct),
                    "semantic_correct_damaged": bool(sem_top1_correct and (not correct)),
                }

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
                "variants": variant_rows,
            }
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            if self._decision_id < self.debug_decisions:
                short = []
                for v in VARIANTS:
                    vr = variant_rows[v]
                    short.append(
                        f"{v}:T{vr['dynamics_winner_task']}"
                        f"({'Y' if vr['dynamics_correct'] else 'N'})"
                    )
                logging.info(
                    "[RoutingV2][2x3][decision=%d sample=%d GT=T%d] semTop2=%s C=%.6f | %s",
                    self._decision_id,
                    i,
                    self.gt_task_id,
                    top2,
                    row["semantic_normalized_gap"],
                    " ".join(short),
                )
            self._decision_id += 1

    @torch.inference_mode()
    def predict_action(self, examples: Sequence[dict], return_intermediates: bool = False, **kwargs):
        self._probe(examples)
        # GT delta has been restored; no Dynamics variant can control the robot.
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
    ap.add_argument(
        "--semantic-memory",
        action="append",
        required=True,
        help="TASK_ID=/path/to/original/routing_memory.pt",
    )
    ap.add_argument(
        "--dynamics-memory",
        action="append",
        required=True,
        help="VARIANT:TASK_ID=/path/to/dynamics_ae.pt",
    )
    ap.add_argument(
        "--upstream-delta",
        action="append",
        required=True,
        help="TASK_ID=/path/to/routing_upstream_delta.pt",
    )
    ap.add_argument("--probe-output", required=True)
    ap.add_argument("--debug-decisions", type=int, default=8)
    ap.add_argument("--dynamics-heads", type=int, default=6)
    ap.add_argument("--dynamics-z-weight", type=float, default=0.5)
    return ap


def main(args) -> None:
    policy = load_policy_from_checkpoint(
        args.ckpt_path,
        use_bf16=bool(args.use_bf16),
        device="cuda",
    )
    device = next(policy.parameters()).device
    backend = policy.policy_backend
    latent_dim = int(getattr(backend.lam, "code_dim"))

    semantic_bank = {}
    semantic_paths = {}
    for spec in args.semantic_memory:
        t, p = _parse_task_path(spec, "semantic-memory")
        if t in semantic_bank:
            raise ValueError(f"Duplicate Semantic memory T{t}")
        semantic_bank[t] = _load_semantic(p, device=device)
        semantic_paths[str(t)] = str(p)
        logging.info("[RoutingV2][2x3-PROBE] loaded T%d Semantic AE %s", t, p)

    dynamics_banks: dict[str, dict[int, SpatialDynamicsAutoencoder]] = {
        v: {} for v in VARIANTS
    }
    dynamics_paths: dict[str, dict[str, str]] = {v: {} for v in VARIANTS}
    for spec in args.dynamics_memory:
        v, t, p = _parse_variant_path(spec)
        if t in dynamics_banks[v]:
            raise ValueError(f"Duplicate Dynamics memory {v}/T{t}")
        input_mode = VARIANT_SPECS[v][1]
        dynamics_banks[v][t] = _load_dynamics(
            p,
            device=device,
            input_mode=input_mode,
            latent_dim=latent_dim,
            dynamics_heads=int(args.dynamics_heads),
            z_weight=float(args.dynamics_z_weight),
        )
        dynamics_paths[v][str(t)] = str(p)
        logging.info("[RoutingV2][2x3-PROBE] loaded %s/T%d %s", v, t, p)

    tasks = sorted(semantic_bank)
    for v in VARIANTS:
        if sorted(dynamics_banks[v]) != tasks:
            raise RuntimeError(
                f"Variant {v} incomplete: expected tasks={tasks}, got={sorted(dynamics_banks[v])}"
            )

    delta_bank = {}
    delta_paths = {}
    for spec in args.upstream_delta:
        t, p = _parse_task_path(spec, "upstream-delta")
        if t in delta_bank:
            raise ValueError(f"Duplicate upstream delta T{t}")
        delta_bank[t] = _load_upstream_delta(p, policy=policy, device=device)
        delta_paths[str(t)] = str(p)
        logging.info(
            "[RoutingV2][2x3-PROBE] loaded T%d upstream delta (%d tensors)",
            t,
            len(delta_bank[t]),
        )
    if sorted(delta_bank) != tasks:
        raise RuntimeError(
            f"Upstream delta bank mismatch: Semantic={tasks} delta={sorted(delta_bank)}"
        )

    wrapper = Passive2x3DynamicsProbePolicy(
        policy=policy,
        semantic_bank=semantic_bank,
        dynamics_banks=dynamics_banks,
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
        server_type="routing_v2_passive_2x3_dynamics_probe",
        env="generic",
        supported_eval_envs=["libero"],
        extra_metadata={
            "routing_v2_probe": "paired_2x3_dynamics_ablation",
            "routing_v2_probe_passive": True,
            "routing_v2_gt_task_id_diagnostics_only": int(args.gt_task_id),
            "routing_v2_bank": tasks,
            "routing_v2_variants": list(VARIANTS),
            "routing_v2_semantic_paths": semantic_paths,
            "routing_v2_dynamics_paths": dynamics_paths,
            "routing_v2_upstream_delta_paths": delta_paths,
        },
    )
    logging.info("Creating passive 2x3 probe server host=%s ip=%s", hostname, local_ip)
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
