from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.latent_world.runtime.output_mapper import map_policy_infer_output


_DENSE_BLOCK_RE = re.compile(r"^policy_backend\.flow\.DiT\.transformer_blocks\.(\d+)\.")
_ROUTING_META_KEYS = {
    "routing_gt_task_id",
    "routing_episode_id",
    "routing_slot_id",
}


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    for wrapper in ("state_dict", "model", "module"):
        inner = obj.get(wrapper)
        if isinstance(inner, dict) and inner and any(torch.is_tensor(v) for v in inner.values()):
            obj = inner
            break
    return obj




def _checkpoint_tensor(state: Mapping[str, torch.Tensor], canonical_key: str) -> torch.Tensor | None:
    if canonical_key in state:
        return state[canonical_key]
    if canonical_key.startswith("policy_backend.flow."):
        alias = "policy_action_head." + canonical_key[len("policy_backend.flow.") :]
        if alias in state:
            return state[alias]
    if canonical_key.startswith("policy_backend.vlm."):
        alias = "policy_vlm_adapter.model." + canonical_key[len("policy_backend.vlm.") :]
        if alias in state:
            return state[alias]
    return None
def _wrapper_to_ckpt_key(wrapper_name: str) -> str:
    if not wrapper_name.startswith("backend."):
        raise ValueError(f"Unexpected routing-wrapper parameter name: {wrapper_name}")
    return "policy_backend." + wrapper_name[len("backend.") :]


def _is_conditioning_adapter(key: str) -> bool:
    return key.startswith("policy_backend.flow.") and ".conditioning_adapter." in key


def _is_nonlinear_adapter(key: str) -> bool:
    return key.startswith("policy_backend.flow.") and (
        ".attn_nonlinear_adapter." in key or ".ffn_nonlinear_adapter." in key
    )


def _is_expert_latent_head(key: str) -> bool:
    return key.startswith("policy_backend.flow.expert_latent_head.")


def _is_dense_last4(key: str) -> bool:
    match = _DENSE_BLOCK_RE.match(key)
    return match is not None and int(match.group(1)) in {12, 13, 14, 15}


def _is_text_lora(key: str) -> bool:
    return key.startswith("policy_backend.vlm.") and (
        key.endswith(".lora_A") or key.endswith(".lora_B")
    )


def _is_task_specific(key: str, mode: str) -> bool:
    if _is_expert_latent_head(key) or _is_conditioning_adapter(key) or _is_nonlinear_adapter(key):
        return True
    if _is_dense_last4(key):
        return True
    if mode == "b2" and _is_text_lora(key):
        return True
    return False


def _category(key: str) -> str:
    if _is_expert_latent_head(key):
        return "latent_head"
    if _is_conditioning_adapter(key):
        return "conditioning"
    if _is_nonlinear_adapter(key):
        return "nonlinear"
    if _is_dense_last4(key):
        return "dense_last4"
    if _is_text_lora(key):
        return "text_lora"
    return "other"


def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if tuple(x.shape) != tuple(y.shape):
        raise ValueError(f"Routing cosine shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    distance = 1.0 - F.cosine_similarity(x.float(), y.float(), dim=-1)
    if distance.ndim == 1:
        return distance
    return distance.reshape(distance.shape[0], -1).mean(dim=1)


class _BackendRoutingCall(nn.Module):
    """Thin wrapper so torch.func.functional_call can swap branch parameters."""

    def __init__(self, backend: nn.Module) -> None:
        super().__init__()
        self.backend = backend

    def forward(
        self,
        batch: dict[str, torch.Tensor | None],
        initial_noise: torch.Tensor,
        guidance_scale: float | None,
        num_inference_steps: int | None,
    ) -> Dict[str, torch.Tensor]:
        return self.backend.predict_action_routing(
            batch=batch,
            initial_noise=initial_noise,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )


class _FlowRoutingCall(nn.Module):
    """B1 fast path: shared VLM/QFormer/LaWM are computed once, then only Flow varies."""

    def __init__(self, flow: nn.Module) -> None:
        super().__init__()
        self.flow = flow

    def forward(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        action_hz: torch.Tensor,
        embodiment_id: torch.Tensor,
        attention_mask: torch.Tensor,
        initial_noise: torch.Tensor,
        guidance_scale: float | None,
        num_inference_steps: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.flow.sample_actions_cfg(
            h_t=h_t,
            h_t1_star=h_t1_star,
            h_vlm=h_vlm,
            state=state,
            state_mask=state_mask,
            action_hz=action_hz,
            embodiment_id=embodiment_id,
            cfg_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            attention_mask=attention_mask,
            initial_noise=initial_noise,
            return_expert_latent=True,
        )
        if not isinstance(out, tuple) or len(out) != 2:
            raise RuntimeError("B1 Routing-V1 expected Flow to return (actions, z*).")
        return out


class RoutingV1ExpertBankPolicy:
    """Task-agnostic Routing-V1 policy over a Base/B1/B2 expert bank.

    The full model is materialized once from one task-expert checkpoint. Only the
    task-specific tensors are kept for each candidate branch and injected via
    `torch.func.functional_call`, avoiding K copies of the 2B+ shared LaWAM.
    """

    def __init__(
        self,
        *,
        base_checkpoint: str | Path,
        expert_checkpoints: Mapping[int, str | Path],
        candidate_labels: Sequence[str],
        mode: str,
        score_mode: str,
        alpha: float,
        score_normalization: str = "none",
        temporal_mode: str = "none",
        temporal_beta: float = 0.0,
        temporal_margin: float = 0.0,
        execution_mode: str = "route",
        device: str = "cuda",
        use_bf16: bool = True,
        routing_log_path: str | Path | None = None,
        context_label: str | None = None,
        require_gt_diagnostics: bool = False,
        debug_requests: int = 5,
        guidance_scale: float | None = None,
        num_inference_steps: int | None = None,
    ) -> None:
        mode = str(mode).lower()
        if mode not in {"b1", "b2"}:
            raise ValueError(f"Routing-V1 mode must be b1 or b2, got {mode!r}.")
        score_mode = str(score_mode).lower()
        if score_mode not in {"latent", "world", "combined"}:
            raise ValueError(
                f"score_mode must be latent/world/combined, got {score_mode!r}."
            )
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}.")
        score_normalization = str(score_normalization).lower()
        if score_normalization not in {"none", "candidate_mean"}:
            raise ValueError(
                f"score_normalization must be none/candidate_mean, got {score_normalization!r}."
            )
        temporal_mode = str(temporal_mode).lower()
        if temporal_mode not in {"none", "ema", "ema_hysteresis"}:
            raise ValueError(
                f"temporal_mode must be none/ema/ema_hysteresis, got {temporal_mode!r}."
            )
        if not 0.0 <= float(temporal_beta) < 1.0:
            raise ValueError(f"temporal_beta must be in [0,1), got {temporal_beta}.")
        if float(temporal_margin) < 0.0:
            raise ValueError(f"temporal_margin must be >= 0, got {temporal_margin}.")
        execution_mode = str(execution_mode).lower()
        if execution_mode not in {"route", "oracle_execute"}:
            raise ValueError(
                f"execution_mode must be route/oracle_execute, got {execution_mode!r}."
            )

        self.mode = mode
        self.score_mode = score_mode
        self.alpha = float(alpha)
        self.score_normalization = score_normalization
        self.temporal_mode = temporal_mode
        self.temporal_beta = float(temporal_beta)
        self.temporal_margin = float(temporal_margin)
        self.execution_mode = execution_mode
        self.device = torch.device(device)
        self.debug_requests = int(debug_requests)
        self.context_label = None if context_label is None else str(context_label)
        self.require_gt_diagnostics = bool(require_gt_diagnostics)
        if self.execution_mode == "oracle_execute" and not self.require_gt_diagnostics:
            raise ValueError("oracle_execute requires require_gt_diagnostics=True.")
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.base_checkpoint = Path(base_checkpoint).expanduser().resolve()
        self.expert_checkpoints = {
            int(task): Path(path).expanduser().resolve()
            for task, path in expert_checkpoints.items()
        }
        self.candidate_labels = list(candidate_labels)
        if not self.candidate_labels:
            raise ValueError("Routing-V1 requires at least one candidate expert.")
        if len(set(self.candidate_labels)) != len(self.candidate_labels):
            raise ValueError(f"Routing-V1 candidate list contains duplicates: {self.candidate_labels}")
        for label in self.candidate_labels:
            if label == "base":
                continue
            if not re.fullmatch(r"t[6-9]", label):
                raise ValueError(f"Unsupported candidate label {label!r}; expected base or t6..t9.")
            task = int(label[1:])
            if task not in self.expert_checkpoints:
                raise ValueError(f"Candidate {label} has no checkpoint mapping.")

        template_label = next((x for x in self.candidate_labels if x != "base"), None)
        if template_label is None:
            # Base-only routing is trivial and should normally use the ordinary server,
            # but supporting it makes smoke checks convenient.
            self.template_checkpoint = self.base_checkpoint
        else:
            self.template_checkpoint = self.expert_checkpoints[int(template_label[1:])]

        logging.info("[RoutingV1] loading template checkpoint: %s", self.template_checkpoint)
        policy = baseframework.from_pretrained(str(self.template_checkpoint))
        if use_bf16:
            policy = policy.to(torch.bfloat16)
        policy = policy.to(self.device).eval()
        self.template_policy = policy
        self.config = policy.config
        self.norm_stats = policy.norm_stats
        self.processor = policy.processor
        self.policy_backend = policy.policy_backend
        self.policy_infer_batch_builder = policy.policy_infer_batch_builder
        self._call_module = _BackendRoutingCall(self.policy_backend).to(self.device).eval()
        self._flow_call_module = _FlowRoutingCall(self.policy_backend.flow).to(self.device).eval()

        if not bool(getattr(self.policy_backend.flow.config, "enable_expert_latent_head", False)):
            raise RuntimeError("Routing-V1 template does not enable expert_latent_head.")

        self._template_params = dict(self._call_module.named_parameters(remove_duplicate=True))
        self._task_param_names = [
            name
            for name in self._template_params
            if _is_task_specific(_wrapper_to_ckpt_key(name), self.mode)
        ]
        if not self._task_param_names:
            raise RuntimeError("Routing-V1 found no task-specific parameter names in template model.")

        self._candidate_overrides: dict[str, dict[str, torch.Tensor]] = {}
        self._load_candidate_overrides()
        self._candidate_flow_overrides: dict[str, dict[str, torch.Tensor]] = {
            label: {
                "flow." + name[len("backend.flow.") :]: tensor
                for name, tensor in override.items()
                if name.startswith("backend.flow.")
            }
            for label, override in self._candidate_overrides.items()
        }
        self._audit_protocol()

        self._request_index = 0
        self._episode_chunk_counter: dict[str, int] = defaultdict(int)
        # Per-episode temporal state. Keys are routing_episode_id only; GT task IDs
        # are never used for temporal selection. The evaluator provides stable,
        # unique episode IDs across chunks.
        self._temporal_ema: dict[str, torch.Tensor] = {}
        self._temporal_selected: dict[str, int] = {}
        self._routing_log_path = (
            None if routing_log_path is None else Path(routing_log_path).expanduser().resolve()
        )
        self._routing_log_fp = None
        if self._routing_log_path is not None:
            self._routing_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._routing_log_fp = self._routing_log_path.open("a", encoding="utf-8", buffering=1)
            logging.info("[RoutingV1] chunk routing log: %s", self._routing_log_path)

        self._print_bank_summary()

    def _load_candidate_overrides(self) -> None:
        base_state = _load_state(self.base_checkpoint)
        try:
            for label in self.candidate_labels:
                if label == "base":
                    state = base_state
                else:
                    state = _load_state(self.expert_checkpoints[int(label[1:])])
                override: dict[str, torch.Tensor] = {}
                category_numel: dict[str, int] = defaultdict(int)
                missing: list[str] = []
                for wrapper_name in self._task_param_names:
                    ckpt_key = _wrapper_to_ckpt_key(wrapper_name)
                    template_param = self._template_params[wrapper_name]
                    source = _checkpoint_tensor(state, ckpt_key)
                    if source is not None:
                        pass
                    elif label == "base" and (
                        _is_conditioning_adapter(ckpt_key)
                        or _is_nonlinear_adapter(ckpt_key)
                        or _is_text_lora(ckpt_key)
                    ):
                        # Base was trained before B1/B2 adapters/LoRA were instantiated.
                        # Zero-valued side branches exactly emulate "adapter disabled".
                        source = torch.zeros_like(template_param, device="cpu")
                    else:
                        missing.append(ckpt_key)
                        continue
                    tensor = source.to(
                        device=self.device,
                        dtype=template_param.dtype,
                        non_blocking=False,
                    ).detach()
                    override[wrapper_name] = tensor
                    category_numel[_category(ckpt_key)] += int(tensor.numel())
                if missing:
                    raise RuntimeError(
                        f"Routing-V1 candidate {label} is missing task-specific tensors: {missing[:20]}"
                    )
                self._candidate_overrides[label] = override
                logging.info(
                    "[RoutingV1][bank] %s tensors=%d params=%s categories=%s",
                    label,
                    len(override),
                    f"{sum(x.numel() for x in override.values()):,}",
                    dict(sorted(category_numel.items())),
                )
        finally:
            del base_state

    def _audit_protocol(self) -> None:
        # Strong structural assertions for the two variants.
        selected_keys = [_wrapper_to_ckpt_key(x) for x in self._task_param_names]
        bad_b1 = [k for k in selected_keys if self.mode == "b1" and _is_text_lora(k)]
        if bad_b1:
            raise RuntimeError(f"B1 routing unexpectedly contains Text-LoRA tensors: {bad_b1[:8]}")
        if self.mode == "b2":
            lora_names = [x for x in self._task_param_names if _is_text_lora(_wrapper_to_ckpt_key(x))]
            if not lora_names:
                raise RuntimeError("B2 routing template contains no Text-LoRA parameters.")
            for label in self.candidate_labels:
                if label == "base":
                    continue
                nonzero = sum(
                    int(torch.count_nonzero(self._candidate_overrides[label][name]).item())
                    for name in lora_names
                )
                if nonzero <= 0:
                    raise RuntimeError(f"B2 candidate {label} has all-zero Text-LoRA tensors.")
            if "base" in self.candidate_labels:
                base_nonzero = sum(
                    int(torch.count_nonzero(self._candidate_overrides["base"][name]).item())
                    for name in lora_names
                )
                if base_nonzero != 0:
                    raise RuntimeError("Base emulation in B2 must disable Text-LoRA exactly (all zero).")
        if "base" in self.candidate_labels:
            for name in self._task_param_names:
                key = _wrapper_to_ckpt_key(name)
                if _is_conditioning_adapter(key) or _is_nonlinear_adapter(key):
                    if torch.count_nonzero(self._candidate_overrides["base"][name]).item() != 0:
                        raise RuntimeError(f"Base emulation side adapter is not zero: {key}")
        logging.info(
            "[RoutingV1][CHECK] task-specific namespace audit PASS | mode=%s | keys=%d",
            self.mode,
            len(self._task_param_names),
        )

    def _print_bank_summary(self) -> None:
        flow_cfg = self.policy_backend.flow.config
        logging.info("=" * 72)
        logging.info("[RoutingV1] Expert bank ready")
        logging.info("  mode             : %s", self.mode)
        logging.info("  candidates       : %s", self.candidate_labels)
        logging.info("  score_mode       : %s", self.score_mode)
        logging.info("  alpha            : %.4f", self.alpha)
        logging.info("  score_norm       : %s", self.score_normalization)
        logging.info("  temporal_mode    : %s", self.temporal_mode)
        logging.info("  temporal_beta    : %.4f", self.temporal_beta)
        logging.info("  temporal_margin  : %.4f", self.temporal_margin)
        logging.info("  execution_mode   : %s", self.execution_mode)
        logging.info("  context          : %s", self.context_label)
        logging.info("  require GT diag  : %s (diagnostics only)", self.require_gt_diagnostics)
        logging.info("  z* source        : final denoising forward / final DiT hidden")
        logging.info("  template         : %s", self.template_checkpoint)
        logging.info("  base             : %s", self.base_checkpoint)
        logging.info("  latent_dim       : %s", getattr(flow_cfg, "expert_latent_dim", None))
        logging.info("  inference_steps  : %s", self.num_inference_steps or getattr(flow_cfg, "num_inference_steps", None))
        logging.info("  guidance_scale   : %s", self.guidance_scale or getattr(flow_cfg, "cfg_guidance_scale", None))
        logging.info("  task params      : %s", f"{sum(self._template_params[n].numel() for n in self._task_param_names):,}")
        logging.info("=" * 72)

    def _sanitize_examples(self, examples: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        clean: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        for ex in examples:
            metadata.append({key: ex.get(key) for key in _ROUTING_META_KEYS})
            clean.append({key: value for key, value in ex.items() if key not in _ROUTING_META_KEYS})
        return clean, metadata

    def _sample_shared_noise(self, batch_size: int) -> torch.Tensor:
        flow = self.policy_backend.flow
        dtype = flow._compute_dtype()
        shape = (
            int(batch_size),
            int(flow.action_horizon),
            int(flow.config.action_dim),
        )
        return flow.sample_noise(shape=shape, device=self.device, dtype=dtype)

    def _prepare_b1_shared(self, batch: dict[str, torch.Tensor | None]):
        prepared_batch = self.policy_backend._prepare_infer_batch(batch=batch)
        shared = self.policy_backend._run_shared_encoding_infer(
            prepared_batch=prepared_batch,
            source="RoutingV1ExpertBankPolicy/B1-shared",
            lam_features_with_no_grad=False,
        )
        attn_flow = prepared_batch["attention_mask"] == 1
        return prepared_batch, shared, attn_flow

    def _b1_candidate_forward(
        self,
        label: str,
        prepared_batch: dict[str, torch.Tensor | None],
        shared,
        attn_flow: torch.Tensor,
        initial_noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        flow_out = functional_call(
            self._flow_call_module,
            self._candidate_flow_overrides[label],
            args=(
                shared.h_t,
                shared.h_t1_pred,
                shared.h_vlm,
                prepared_batch["state"],
                prepared_batch["state_mask"],
                prepared_batch["action_hz"],
                prepared_batch["embodiment_id"],
                attn_flow,
                initial_noise,
                self.guidance_scale,
                self.num_inference_steps,
            ),
            tie_weights=True,
            strict=False,
        )
        actions, expert_latent = flow_out
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            expert_future = self.policy_backend._decode_future_tokens_strict_single_query(
                h_t=shared.h_t,
                pred_action_emb=expert_latent,
                source="RoutingV1ExpertBankPolicy/B1-expert-latent",
            )
        return {
            "actions": actions,
            "z_online": shared.pred_action_emb,
            "h_online": shared.h_t1_pred,
            "z_expert": expert_latent,
            "h_expert": expert_future,
            "h_current": shared.h_t,
        }

    def _candidate_forward(
        self,
        label: str,
        batch: dict[str, torch.Tensor | None],
        initial_noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return functional_call(
            self._call_module,
            self._candidate_overrides[label],
            args=(batch, initial_noise, self.guidance_scale, self.num_inference_steps),
            tie_weights=True,
            strict=False,
        )

    def _score_components(self, output: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        dz = _cosine_distance(output["z_online"], output["z_expert"])
        dh = _cosine_distance(output["h_online"], output["h_expert"])
        return dz, dh

    def _build_instant_scores(
        self, metrics: dict[str, dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dz_matrix = torch.stack([metrics[label]["dz"] for label in self.candidate_labels], dim=1)
        dh_matrix = torch.stack([metrics[label]["dh"] for label in self.candidate_labels], dim=1)
        if self.score_normalization == "candidate_mean":
            eps = 1e-8
            dz_used = dz_matrix / dz_matrix.mean(dim=1, keepdim=True).clamp_min(eps)
            dh_used = dh_matrix / dh_matrix.mean(dim=1, keepdim=True).clamp_min(eps)
        else:
            dz_used = dz_matrix
            dh_used = dh_matrix

        if self.score_mode == "latent":
            score_matrix = dz_used
        elif self.score_mode == "world":
            score_matrix = dh_used
        else:
            score_matrix = self.alpha * dz_used + (1.0 - self.alpha) * dh_used
        return dz_matrix, dh_matrix, dz_used, dh_used, score_matrix

    def _temporal_episode_key(self, meta: dict[str, Any], sample_idx: int) -> str:
        episode_id = meta.get("routing_episode_id")
        if episode_id is None:
            if self.temporal_mode != "none":
                raise RuntimeError(
                    "Temporal Routing-V1 requires routing_episode_id metadata so EMA state can be reset "
                    "between episodes without consulting task identity."
                )
            return f"sample:{sample_idx}"
        return str(episode_id)

    def _apply_temporal_stabilization(
        self,
        instant_scores: torch.Tensor,
        metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        instant_selected = torch.argmin(instant_scores, dim=1)
        if self.temporal_mode == "none":
            details = [
                {
                    "episode_key": self._temporal_episode_key(metadata[b], b),
                    "previous_index": None,
                    "proposed_index": int(instant_selected[b].item()),
                    "selected_index": int(instant_selected[b].item()),
                    "switch_improvement": None,
                    "hysteresis_blocked": False,
                }
                for b in range(instant_scores.shape[0])
            ]
            return instant_scores, instant_selected, details

        temporal_rows: list[torch.Tensor] = []
        selected_indices: list[int] = []
        details: list[dict[str, Any]] = []
        for b in range(instant_scores.shape[0]):
            key = self._temporal_episode_key(metadata[b], b)
            current = instant_scores[b].float()
            prev_ema = self._temporal_ema.get(key)
            if prev_ema is None or tuple(prev_ema.shape) != tuple(current.shape):
                ema = current.detach().clone()
            else:
                ema = self.temporal_beta * prev_ema + (1.0 - self.temporal_beta) * current
                ema = ema.detach()
            self._temporal_ema[key] = ema

            proposed = int(torch.argmin(ema).item())
            previous = self._temporal_selected.get(key)
            selected = proposed
            improvement = None
            blocked = False
            if self.temporal_mode == "ema_hysteresis" and previous is not None and proposed != previous:
                improvement = float((ema[previous] - ema[proposed]).item())
                if improvement < self.temporal_margin:
                    selected = previous
                    blocked = True
            self._temporal_selected[key] = selected
            temporal_rows.append(ema)
            selected_indices.append(selected)
            details.append(
                {
                    "episode_key": key,
                    "previous_index": previous,
                    "proposed_index": proposed,
                    "selected_index": selected,
                    "switch_improvement": improvement,
                    "hysteresis_blocked": blocked,
                }
            )

        temporal_matrix = torch.stack(temporal_rows, dim=0).to(instant_scores.device)
        selected_tensor = torch.tensor(selected_indices, device=instant_scores.device, dtype=torch.long)
        return temporal_matrix, selected_tensor, details

    def _expected_label(self, gt_task_id: Any) -> str | None:
        if gt_task_id is None:
            return None
        try:
            task_id = int(gt_task_id)
        except Exception:
            return None
        return "base" if task_id <= 5 else f"t{task_id}"

    def _log_request(
        self,
        metadata: list[dict[str, Any]],
        candidate_metrics: dict[str, dict[str, torch.Tensor]],
        instant_selected_indices: torch.Tensor,
        selected_indices: torch.Tensor,
        executed_indices: torch.Tensor,
        temporal_details: list[dict[str, Any]],
    ) -> None:
        if self._routing_log_fp is None:
            return
        for sample_idx, meta in enumerate(metadata):
            instant_selected_label = self.candidate_labels[int(instant_selected_indices[sample_idx].item())]
            selected_label = self.candidate_labels[int(selected_indices[sample_idx].item())]
            executed_label = self.candidate_labels[int(executed_indices[sample_idx].item())]
            gt_task_id = meta.get("routing_gt_task_id")
            expected = self._expected_label(gt_task_id)
            episode_id = meta.get("routing_episode_id")
            if episode_id is None:
                episode_id = f"slot:{meta.get('routing_slot_id', sample_idx)}|task:{gt_task_id}"
            episode_id = str(episode_id)
            chunk_index = self._episode_chunk_counter[episode_id]
            self._episode_chunk_counter[episode_id] += 1
            row = {
                "context": self.context_label,
                "request_index": int(self._request_index),
                "sample_index": int(sample_idx),
                "episode_id": episode_id,
                "chunk_index": int(chunk_index),
                "slot_id": meta.get("routing_slot_id"),
                "gt_task_id": gt_task_id,
                "expected_expert": expected,
                "selected_expert": selected_label,
                "instant_selected_expert": instant_selected_label,
                "score_selected_expert": instant_selected_label,
                "temporal_selected_expert": selected_label,
                "executed_expert": executed_label,
                "routing_correct": None if expected not in self.candidate_labels else bool(selected_label == expected),
                "instant_routing_correct": None if expected not in self.candidate_labels else bool(instant_selected_label == expected),
                "execution_mode": self.execution_mode,
                "mode": self.mode,
                "score_mode": self.score_mode,
                "alpha": self.alpha,
                "score_normalization": self.score_normalization,
                "temporal_mode": self.temporal_mode,
                "temporal_beta": self.temporal_beta,
                "temporal_margin": self.temporal_margin,
                "temporal_previous_expert": (
                    None
                    if temporal_details[sample_idx]["previous_index"] is None
                    else self.candidate_labels[int(temporal_details[sample_idx]["previous_index"])]
                ),
                "temporal_proposed_expert": self.candidate_labels[int(temporal_details[sample_idx]["proposed_index"])],
                "temporal_switch_improvement": temporal_details[sample_idx]["switch_improvement"],
                "temporal_hysteresis_blocked": bool(temporal_details[sample_idx]["hysteresis_blocked"]),
                "candidates": {
                    label: {
                        "dz": float(candidate_metrics[label]["dz"][sample_idx].item()),
                        "dh": float(candidate_metrics[label]["dh"][sample_idx].item()),
                        "dz_used": float(candidate_metrics[label]["dz_used"][sample_idx].item()),
                        "dh_used": float(candidate_metrics[label]["dh_used"][sample_idx].item()),
                        "score": float(candidate_metrics[label]["score"][sample_idx].item()),
                        "temporal_score": float(candidate_metrics[label]["temporal_score"][sample_idx].item()),
                    }
                    for label in self.candidate_labels
                },
            }
            self._routing_log_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    @torch.inference_mode()
    def predict_action(self, examples: Sequence[dict[str, Any]], **kwargs) -> Dict[str, Any]:
        if not examples:
            raise ValueError("Routing-V1 predict_action requires at least one example.")
        return_intermediates = bool(kwargs.pop("return_intermediates", False))
        if kwargs:
            logging.debug("[RoutingV1] ignored predict kwargs: %s", sorted(kwargs))

        clean_examples, metadata = self._sanitize_examples(examples)
        if self.require_gt_diagnostics:
            for sample_idx, meta in enumerate(metadata):
                if meta.get("routing_gt_task_id") is None:
                    raise RuntimeError(
                        f"Routing-V1 diagnostic wiring failure: sample {sample_idx} has no routing_gt_task_id."
                    )
                expected = self._expected_label(meta.get("routing_gt_task_id"))
                if expected not in self.candidate_labels:
                    raise RuntimeError(
                        "Routing-V1 protocol mismatch: diagnostic expected expert "
                        f"{expected!r} is absent from fixed candidate bank {self.candidate_labels}. "
                        "Task ID is used only for this assertion/logging, never for selection."
                    )
        if self._request_index < self.debug_requests:
            leaked = [
                key
                for ex in clean_examples
                for key in _ROUTING_META_KEYS
                if key in ex
            ]
            if leaked:
                raise RuntimeError(
                    f"Routing diagnostic metadata leaked into policy examples: {sorted(set(leaked))}"
                )
            logging.info(
                "[RoutingV1][CHECK][req=%d] GT/task metadata stripped before build_infer_batch; "
                "model features/scores cannot read task ID. execution_mode=%s",
                self._request_index,
                self.execution_mode,
            )
        batch = self.policy_infer_batch_builder.build_infer_batch(clean_examples)
        batch_size = int(batch["action_hz"].shape[0])
        initial_noise = self._sample_shared_noise(batch_size)

        if self._request_index < self.debug_requests:
            logging.info(
                "[RoutingV1][CHECK][req=%d] common noise shape=%s mean=%.6f std=%.6f",
                self._request_index,
                tuple(initial_noise.shape),
                float(initial_noise.float().mean().item()),
                float(initial_noise.float().std().item()),
            )

        outputs: dict[str, Dict[str, torch.Tensor]] = {}
        metrics: dict[str, dict[str, torch.Tensor]] = {}
        b1_shared_bundle = self._prepare_b1_shared(batch) if self.mode == "b1" else None
        if self.mode == "b1" and self._request_index < self.debug_requests:
            logging.info(
                "[RoutingV1][CHECK][req=%d] B1 shared VLM/QFormer/LaWM computed exactly once; only Flow expert tensors vary.",
                self._request_index,
            )
        for label in self.candidate_labels:
            if self.mode == "b1":
                assert b1_shared_bundle is not None
                prepared_batch, shared, attn_flow = b1_shared_bundle
                output = self._b1_candidate_forward(
                    label, prepared_batch, shared, attn_flow, initial_noise
                )
            else:
                output = self._candidate_forward(label, batch, initial_noise)
            required = {"actions", "z_online", "h_online", "z_expert", "h_expert"}
            missing = required - set(output)
            if missing:
                raise RuntimeError(f"Routing candidate {label} missing outputs {sorted(missing)}")
            for key in required:
                if not torch.isfinite(output[key]).all():
                    raise RuntimeError(f"Routing candidate {label}/{key} contains non-finite values")
            dz, dh = self._score_components(output)
            outputs[label] = output
            metrics[label] = {"dz": dz, "dh": dh}

        dz_matrix, dh_matrix, dz_used, dh_used, score_matrix = self._build_instant_scores(metrics)
        if not torch.isfinite(score_matrix).all():
            raise RuntimeError("Routing-V1 produced non-finite instantaneous scores")
        instant_selected = torch.argmin(score_matrix, dim=1)
        temporal_score_matrix, selected, temporal_details = self._apply_temporal_stabilization(
            score_matrix, metadata
        )
        if not torch.isfinite(temporal_score_matrix).all():
            raise RuntimeError("Routing-V1 produced non-finite temporal scores")
        for idx, label in enumerate(self.candidate_labels):
            metrics[label]["dz_used"] = dz_used[:, idx]
            metrics[label]["dh_used"] = dh_used[:, idx]
            metrics[label]["score"] = score_matrix[:, idx]
            metrics[label]["temporal_score"] = temporal_score_matrix[:, idx]

        # Diagnostic oracle-execution mode keeps the robot on the ground-truth
        # expert's trajectory while leaving ALL feature extraction and routing
        # scores task-agnostic. GT task metadata is consulted only here, after
        # every candidate score has already been computed.
        if self.execution_mode == "oracle_execute":
            if not self.require_gt_diagnostics:
                raise RuntimeError(
                    "oracle_execute requires --require_gt_diagnostics so the expected expert is available."
                )
            executed_list = []
            for sample_idx, meta in enumerate(metadata):
                expected = self._expected_label(meta.get("routing_gt_task_id"))
                if expected not in self.candidate_labels:
                    raise RuntimeError(
                        f"oracle_execute expected {expected!r}, absent from candidates {self.candidate_labels}."
                    )
                executed_list.append(self.candidate_labels.index(expected))
            executed = torch.tensor(executed_list, device=selected.device, dtype=selected.dtype)
        else:
            executed = selected

        selected_actions = torch.stack(
            [outputs[self.candidate_labels[int(executed[b].item())]]["actions"][b] for b in range(batch_size)],
            dim=0,
        )

        if self._request_index < self.debug_requests:
            # B1 must have exactly shared z/h across every candidate. B2 intentionally
            # permits branch-specific Text-LoRA to change z/h.
            ref = outputs[self.candidate_labels[0]]
            for label in self.candidate_labels[1:]:
                z_delta = float((outputs[label]["z_online"] - ref["z_online"]).float().abs().max().item())
                h_delta = float((outputs[label]["h_online"] - ref["h_online"]).float().abs().max().item())
                if self.mode == "b1" and (z_delta != 0.0 or h_delta != 0.0):
                    raise RuntimeError(
                        "B1 routing invariant violated: shared z/h changed across candidates "
                        f"({self.candidate_labels[0]} vs {label}: max|dz|={z_delta}, max|dh|={h_delta})."
                    )
                logging.info(
                    "[RoutingV1][CHECK][req=%d] upstream %s vs %s max_abs_z=%.8g max_abs_h=%.8g",
                    self._request_index,
                    self.candidate_labels[0],
                    label,
                    z_delta,
                    h_delta,
                )
            if self.mode == "b2" and len(self.candidate_labels) > 1:
                any_branch_delta = any(
                    float((outputs[label]["z_online"] - ref["z_online"]).float().abs().max().item()) > 0.0
                    or float((outputs[label]["h_online"] - ref["h_online"]).float().abs().max().item()) > 0.0
                    for label in self.candidate_labels[1:]
                )
                if not any_branch_delta:
                    logging.warning(
                        "[RoutingV1][CHECK][req=%d] B2 Text-LoRA branches produced identical z/h on this request; "
                        "weights are audited as nonzero, but verify branch injection if this persists.",
                        self._request_index,
                    )
            for sample_idx in range(batch_size):
                score_text = " | ".join(
                    f"{label}:dz={float(metrics[label]['dz'][sample_idx]):.5f},"
                    f"dh={float(metrics[label]['dh'][sample_idx]):.5f},"
                    f"dzN={float(metrics[label]['dz_used'][sample_idx]):.5f},"
                    f"dhN={float(metrics[label]['dh_used'][sample_idx]):.5f},"
                    f"S={float(metrics[label]['score'][sample_idx]):.5f},"
                    f"Sema={float(metrics[label]['temporal_score'][sample_idx]):.5f}"
                    for label in self.candidate_labels
                )
                detail = temporal_details[sample_idx]
                logging.info(
                    "[RoutingV1][SCORE][req=%d sample=%d gt=%s] %s -> instant=%s temporal=%s executed=%s prev=%s blocked=%s",
                    self._request_index,
                    sample_idx,
                    metadata[sample_idx].get("routing_gt_task_id"),
                    score_text,
                    self.candidate_labels[int(instant_selected[sample_idx].item())],
                    self.candidate_labels[int(selected[sample_idx].item())],
                    self.candidate_labels[int(executed[sample_idx].item())],
                    (None if detail["previous_index"] is None else self.candidate_labels[int(detail["previous_index"])]),
                    detail["hysteresis_blocked"],
                )
            logging.info(
                "[RoutingV1][CHECK][req=%d] shapes action=%s z=%s h=%s z*=%s h*=%s",
                self._request_index,
                tuple(ref["actions"].shape),
                tuple(ref["z_online"].shape),
                tuple(ref["h_online"].shape),
                tuple(ref["z_expert"].shape),
                tuple(ref["h_expert"].shape),
            )

        self._log_request(
            metadata, metrics, instant_selected, selected, executed, temporal_details
        )
        self._request_index += 1

        intermediates = None
        if return_intermediates:
            # Keep compatibility with similarity visualization by returning the
            # selected candidate's world tokens for each sample.
            h_t = torch.stack(
                [outputs[self.candidate_labels[int(executed[b].item())]]["h_current"][b] for b in range(batch_size)],
                dim=0,
            )
            h_pred = torch.stack(
                [outputs[self.candidate_labels[int(executed[b].item())]]["h_online"][b] for b in range(batch_size)],
                dim=0,
            )
            num_tokens = int(h_t.shape[1])
            hw = int(math.sqrt(num_tokens))
            if hw * hw != num_tokens:
                hw = 16
            intermediates = {
                "h_t": h_t.detach().cpu(),
                "h_t1_pred": h_pred.detach().cpu(),
                "vision_tokens_hw": torch.tensor([[hw, hw]] * batch_size, dtype=torch.long),
            }
        return map_policy_infer_output(selected_actions, intermediates=intermediates)
