#!/usr/bin/env python3
"""Extract compact Semantic/Dynamics AE snapshots from memory-phase checkpoints."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import torch


PREFIX = "policy_backend.routing_v2_memory."


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return dict(obj)
    if isinstance(obj, dict):
        for wrapper in ("state_dict", "model", "module"):
            inner = obj.get(wrapper)
            if isinstance(inner, dict) and inner:
                return {key: value for key, value in inner.items() if torch.is_tensor(value)}
    raise RuntimeError(f"Unsupported checkpoint structure: {path}")


def source_for_step(run: Path, step: int, max_step: int) -> Path:
    periodic = run / "checkpoints" / f"steps_{step}_pytorch_model.pt"
    if periodic.is_file():
        return periodic
    final = run / "final_model" / "pytorch_model.pt"
    if step == max_step and final.is_file():
        return final
    raise FileNotFoundError(
        f"No full checkpoint for AE step {step}: expected {periodic}"
        + (f" or {final}" if step == max_step else "")
    )


def strip_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    memory = {
        key[len(PREFIX):]: value
        for key, value in state.items()
        if key.startswith(PREFIX)
    }
    if not memory:
        raise RuntimeError(f"No {PREFIX} tensors found")
    return memory


def component(memory: Dict[str, torch.Tensor], name: str) -> Dict[str, torch.Tensor]:
    prefix = f"{name}."
    out = {
        key[len(prefix):]: value
        for key, value in memory.items()
        if key.startswith(prefix)
    }
    if not out:
        raise RuntimeError(f"No {name} AE tensors found")
    return out


def validate_b2_memory(semantic: Dict[str, torch.Tensor], dynamics: Dict[str, torch.Tensor]) -> None:
    if "encoder.0.weight" not in semantic or "decoder.weight" not in semantic:
        raise RuntimeError("Semantic AE has unexpected tensor structure")
    required = {"input_proj.weight", "pos_embed", "delta_decoder.weight"}
    missing = sorted(required - set(dynamics))
    if missing:
        raise RuntimeError(f"Dynamics AE is missing tensors: {missing}")
    if any(key.startswith("z_decoder.") for key in dynamics):
        raise RuntimeError("Expected B2 [h, Delta h] Dynamics AE, but z_decoder is present")
    input_dim = int(dynamics["input_proj.weight"].shape[1])
    vision_dim = int(dynamics["delta_decoder.weight"].shape[0])
    if input_dim != 2 * vision_dim:
        raise RuntimeError(
            f"Expected [h, Delta h] input geometry: input={input_dim}, vision={vision_dim}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--max-step", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--latest-step", type=int, default=None)
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    config = args.config.expanduser().resolve()
    steps = list(dict.fromkeys(int(step) for step in args.steps))
    if not steps or any(step <= 0 or step > args.max_step for step in steps):
        raise ValueError(f"Invalid snapshot steps={steps}, max_step={args.max_step}")
    latest_step = int(args.latest_step if args.latest_step is not None else max(steps))
    if latest_step not in steps:
        raise ValueError(f"latest_step={latest_step} is not in steps={steps}")
    if not config.is_file():
        raise FileNotFoundError(config)

    manifest = {
        "run": str(run),
        "config": str(config),
        "max_step": int(args.max_step),
        "latest_step": latest_step,
        "snapshots": [],
    }
    for step in steps:
        source = source_for_step(run, step, args.max_step)
        memory = strip_prefix(load_state(source))
        semantic = component(memory, "semantic")
        dynamics = component(memory, "dynamics")
        validate_b2_memory(semantic, dynamics)

        destination = output_root / f"step_{step}"
        destination.mkdir(parents=True, exist_ok=True)
        combined_path = destination / "routing_memory.pt"
        semantic_path = destination / "semantic_ae.pt"
        dynamics_path = destination / "dynamics_ae.pt"
        torch.save(memory, combined_path)
        torch.save(semantic, semantic_path)
        torch.save(dynamics, dynamics_path)
        shutil.copy2(config, destination / "memory_train_config.yaml")

        metadata = {
            "step": step,
            "source_full_checkpoint": str(source),
            "skill_path": "vlm_text_lora_plus_action_b2",
            "semantic_source": "shared_base_vlm_and_base_queries",
            "dynamics_source": "task_vlm_text_lora_plus_shared_qformer_plus_shared_base_wm",
            "dynamics_input_mode": "hdh",
            "semantic_params": int(sum(value.numel() for value in semantic.values())),
            "dynamics_params": int(sum(value.numel() for value in dynamics.values())),
            "total_params": int(sum(value.numel() for value in memory.values())),
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        manifest["snapshots"].append(metadata)
        print(
            f"[OK] AE step {step}: semantic={metadata['semantic_params']:,}, "
            f"dynamics={metadata['dynamics_params']:,}, output={destination}"
        )

    latest = output_root / f"step_{latest_step}"
    # Small compatibility copies for current evaluation scripts.
    for filename in ("routing_memory.pt", "semantic_ae.pt", "dynamics_ae.pt"):
        shutil.copy2(latest / filename, output_root / filename)
    (output_root / "latest_step.txt").write_text(f"{latest_step}\n", encoding="utf-8")
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[OK] Latest AE compatibility files point to extracted step {latest_step}")


if __name__ == "__main__":
    main()
