#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_dynamic_state(state: dict) -> tuple[dict, str]:
    prefixes = (
        "policy_backend.routing_v2_memory.dynamics.",
        "routing_v2_memory.dynamics.",
        "dynamics.",
    )
    for prefix in prefixes:
        out = {
            k[len(prefix):]: v
            for k, v in state.items()
            if isinstance(k, str) and k.startswith(prefix) and torch.is_tensor(v)
        }
        if out:
            return out, prefix
    # Already a plain SpatialDynamicsAutoencoder state dict.
    if "input_proj.weight" in state and "delta_decoder.weight" in state:
        out = {k: v for k, v in state.items() if torch.is_tensor(v)}
        return out, "<plain_dynamics_state>"
    raise RuntimeError("No Routing-V2 Dynamics-AE tensors found in source state dict")


def infer_input_mode(dyn: dict) -> str:
    input_w = dyn["input_proj.weight"]
    delta_w = dyn["delta_decoder.weight"]
    vision = int(delta_w.shape[0])
    in_dim = int(input_w.shape[1])
    z_keys = [k for k in dyn if k.startswith("z_decoder.")]
    if z_keys:
        latent = int(dyn["z_decoder.1.weight"].shape[0])
        if in_dim != 2 * vision + latent:
            raise RuntimeError(
                f"hdhz geometry mismatch: input={in_dim}, vision={vision}, latent={latent}"
            )
        return "hdhz"
    if in_dim == 2 * vision:
        return "hdh"
    if in_dim == vision:
        return "dh"
    raise RuntimeError(f"Cannot infer input mode: input_dim={in_dim}, vision_dim={vision}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--wm-source", choices=["task", "base"], required=True)
    ap.add_argument("--input-mode", choices=["hdhz", "hdh", "dh"], required=True)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    state = torch_load(source)
    if not isinstance(state, dict):
        raise RuntimeError(f"Expected state dict in {source}")
    dyn, prefix = extract_dynamic_state(state)
    inferred = infer_input_mode(dyn)
    if inferred != args.input_mode:
        raise RuntimeError(
            f"Requested input_mode={args.input_mode}, but extracted tensors imply {inferred}"
        )

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dyn, out)
    meta = {
        "variant": str(args.variant),
        "wm_source": str(args.wm_source),
        "input_mode": str(args.input_mode),
        "source": str(source),
        "source_prefix": prefix,
        "num_tensors": len(dyn),
        "num_params": int(sum(v.numel() for v in dyn.values())),
    }
    if args.config is not None:
        meta["config"] = str(args.config.expanduser().resolve())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[OK] Dynamics variant extracted: {out} "
        f"variant={args.variant} mode={args.input_mode} wm={args.wm_source} "
        f"tensors={meta['num_tensors']} params={meta['num_params']:,}"
    )


if __name__ == "__main__":
    main()
