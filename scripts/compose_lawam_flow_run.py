#!/usr/bin/env python3
"""
Compose a LaWAM run directory by taking all shared/non-Flow parameters from one
run and replacing ONLY `policy_backend.flow.*` with the Flow head from another
run.

This is intended for the Oracle Flow-Head Isolation diagnostic:
  output = Shared(current CL stage) + Historical Flow(task introduction stage)

Important:
- `policy_backend.flow_action_query` is NOT replaced. It remains part of the
  shared continually learned upstream/interface state.
- `config.yaml` and `dataset_statistics.json` are copied from the shared run,
  because LaWAM evaluation expects these sidecars next to the checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


DEFAULT_FLOW_PREFIX = "policy_backend.flow."
FINAL_REL = Path("final_model") / "pytorch_model.pt"
SIDECARS = ("config.yaml", "dataset_statistics.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compose a complete LaWAM run: all non-Flow weights from --shared-run "
            "plus policy_backend.flow.* from --flow-run."
        )
    )
    p.add_argument("--shared-run", type=Path, required=True)
    p.add_argument("--flow-run", type=Path, required=True)
    p.add_argument("--output-run", type=Path, required=True)
    p.add_argument("--flow-prefix", default=DEFAULT_FLOW_PREFIX)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output run before composing.",
    )
    p.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip conservative free-space check before writing.",
    )
    p.add_argument(
        "--verify-output",
        action="store_true",
        help="Re-open the saved checkpoint and verify Flow tensors against donor.",
    )
    return p.parse_args()


def resolve_run(run: Path, label: str) -> tuple[Path, dict[str, Path]]:
    run = run.expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"{label} run does not exist: {run}")

    ckpt = run / FINAL_REL
    if not ckpt.is_file():
        raise FileNotFoundError(f"{label} checkpoint missing: {ckpt}")

    sidecars: dict[str, Path] = {}
    for name in SIDECARS:
        p = run / name
        if not p.is_file():
            raise FileNotFoundError(f"{label} sidecar missing: {p}")
        sidecars[name] = p

    return ckpt, sidecars


def torch_load_state(path: Path) -> dict[str, Any]:
    """
    Prefer mmap + weights_only to avoid materializing two full 2B checkpoints
    in anonymous RAM. Fall back for older PyTorch versions.
    """
    kwargs = {"map_location": "cpu"}

    try:
        obj = torch.load(
            path,
            weights_only=True,
            mmap=True,
            **kwargs,
        )
    except TypeError:
        try:
            obj = torch.load(
                path,
                weights_only=True,
                **kwargs,
            )
        except TypeError:
            obj = torch.load(path, **kwargs)

    if isinstance(obj, dict):
        # LaWAM final checkpoints are plain state dicts, but accept common wrappers
        # defensively.
        for wrapper in ("state_dict", "model", "module"):
            nested = obj.get(wrapper)
            if (
                isinstance(nested, dict)
                and nested
                and any(torch.is_tensor(v) for v in nested.values())
            ):
                obj = nested
                break

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Checkpoint is not a state dict: {path} "
            f"(got {type(obj).__name__})"
        )

    return obj


def flow_keys(state: dict[str, Any], prefix: str) -> list[str]:
    return sorted(
        k
        for k, v in state.items()
        if k.startswith(prefix) and torch.is_tensor(v)
    )


def validate_compatibility(
    shared: dict[str, Any],
    donor: dict[str, Any],
    prefix: str,
) -> list[str]:
    shared_flow = flow_keys(shared, prefix)
    donor_flow = flow_keys(donor, prefix)

    if not shared_flow:
        raise RuntimeError(
            f"No Flow tensors found in shared checkpoint with prefix {prefix!r}"
        )
    if not donor_flow:
        raise RuntimeError(
            f"No Flow tensors found in donor checkpoint with prefix {prefix!r}"
        )

    if shared_flow != donor_flow:
        missing_in_donor = sorted(set(shared_flow) - set(donor_flow))
        extra_in_donor = sorted(set(donor_flow) - set(shared_flow))
        raise RuntimeError(
            "Flow key sets are incompatible.\n"
            f"  missing in donor: {missing_in_donor[:12]}\n"
            f"  extra in donor  : {extra_in_donor[:12]}"
        )

    bad = []
    for key in shared_flow:
        a = shared[key]
        b = donor[key]
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            bad.append(
                {
                    "key": key,
                    "shared_shape": tuple(a.shape),
                    "flow_shape": tuple(b.shape),
                    "shared_dtype": str(a.dtype),
                    "flow_dtype": str(b.dtype),
                }
            )

    if bad:
        raise RuntimeError(
            "Flow tensor shape/dtype mismatch. Examples:\n"
            + "\n".join(str(x) for x in bad[:10])
        )

    return shared_flow


def conservative_disk_check(
    output_parent: Path,
    shared_ckpt: Path,
) -> None:
    usage = shutil.disk_usage(output_parent)
    source_size = shared_ckpt.stat().st_size

    # torch.save output should be close to source checkpoint size.
    # Require source size + 1 GiB temporary margin.
    required = source_size + (1 << 30)
    if usage.free < required:
        raise RuntimeError(
            "Insufficient free disk space for composed checkpoint.\n"
            f"  free     : {usage.free / (1<<30):.2f} GiB\n"
            f"  required : {required / (1<<30):.2f} GiB\n"
            f"  parent   : {output_parent}"
        )


def atomic_save_state(state: dict[str, Any], output_ckpt: Path) -> None:
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=".pytorch_model.",
        suffix=".tmp",
        dir=str(output_ckpt.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, output_ckpt)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def verify_saved_flow(
    output_ckpt: Path,
    donor: dict[str, Any],
    keys: list[str],
    prefix: str,
) -> None:
    saved = torch_load_state(output_ckpt)

    saved_keys = flow_keys(saved, prefix)
    if set(saved_keys) != set(keys):
        raise RuntimeError(
            "Saved checkpoint Flow key set changed unexpectedly."
        )

    changed = []
    for key in keys:
        a = saved[key]
        b = donor[key]
        if (
            tuple(a.shape) != tuple(b.shape)
            or a.dtype != b.dtype
            or not torch.equal(a, b)
        ):
            changed.append(key)

    if changed:
        raise RuntimeError(
            "Saved composed Flow does not exactly equal donor Flow. "
            f"Examples: {changed[:10]}"
        )

    del saved
    gc.collect()


def main() -> int:
    args = parse_args()

    shared_run = args.shared_run.expanduser().resolve()
    flow_run = args.flow_run.expanduser().resolve()
    output_run = args.output_run.expanduser().resolve()

    if output_run in (shared_run, flow_run):
        raise RuntimeError(
            "--output-run must be different from both source run directories."
        )

    shared_ckpt, shared_sidecars = resolve_run(shared_run, "shared")
    flow_ckpt, _ = resolve_run(flow_run, "flow")

    if output_run.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output run already exists: {output_run}\n"
                "Use --overwrite if it is a disposable composition directory."
            )
        shutil.rmtree(output_run)

    output_run.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_disk_check:
        conservative_disk_check(output_run.parent, shared_ckpt)

    print("[compose] loading shared checkpoint:")
    print(f"          {shared_ckpt}")
    shared_state = torch_load_state(shared_ckpt)

    print("[compose] loading Flow donor checkpoint:")
    print(f"          {flow_ckpt}")
    donor_state = torch_load_state(flow_ckpt)

    keys = validate_compatibility(
        shared_state,
        donor_state,
        args.flow_prefix,
    )

    # Shallow dict copy: tensors remain mmap-backed where supported.
    composed = dict(shared_state)
    for key in keys:
        composed[key] = donor_state[key]

    output_ckpt = output_run / FINAL_REL
    output_run.mkdir(parents=True, exist_ok=True)

    # Evaluation needs the shared stage's config/statistics.
    for name, src in shared_sidecars.items():
        shutil.copy2(src, output_run / name)

    print(
        f"[compose] replacing {len(keys)} Flow tensors "
        f"with prefix {args.flow_prefix!r}"
    )
    print(f"[compose] writing: {output_ckpt}")
    atomic_save_state(composed, output_ckpt)

    meta = {
        "type": "lawam_oracle_flow_composition",
        "shared_run": str(shared_run),
        "shared_checkpoint": str(shared_ckpt),
        "flow_run": str(flow_run),
        "flow_checkpoint": str(flow_ckpt),
        "output_run": str(output_run),
        "output_checkpoint": str(output_ckpt),
        "flow_prefix": args.flow_prefix,
        "flow_tensor_count": len(keys),
        "shared_sidecars": [str(shared_sidecars[x]) for x in SIDECARS],
        "note": (
            "Only policy_backend.flow.* was replaced. "
            "policy_backend.flow_action_query remains from shared_run."
        ),
    }
    (output_run / "composition.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.verify_output:
        print("[compose] verifying saved Flow against donor...")
        verify_saved_flow(output_ckpt, donor_state, keys, args.flow_prefix)
        print("[OK] saved Flow is bitwise identical to donor Flow.")

    print("[OK] composed LaWAM run ready:")
    print(f"     {output_run}")
    print(f"     {output_ckpt}")

    del composed, shared_state, donor_state
    gc.collect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
