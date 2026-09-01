#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

FINAL_REL = Path("final_model") / "pytorch_model.pt"
SIDECARS = ("config.yaml", "dataset_statistics.json")
CANON_PREFIX = "policy_backend.flow."
ALIAS_PREFIX = "policy_action_head."


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compose a LaWAM checkpoint using all shared/non-Flow tensors from "
            "--shared-run and BOTH Flow namespaces from --flow-run."
        )
    )
    p.add_argument("--shared-run", type=Path, required=True)
    p.add_argument("--flow-run", type=Path, required=True)
    p.add_argument("--output-run", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify-output", action="store_true")
    return p.parse_args()


def resolve_run(run: Path, label: str):
    run = run.expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"{label} run missing: {run}")
    ckpt = run / FINAL_REL
    if not ckpt.is_file():
        raise FileNotFoundError(f"{label} checkpoint missing: {ckpt}")
    sidecars = {}
    for name in SIDECARS:
        p = run / name
        if not p.is_file():
            raise FileNotFoundError(f"{label} sidecar missing: {p}")
        sidecars[name] = p
    return run, ckpt, sidecars


def load_state(path: Path) -> dict[str, Any]:
    kwargs = dict(map_location="cpu")
    try:
        obj = torch.load(path, weights_only=True, mmap=True, **kwargs)
    except TypeError:
        try:
            obj = torch.load(path, weights_only=True, **kwargs)
        except TypeError:
            obj = torch.load(path, **kwargs)

    if isinstance(obj, dict):
        for wrapper in ("state_dict", "model", "module"):
            nested = obj.get(wrapper, None)
            if isinstance(nested, dict) and nested and any(torch.is_tensor(v) for v in nested.values()):
                obj = nested
                break

    if not isinstance(obj, dict):
        raise RuntimeError(f"Checkpoint is not a state dict: {path}")
    return obj


def keys_with_prefix(state, prefix):
    return sorted(
        k for k, v in state.items()
        if k.startswith(prefix) and torch.is_tensor(v)
    )


def suffix_map(state, prefix):
    return {
        k[len(prefix):]: k
        for k in keys_with_prefix(state, prefix)
    }


def assert_namespace_compatible(shared, donor, prefix, label):
    s = keys_with_prefix(shared, prefix)
    d = keys_with_prefix(donor, prefix)
    if not s or not d:
        raise RuntimeError(
            f"{label}: required Flow namespace {prefix!r} missing; "
            f"shared={len(s)} donor={len(d)}"
        )
    if s != d:
        raise RuntimeError(
            f"{label}: namespace key mismatch for {prefix!r}; "
            f"missing_in_donor={sorted(set(s)-set(d))[:8]}, "
            f"extra_in_donor={sorted(set(d)-set(s))[:8]}"
        )
    for key in s:
        a, b = shared[key], donor[key]
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            raise RuntimeError(
                f"{label}: shape/dtype mismatch at {key}: "
                f"{tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}"
            )
    return s


def assert_internal_alias_consistency(state, label):
    canon = suffix_map(state, CANON_PREFIX)
    alias = suffix_map(state, ALIAS_PREFIX)
    if not canon or not alias:
        raise RuntimeError(
            f"{label}: both Flow namespaces are mandatory for this diagnostic; "
            f"canonical={len(canon)}, alias={len(alias)}"
        )
    if set(canon) != set(alias):
        raise RuntimeError(
            f"{label}: canonical/alias suffix sets differ; "
            f"canonical_only={sorted(set(canon)-set(alias))[:8]}, "
            f"alias_only={sorted(set(alias)-set(canon))[:8]}"
        )

    unequal = []
    for suffix in sorted(canon):
        a = state[canon[suffix]]
        b = state[alias[suffix]]
        if (
            tuple(a.shape) != tuple(b.shape)
            or a.dtype != b.dtype
            or not torch.equal(a, b)
        ):
            unequal.append(suffix)

    if unequal:
        raise RuntimeError(
            f"{label}: canonical and alias Flow tensors are not identical; "
            f"examples={unequal[:12]}"
        )

    return len(canon), len(alias)


def atomic_save(state, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".pytorch_model.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp = Path(tmp)
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def verify_saved(output_ckpt: Path, donor):
    saved = load_state(output_ckpt)

    for prefix in (CANON_PREFIX, ALIAS_PREFIX):
        donor_keys = keys_with_prefix(donor, prefix)
        saved_keys = keys_with_prefix(saved, prefix)
        if donor_keys != saved_keys:
            raise RuntimeError(
                f"Saved namespace mismatch for {prefix!r}: "
                f"donor={len(donor_keys)}, saved={len(saved_keys)}"
            )
        changed = []
        for key in donor_keys:
            if not torch.equal(saved[key], donor[key]):
                changed.append(key)
        if changed:
            raise RuntimeError(
                f"Saved composed checkpoint does not match donor for "
                f"{prefix!r}; examples={changed[:10]}"
            )

    assert_internal_alias_consistency(saved, "saved output")
    del saved
    gc.collect()


def main():
    args = parse_args()

    shared_run, shared_ckpt, shared_sidecars = resolve_run(args.shared_run, "shared")
    donor_run, donor_ckpt, _ = resolve_run(args.flow_run, "flow donor")
    output_run = args.output_run.expanduser().resolve()

    if output_run in (shared_run, donor_run):
        raise RuntimeError("--output-run must differ from source runs")

    if output_run.exists():
        if not args.overwrite:
            raise FileExistsError(output_run)
        shutil.rmtree(output_run)

    output_run.mkdir(parents=True, exist_ok=True)

    print("[compose-v2] loading shared:", shared_ckpt)
    shared = load_state(shared_ckpt)
    print("[compose-v2] loading donor :", donor_ckpt)
    donor = load_state(donor_ckpt)

    canonical = assert_namespace_compatible(
        shared, donor, CANON_PREFIX, "compose"
    )
    alias = assert_namespace_compatible(
        shared, donor, ALIAS_PREFIX, "compose"
    )

    sc, sa = assert_internal_alias_consistency(shared, "shared")
    dc, da = assert_internal_alias_consistency(donor, "donor")

    composed = dict(shared)
    for key in canonical:
        composed[key] = donor[key]
    for key in alias:
        composed[key] = donor[key]

    for name, src in shared_sidecars.items():
        shutil.copy2(src, output_run / name)

    output_ckpt = output_run / FINAL_REL

    print(
        f"[compose-v2] replacing canonical={len(canonical)} + "
        f"alias={len(alias)} Flow tensors"
    )
    print(
        "[compose-v2] policy_backend.flow.* AND "
        "policy_action_head.* come from donor"
    )
    atomic_save(composed, output_ckpt)

    meta = {
        "type": "lawam_oracle_flow_composition_alias_aware_v2",
        "shared_run": str(shared_run),
        "flow_run": str(donor_run),
        "output_run": str(output_run),
        "canonical_prefix": CANON_PREFIX,
        "alias_prefix": ALIAS_PREFIX,
        "canonical_tensor_count": len(canonical),
        "alias_tensor_count": len(alias),
        "shared_alias_consistent": True,
        "donor_alias_consistent": True,
        "note": (
            "Both policy_backend.flow.* and policy_action_head.* were "
            "replaced from the Flow donor. flow_action_query remains shared."
        ),
    }
    (output_run / "composition.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.verify_output:
        print("[compose-v2] verifying both saved Flow namespaces...")
        verify_saved(output_ckpt, donor)
        print("[OK] saved canonical + alias Flow are bitwise donor-identical.")

    print("[OK] alias-aware composed run:", output_run)

    del shared, donor, composed
    gc.collect()


if __name__ == "__main__":
    main()