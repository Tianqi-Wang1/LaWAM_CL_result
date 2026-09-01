#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Analyze raw dz/dh scores collected while GT experts were executed. "
            "No rollout is repeated; multiple score formulas are reconstructed offline."
        )
    )
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--combined-alpha", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1e-8)
    return p.parse_args()


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse {path}:{line_no}: {exc}") from exc
            if row.get("expected_expert") not in row.get("candidates", {}):
                raise RuntimeError(
                    f"Row {line_no} expected_expert={row.get('expected_expert')!r} "
                    f"not in candidates={list(row.get('candidates', {}))}"
                )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Empty routing log: {path}")
    return rows


def scores_for_row(row, method: str, alpha: float, eps: float):
    items = row["candidates"]
    labels = list(items)
    dz = {k: float(items[k]["dz"]) for k in labels}
    dh = {k: float(items[k]["dh"]) for k in labels}
    if method == "latent":
        score = dz
    elif method == "world":
        score = dh
    elif method == "combined_raw":
        score = {k: alpha * dz[k] + (1.0 - alpha) * dh[k] for k in labels}
    elif method == "combined_mean_norm":
        mean_z = max(fmean(dz.values()), eps)
        mean_h = max(fmean(dh.values()), eps)
        score = {
            k: alpha * (dz[k] / mean_z) + (1.0 - alpha) * (dh[k] / mean_h)
            for k in labels
        }
    else:
        raise ValueError(method)
    return score, dz, dh


def evaluate(rows, method, alpha, eps):
    correct = 0
    margins = []
    selections = Counter()
    by_task = defaultdict(lambda: {"n": 0, "correct": 0, "margins": [], "sel": Counter()})
    confusion = Counter()
    all_dz = []
    all_dh = []
    correct_dz = []
    wrong_dz = []
    correct_dh = []
    wrong_dh = []

    for row in rows:
        expected = row["expected_expert"]
        score, dz, dh = scores_for_row(row, method, alpha, eps)
        selected = min(score, key=score.get)
        wrong = [v for k, v in score.items() if k != expected]
        margin = min(wrong) - score[expected] if wrong else math.nan
        ok = selected == expected
        correct += int(ok)
        selections[selected] += 1
        task = int(row["gt_task_id"])
        stat = by_task[task]
        stat["n"] += 1
        stat["correct"] += int(ok)
        stat["sel"][selected] += 1
        if not math.isnan(margin):
            stat["margins"].append(margin)
            margins.append(margin)
        confusion[(expected, selected)] += 1

        for k in score:
            all_dz.append(dz[k])
            all_dh.append(dh[k])
            if k == expected:
                correct_dz.append(dz[k])
                correct_dh.append(dh[k])
            else:
                wrong_dz.append(dz[k])
                wrong_dh.append(dh[k])

    labels = sorted({k for r in rows for k in r["candidates"]})
    k = len(labels)
    chance = 1.0 / k
    acc = correct / len(rows)
    norm_acc = (acc - chance) / (1.0 - chance) if chance < 1.0 else 0.0
    result = {
        "method": method,
        "alpha": alpha,
        "decisions": len(rows),
        "num_candidates": k,
        "accuracy": acc,
        "chance_accuracy": chance,
        "chance_normalized_accuracy": norm_acc,
        "mean_margin": fmean(margins) if margins else math.nan,
        "positive_margin_rate": (sum(x > 0 for x in margins) / len(margins)) if margins else math.nan,
        "selection_counts": dict(selections),
        "mean_dz_all": fmean(all_dz),
        "mean_dh_all": fmean(all_dh),
        "dz_to_dh_scale_ratio": fmean(all_dz) / max(fmean(all_dh), eps),
        "mean_correct_dz": fmean(correct_dz),
        "mean_wrong_dz": fmean(wrong_dz),
        "mean_correct_dh": fmean(correct_dh),
        "mean_wrong_dh": fmean(wrong_dh),
    }
    task_rows = []
    for task in sorted(by_task):
        st = by_task[task]
        vals = st["margins"]
        task_rows.append({
            "method": method,
            "alpha": alpha,
            "task_id": task,
            "decisions": st["n"],
            "accuracy": st["correct"] / st["n"],
            "mean_margin": fmean(vals) if vals else math.nan,
            "positive_margin_rate": (sum(x > 0 for x in vals) / len(vals)) if vals else math.nan,
            "selection_counts": json.dumps(dict(st["sel"]), sort_keys=True),
        })
    return result, task_rows, confusion, labels


def safe(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def main():
    a = parse_args()
    if not 0.0 <= a.combined_alpha <= 1.0:
        raise ValueError("--combined-alpha must be in [0,1]")
    rows = load_rows(a.log)
    modes = ["latent", "world", "combined_raw", "combined_mean_norm"]
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_task_rows = []
    confusion_by_method = {}
    labels = None
    for method in modes:
        result, task_rows, confusion, method_labels = evaluate(
            rows, method, a.combined_alpha, a.eps
        )
        all_results.append(result)
        all_task_rows.extend(task_rows)
        confusion_by_method[method] = confusion
        labels = method_labels

    with (out / "oracle_probe_methods.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "method", "alpha", "decisions", "num_candidates", "accuracy", "chance_accuracy",
            "chance_normalized_accuracy", "mean_margin", "positive_margin_rate",
            "mean_dz_all", "mean_dh_all", "dz_to_dh_scale_ratio",
            "mean_correct_dz", "mean_wrong_dz", "mean_correct_dh", "mean_wrong_dh",
            "selection_counts",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            rr = dict(r)
            rr["selection_counts"] = json.dumps(rr["selection_counts"], sort_keys=True)
            w.writerow(rr)

    with (out / "oracle_probe_per_task.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["method", "alpha", "task_id", "decisions", "accuracy", "mean_margin", "positive_margin_rate", "selection_counts"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_task_rows)

    # Alpha sweep uses the mean-normalized fusion because raw fusion is dominated by dz scale.
    sweep = []
    for alpha in [i / 10 for i in range(11)]:
        result, _, _, _ = evaluate(rows, "combined_mean_norm", alpha, a.eps)
        sweep.append(result)
    with (out / "oracle_probe_normalized_alpha_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["alpha", "accuracy", "chance_accuracy", "chance_normalized_accuracy", "mean_margin", "positive_margin_rate"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sweep:
            w.writerow({k: r[k] for k in fields})

    assert labels is not None
    for method, confusion in confusion_by_method.items():
        with (out / f"confusion_{method}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["expected\\selected", *labels])
            for expected in labels:
                w.writerow([expected, *[confusion[(expected, selected)] for selected in labels]])

    payload = {
        "log": str(a.log),
        "execution_modes": sorted({str(r.get("execution_mode")) for r in rows}),
        "contexts": sorted({str(r.get("context")) for r in rows}),
        "methods": [{k: safe(v) for k, v in r.items()} for r in all_results],
    }
    (out / "oracle_probe_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[RoutingV1 oracle-state score probe]")
    for r in all_results:
        print(
            f"  {r['method']:>20s}: acc={r['accuracy']:.4f} chance={r['chance_accuracy']:.4f} "
            f"norm={r['chance_normalized_accuracy']:.4f} margin={r['mean_margin']:.6f} "
            f"pos={r['positive_margin_rate']:.4f}"
        )
    best = max(all_results, key=lambda x: x["accuracy"])
    print(f"  best fixed method: {best['method']} acc={best['accuracy']:.4f}")
    print(f"  dz/dh raw scale ratio: {all_results[0]['dz_to_dh_scale_ratio']:.3f}x")
    print(f"  output: {out}")


if __name__ == "__main__":
    main()
