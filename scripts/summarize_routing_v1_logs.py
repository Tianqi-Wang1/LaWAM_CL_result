#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def stage_key(value: str | None) -> tuple[int, str]:
    text = "" if value is None else str(value)
    if text == "Base":
        return (0, text)
    if text.startswith("CL"):
        try:
            return (int(text[2:]), text)
        except ValueError:
            pass
    return (999, text)


def safe_mean(vals: list[float]) -> float:
    return fmean(vals) if vals else math.nan


def normalized_accuracy(acc: float, chance: float) -> float:
    if math.isnan(acc) or chance >= 1.0:
        return math.nan
    return (acc - chance) / (1.0 - chance)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = []
    with args.log.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Failed to parse {args.log}:{line_no}: {exc}") from exc
    if not rows:
        raise RuntimeError(f"Routing log is empty: {args.log}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    known = [r for r in rows if r.get("routing_correct") is not None]
    accuracy = (sum(bool(r["routing_correct"]) for r in known) / len(known)) if known else math.nan
    instant_known = [r for r in rows if r.get("instant_routing_correct") is not None]
    instant_accuracy = (
        sum(bool(r["instant_routing_correct"]) for r in instant_known) / len(instant_known)
        if instant_known else accuracy
    )

    by_task = defaultdict(list)
    by_context = defaultdict(list)
    by_context_task = defaultdict(list)
    selection = Counter()
    confusion = Counter()
    episodes = defaultdict(list)

    for row in rows:
        selected = str(row["selected_expert"])
        selection[selected] += 1
        gt = row.get("gt_task_id")
        context = str(row.get("context") or "")
        expected = row.get("expected_expert")
        if gt is not None:
            by_task[int(gt)].append(row)
            by_context_task[(context, int(gt))].append(row)
        by_context[context].append(row)
        if expected is not None:
            confusion[(str(expected), selected)] += 1
        episode_id = str(row.get("episode_id", "unknown"))
        episode_key = f"{context}::{episode_id}" if context else episode_id
        episodes[episode_key].append(row)

    def row_margin(row: dict, field: str = "score") -> float | None:
        expected = row.get("expected_expert")
        candidates = row.get("candidates", {})
        if expected not in candidates or len(candidates) <= 1:
            return None
        if field not in candidates[expected]:
            return None
        correct_score = float(candidates[expected][field])
        wrong_scores = [float(v[field]) for k, v in candidates.items() if k != expected and field in v]
        if not wrong_scores:
            return None
        return min(wrong_scores) - correct_score

    per_task_rows = []
    for task in sorted(by_task):
        task_rows = by_task[task]
        valid = [x for x in task_rows if x.get("routing_correct") is not None]
        margins = [m for m in (row_margin(x, "score") for x in task_rows) if m is not None]
        temporal_margins = [m for m in (row_margin(x, "temporal_score") for x in task_rows) if m is not None]
        instant_valid = [x for x in task_rows if x.get("instant_routing_correct") is not None]
        per_task_rows.append({
            "task_id": task,
            "routing_decisions": len(task_rows),
            "instant_routing_accuracy": (sum(bool(x["instant_routing_correct"]) for x in instant_valid) / len(instant_valid) if instant_valid else math.nan),
            "routing_accuracy": sum(bool(x["routing_correct"]) for x in valid) / len(valid) if valid else math.nan,
            "mean_instant_margin": safe_mean(margins),
            "mean_temporal_margin": safe_mean(temporal_margins),
            "positive_instant_margin_rate": sum(v > 0 for v in margins) / len(margins) if margins else math.nan,
            "positive_temporal_margin_rate": sum(v > 0 for v in temporal_margins) / len(temporal_margins) if temporal_margins else math.nan,
        })

    context_rows = []
    for context in sorted(by_context, key=stage_key):
        ctx_rows = by_context[context]
        valid = [x for x in ctx_rows if x.get("routing_correct") is not None]
        candidate_counts = [len(x.get("candidates", {})) for x in ctx_rows]
        unique_counts = sorted(set(candidate_counts))
        if len(unique_counts) != 1:
            raise RuntimeError(f"Context {context} has inconsistent candidate counts: {unique_counts}")
        k = unique_counts[0] if unique_counts else 0
        acc = sum(bool(x["routing_correct"]) for x in valid) / len(valid) if valid else math.nan
        instant_valid = [x for x in ctx_rows if x.get("instant_routing_correct") is not None]
        instant_acc = (sum(bool(x["instant_routing_correct"]) for x in instant_valid) / len(instant_valid) if instant_valid else math.nan)
        chance = (1.0 / k) if k > 0 else math.nan
        margins = [m for m in (row_margin(x, "score") for x in ctx_rows) if m is not None]
        temporal_margins = [m for m in (row_margin(x, "temporal_score") for x in ctx_rows) if m is not None]
        context_rows.append({
            "context": context,
            "num_candidates": k,
            "routing_decisions": len(ctx_rows),
            "instant_routing_accuracy": instant_acc,
            "routing_accuracy": acc,
            "temporal_accuracy_gain": (acc - instant_acc) if not math.isnan(acc) and not math.isnan(instant_acc) else math.nan,
            "chance_accuracy": chance,
            "chance_normalized_accuracy": normalized_accuracy(acc, chance),
            "mean_instant_margin": safe_mean(margins),
            "mean_temporal_margin": safe_mean(temporal_margins),
            "positive_instant_margin_rate": sum(v > 0 for v in margins) / len(margins) if margins else math.nan,
            "positive_temporal_margin_rate": sum(v > 0 for v in temporal_margins) / len(temporal_margins) if temporal_margins else math.nan,
        })

    context_task_rows = []
    for (context, task), task_rows in sorted(by_context_task.items(), key=lambda kv: (stage_key(kv[0][0]), kv[0][1])):
        valid = [x for x in task_rows if x.get("routing_correct") is not None]
        margins = [m for m in (row_margin(x, "score") for x in task_rows) if m is not None]
        temporal_margins = [m for m in (row_margin(x, "temporal_score") for x in task_rows) if m is not None]
        k_values = sorted({len(x.get("candidates", {})) for x in task_rows})
        k = k_values[0] if len(k_values) == 1 else 0
        acc = sum(bool(x["routing_correct"]) for x in valid) / len(valid) if valid else math.nan
        instant_valid = [x for x in task_rows if x.get("instant_routing_correct") is not None]
        instant_acc = (sum(bool(x["instant_routing_correct"]) for x in instant_valid) / len(instant_valid) if instant_valid else math.nan)
        chance = (1.0 / k) if k > 0 else math.nan
        context_task_rows.append({
            "context": context,
            "task_id": task,
            "num_candidates": k,
            "routing_decisions": len(task_rows),
            "instant_routing_accuracy": instant_acc,
            "routing_accuracy": acc,
            "temporal_accuracy_gain": (acc - instant_acc) if not math.isnan(acc) and not math.isnan(instant_acc) else math.nan,
            "chance_accuracy": chance,
            "chance_normalized_accuracy": normalized_accuracy(acc, chance),
            "mean_instant_margin": safe_mean(margins),
            "mean_temporal_margin": safe_mean(temporal_margins),
            "positive_instant_margin_rate": sum(v > 0 for v in margins) / len(margins) if margins else math.nan,
            "positive_temporal_margin_rate": sum(v > 0 for v in temporal_margins) / len(temporal_margins) if temporal_margins else math.nan,
        })

    episode_rows = []
    total_switches = 0
    total_transitions = 0
    total_instant_switches = 0
    total_instant_transitions = 0
    for episode_id, ep_rows in sorted(episodes.items()):
        ep_rows = sorted(ep_rows, key=lambda r: (int(r.get("chunk_index", 0)), int(r.get("request_index", 0))))
        selections = [str(r["selected_expert"]) for r in ep_rows]
        instant_selections = [str(r.get("instant_selected_expert", r["selected_expert"])) for r in ep_rows]
        switches = sum(a != b for a, b in zip(selections, selections[1:]))
        instant_switches = sum(a != b for a, b in zip(instant_selections, instant_selections[1:]))
        transitions = max(0, len(selections) - 1)
        total_switches += switches
        total_transitions += transitions
        total_instant_switches += instant_switches
        total_instant_transitions += transitions
        valid = [r for r in ep_rows if r.get("routing_correct") is not None]
        instant_valid = [r for r in ep_rows if r.get("instant_routing_correct") is not None]
        episode_rows.append({
            "episode_id": episode_id,
            "context": ep_rows[0].get("context"),
            "gt_task_id": ep_rows[0].get("gt_task_id"),
            "routing_decisions": len(ep_rows),
            "instant_switches": instant_switches,
            "instant_switch_rate": instant_switches / transitions if transitions else 0.0,
            "switches": switches,
            "switch_rate": switches / transitions if transitions else 0.0,
            "instant_routing_accuracy": (sum(bool(r["instant_routing_correct"]) for r in instant_valid) / len(instant_valid) if instant_valid else math.nan),
            "routing_accuracy": sum(bool(r["routing_correct"]) for r in valid) / len(valid) if valid else math.nan,
            "instant_selection_sequence": "|".join(instant_selections),
            "selection_sequence": "|".join(selections),
        })

    labels = sorted(set([x for pair in confusion for x in pair] + list(selection)))
    confusion_path = out / "routing_confusion.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["expected\\selected", *labels])
        for expected in labels:
            writer.writerow([expected, *[confusion[(expected, selected)] for selected in labels]])

    per_task_path = out / "routing_per_task.csv"
    write_csv(per_task_path, ["task_id", "routing_decisions", "instant_routing_accuracy", "routing_accuracy", "mean_instant_margin", "mean_temporal_margin", "positive_instant_margin_rate", "positive_temporal_margin_rate"], per_task_rows)

    context_path = out / "routing_per_context.csv"
    write_csv(context_path, ["context", "num_candidates", "routing_decisions", "instant_routing_accuracy", "routing_accuracy", "temporal_accuracy_gain", "chance_accuracy", "chance_normalized_accuracy", "mean_instant_margin", "mean_temporal_margin", "positive_instant_margin_rate", "positive_temporal_margin_rate"], context_rows)

    context_task_path = out / "routing_per_context_task.csv"
    write_csv(context_task_path, ["context", "task_id", "num_candidates", "routing_decisions", "instant_routing_accuracy", "routing_accuracy", "temporal_accuracy_gain", "chance_accuracy", "chance_normalized_accuracy", "mean_instant_margin", "mean_temporal_margin", "positive_instant_margin_rate", "positive_temporal_margin_rate"], context_task_rows)

    episode_path = out / "routing_per_episode.csv"
    write_csv(episode_path, ["episode_id", "context", "gt_task_id", "routing_decisions", "instant_switches", "instant_switch_rate", "switches", "switch_rate", "instant_routing_accuracy", "routing_accuracy", "instant_selection_sequence", "selection_sequence"], episode_rows)

    candidate_sets = []
    seen_candidate_sets = set()
    for row in rows:
        candidate_set = tuple(row.get("candidates", {}).keys())
        if candidate_set not in seen_candidate_sets:
            seen_candidate_sets.add(candidate_set)
            candidate_sets.append(list(candidate_set))
    contexts = sorted({str(r.get("context")) for r in rows if r.get("context") not in (None, "")}, key=stage_key)

    final_context = max(contexts, key=stage_key) if contexts else None
    final_rows = by_context.get(final_context, []) if final_context is not None else []
    final_valid = [r for r in final_rows if r.get("routing_correct") is not None]
    final_bank_accuracy = sum(bool(r["routing_correct"]) for r in final_valid) / len(final_valid) if final_valid else math.nan
    final_instant_valid = [r for r in final_rows if r.get("instant_routing_correct") is not None]
    final_bank_instant_accuracy = (
        sum(bool(r["instant_routing_correct"]) for r in final_instant_valid) / len(final_instant_valid)
        if final_instant_valid else math.nan
    )
    final_k = len(final_rows[0].get("candidates", {})) if final_rows else 0
    final_chance = 1.0 / final_k if final_k > 0 else math.nan

    task_accs = [x["routing_accuracy"] for x in per_task_rows if not math.isnan(x["routing_accuracy"])]
    summary = {
        "num_routing_decisions": len(rows),
        "num_labeled_decisions": len(known),
        "instant_routing_accuracy_micro_all_stages": instant_accuracy,
        "routing_accuracy_micro_all_stages": accuracy,
        "temporal_accuracy_gain_micro_all_stages": accuracy - instant_accuracy if not math.isnan(accuracy) and not math.isnan(instant_accuracy) else math.nan,
        "routing_accuracy_macro_task_all_stages": safe_mean(task_accs),
        "instant_expert_switch_rate": total_instant_switches / total_instant_transitions if total_instant_transitions else 0.0,
        "expert_switch_rate": total_switches / total_transitions if total_transitions else 0.0,
        "final_context": final_context,
        "final_bank_instant_routing_accuracy": final_bank_instant_accuracy,
        "final_bank_routing_accuracy": final_bank_accuracy,
        "final_bank_temporal_accuracy_gain": final_bank_accuracy - final_bank_instant_accuracy if not math.isnan(final_bank_accuracy) and not math.isnan(final_bank_instant_accuracy) else math.nan,
        "final_bank_num_candidates": final_k,
        "final_bank_chance_accuracy": final_chance,
        "final_bank_chance_normalized_accuracy": normalized_accuracy(final_bank_accuracy, final_chance),
        "selection_counts": dict(selection),
        "mode": rows[0].get("mode"),
        "score_mode": rows[0].get("score_mode"),
        "alpha": rows[0].get("alpha"),
        "score_normalization": rows[0].get("score_normalization", "none"),
        "temporal_mode": rows[0].get("temporal_mode", "none"),
        "temporal_beta": rows[0].get("temporal_beta", 0.0),
        "temporal_margin": rows[0].get("temporal_margin", 0.0),
        "candidates": sorted({k for r in rows for k in r.get("candidates", {})}),
        "candidate_sets": candidate_sets,
        "contexts": contexts,
        "per_task_file": str(per_task_path),
        "per_context_file": str(context_path),
        "per_context_task_file": str(context_task_path),
        "per_episode_file": str(episode_path),
        "confusion_file": str(confusion_path),
    }
    summary_path = out / "routing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")

    print("[RoutingV1 summary]")
    print(f"  decisions             : {len(rows)}")
    print(f"  instant/final micro acc: {instant_accuracy:.4f}/{accuracy:.4f}" if not math.isnan(accuracy) else "  micro acc (all stages): n/a")
    print(f"  macro task acc        : {summary['routing_accuracy_macro_task_all_stages']:.4f}" if not math.isnan(summary["routing_accuracy_macro_task_all_stages"]) else "  macro task acc        : n/a")
    print(f"  instant/final switch : {summary['instant_expert_switch_rate']:.4f}/{summary['expert_switch_rate']:.4f}")
    print(f"  final bank ({final_context}) instant/final/chance: {final_bank_instant_accuracy:.4f}/{final_bank_accuracy:.4f}/{final_chance:.4f}" if not math.isnan(final_bank_accuracy) else "  final bank acc        : n/a")
    print(f"  selections            : {dict(selection)}")
    print("  per-context:")
    for row in context_rows:
        acc = row["routing_accuracy"]
        print(
            f"    {row['context']}: K={row['num_candidates']} instant/final={row['instant_routing_accuracy']:.4f}/{acc:.4f} "
            f"chance={row['chance_accuracy']:.4f} norm={row['chance_normalized_accuracy']:.4f} "
            f"marginI/T={row['mean_instant_margin']:.6f}/{row['mean_temporal_margin']:.6f}"
        )
    print(f"  summary               : {summary_path}")


if __name__ == "__main__":
    main()
