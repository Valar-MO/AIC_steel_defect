#!/usr/bin/env python3
"""Summarize cached-query H0/H1/H2 classification and detection metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


WEAK_CLASSES = ("qilie", "huashang", "yiwuyaru")
CONFUSION_PAIRS = (("qilie", "jieba"), ("yiwuyaru", "jieba"))


def mean_std(values):
    values = [float(value) for value in values]
    return {
        "mean": sum(values) / len(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def confusion_count(metrics, true_name, predicted_name):
    return next((
        row["count"] for row in metrics["top_confusions"]
        if row["true"] == true_name and row["predicted"] == predicted_name
    ), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    classification = json.loads(
        (args.root / "heads" / "classification_report.json").read_text(encoding="utf-8")
    )
    baseline_detection = json.loads(
        (args.root / "baseline_evaluation" / "evaluation_report.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    grouped = defaultdict(list)
    for run in classification["runs"]:
        name = f"{run['variant']}_seed{run['seed']}"
        detection = json.loads(
            (args.root / "evaluations" / name / "evaluation_report.json").read_text(
                encoding="utf-8"
            )
        )
        cls = run["classification_metrics"]
        overall = detection["overall"]
        row = {
            "name": name,
            "variant": run["variant"],
            "seed": run["seed"],
            "best_epoch": run["best_epoch"],
            "classification_accuracy": cls["accuracy"],
            "classification_macro_recall": cls["macro_recall"],
            "classification_macro_f1": cls["macro_f1"],
            "tp": overall["tp"],
            "fp": overall["fp"],
            "fn": overall["fn"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "f2": overall["f2"],
            "map50": overall["map50"],
            "map50_95": overall["map50_95"],
            "weak_recall": {
                class_name: detection["per_class"][class_name]["recall"]
                for class_name in WEAK_CLASSES
            },
            "weak_tp": {
                class_name: detection["per_class"][class_name]["tp"]
                for class_name in WEAK_CLASSES
            },
            "target_confusions": {
                f"{true_name}->{predicted_name}": confusion_count(
                    cls, true_name, predicted_name
                )
                for true_name, predicted_name in CONFUSION_PAIRS
            },
        }
        rows.append(row)
        grouped[run["variant"]].append(row)

    aggregate = {}
    for variant, variant_rows in grouped.items():
        aggregate[variant] = {
            metric: mean_std(row[metric] for row in variant_rows)
            for metric in (
                "classification_accuracy", "classification_macro_recall",
                "classification_macro_f1", "tp", "fp", "precision", "recall",
                "f1", "f2", "map50", "map50_95",
            )
        }
        aggregate[variant]["weak_recall"] = {
            class_name: mean_std(row["weak_recall"][class_name] for row in variant_rows)
            for class_name in WEAK_CLASSES
        }
        aggregate[variant]["target_confusions"] = {
            f"{true_name}->{predicted_name}": mean_std(
                row["target_confusions"][f"{true_name}->{predicted_name}"]
                for row in variant_rows
            )
            for true_name, predicted_name in CONFUSION_PAIRS
        }

    report = {
        "baseline_classification": classification["detector_baseline"],
        "baseline_detection": baseline_detection["overall"],
        "runs": rows,
        "aggregate": aggregate,
    }
    output = args.output or args.root / "summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
