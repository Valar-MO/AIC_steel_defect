#!/usr/bin/env python3
"""Re-run official whole/tile fusion from raw candidates at each score threshold.

Unlike evaluating an already-thresholded ``fused.json``, this applies the
threshold before class-wise NMS for every point.  It therefore measures the
actual Recall/FP tradeoff available in the raw candidate pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_THRESHOLDS = (0.001, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
WEAK_CLASSES = ("huashang", "qilie", "zonglie", "yanghuatiepi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--nms-iou", type=float, default=0.80)
    parser.add_argument("--border-factor", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    if args.split != "val":
        raise ValueError("This diagnostic is validation-only; use --split val")
    if not args.whole.is_file() or not args.tiles.is_file():
        raise FileNotFoundError("Raw whole/tile prediction JSON is required")
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    project = Path(__file__).resolve().parents[1]
    merge = project / "scripts" / "merge_predictions.py"
    evaluate = project / "scripts" / "evaluate_predictions.py"
    audit = project / "scripts" / "analyze_dfine_proposal_precision.py"
    rows: list[dict[str, object]] = []

    for threshold in sorted(set(args.thresholds)):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid threshold: {threshold}")
        tag = f"t{threshold:.3f}".replace(".", "p")
        point_dir = output_dir / tag
        fused = point_dir / "fused.json"
        evaluation_dir = point_dir / "evaluation"
        audit_dir = point_dir / "audit"
        point_dir.mkdir(parents=True, exist_ok=True)
        overwrite = ["--overwrite"] if args.overwrite else []
        run([
            sys.executable, str(merge), "--whole", str(args.whole), "--tiles", str(args.tiles),
            "--classes", str(args.classes), "--min-score", str(threshold),
            "--nms-iou", str(args.nms_iou), "--tile-border-policy", "penalize",
            "--tile-border-score-factor", str(args.border_factor), "--max-det", str(args.max_det),
            "--output", str(fused), *overwrite,
        ])
        run([
            sys.executable, str(evaluate), "--predictions", str(fused), "--manifest", str(args.manifest),
            "--classes", str(args.classes), "--split", args.split, "--score-threshold", str(threshold),
            "--output-dir", str(evaluation_dir), *overwrite,
        ])
        run([
            sys.executable, str(audit), "--predictions", str(fused), "--manifest", str(args.manifest),
            "--classes", str(args.classes), "--split", args.split, "--match-iou", "0.5",
            "--near-iou", "0.3", "--background-iou", "0.1", "--score-thresholds", str(threshold),
            "--output-dir", str(audit_dir), *overwrite,
        ])

        overall = read_json(evaluation_dir / "evaluation_report.json")["overall"]
        audit_summary = read_json(audit_dir / "summary.json")["cumulative_by_threshold"][0]
        report = read_json(fused.with_name("fused_report.json"))
        row: dict[str, object] = {
            "threshold": threshold,
            "tp": overall["tp"], "fp": overall["fp"], "fn": overall["fn"],
            "precision": overall["precision"], "recall": overall["recall"],
            "f1": overall["f1"], "f2": overall["f2"],
            "ap50": overall["map50"], "ap50_95": overall["map50_95"],
            "prediction_count": overall["prediction_count"],
            "pure_background_fp": audit_summary["background"],
            "duplicate": audit_summary["duplicate"], "near_miss": audit_summary["near_miss"],
            "weak_overlap": audit_summary["weak_overlap"],
            "threshold_dropped": report["processing_counts"]["threshold_dropped"],
        }
        per_class = read_json(evaluation_dir / "evaluation_report.json")["per_class"]
        for name in WEAK_CLASSES:
            row[f"recall_{name}"] = per_class[name]["recall"]
            row[f"tp_{name}"] = per_class[name]["tp"]
        rows.append(row)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
