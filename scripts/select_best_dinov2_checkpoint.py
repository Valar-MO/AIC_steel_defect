#!/usr/bin/env python3
"""Evaluate checkpoints with tiled validation and select the official-metric best."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--dinov2-weights", type=Path, required=True)
    parser.add_argument("--rtdetr-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=("map50_95", "map50", "recall", "f1", "f2"),
                        default="map50_95")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--allow-partial-metrics", action="store_true",
                        help="Allow non-comparable smoke metrics when limiting images.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.limit_images and not args.allow_partial_metrics:
        raise ValueError("--limit-images is smoke-only; also pass --allow-partial-metrics explicitly")
    checkpoints = [path.resolve() for path in args.checkpoints]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in checkpoints:
        stem = checkpoint.stem
        raw_path = args.output_dir / f"{stem}_predictions.json"
        metrics_dir = args.output_dir / f"{stem}_metrics"
        prediction_command = [
            sys.executable, "scripts/predict_dinov2_rtdetr_tiles.py",
            "--checkpoint", str(checkpoint),
            "--dinov2-weights", str(args.dinov2_weights),
            "--rtdetr-weights", str(args.rtdetr_weights),
            "--output", str(raw_path),
            "--tile-size", str(args.tile_size),
            "--overlap", str(args.overlap),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--conf", "0.001",
        ]
        if args.limit_images:
            prediction_command += ["--limit-images", str(args.limit_images)]
        if args.overwrite or not raw_path.is_file():
            prediction_command.append("--overwrite")
            run(prediction_command)
        evaluation_command = [
            sys.executable, "scripts/evaluate_predictions.py",
            "--predictions", str(raw_path),
            "--score-threshold", str(args.score_threshold),
            "--output-dir", str(metrics_dir),
            "--overwrite",
        ]
        if args.limit_images:
            evaluation_command.append("--allow-missing-images")
        run(evaluation_command)
        report = json.loads((metrics_dir / "evaluation_report.json").read_text(encoding="utf-8"))
        checkpoint_payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
        row = {
            "checkpoint": str(checkpoint),
            "epoch": int(checkpoint_payload["epoch"]) + 1,
            **report["overall"],
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    rows.sort(key=lambda row: row[args.metric], reverse=True)
    winner = rows[0]
    prefix = "smoke_best" if args.limit_images else "best"
    best_path = args.output_dir / f"{prefix}_{args.metric}.pt"
    shutil.copy2(winner["checkpoint"], best_path)
    summary = {"selection_metric": args.metric, "partial_metrics": bool(args.limit_images),
               "winner": winner, "ranked": rows,
               "best_checkpoint": str(best_path)}
    (args.output_dir / "checkpoint_ranking.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
