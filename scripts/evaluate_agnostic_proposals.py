#!/usr/bin/env python3
"""Evaluate class-agnostic proposal coverage by source class and object size."""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--score-thresholds", nargs="+", type=float,
                        default=[0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1])
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.3, 0.5])
    parser.add_argument("--max-proposals-per-image", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter > 0 else 0.0


def size_bucket(box, image_area: float) -> str:
    relative = ((box[2] - box[0]) * (box[3] - box[1])) / image_area
    if relative < 0.001:
        return "tiny_lt_0.1pct"
    if relative < 0.01:
        return "small_0.1_to_1pct"
    if relative < 0.10:
        return "medium_1_to_10pct"
    return "large_ge_10pct"


def load_gt(manifest: Path, split: str, classes: list[str]):
    class_set = set(classes)
    rows = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    for manifest_row in manifest_rows:
        root = ET.parse(manifest_row["xml_path"]).getroot()
        width, height = int(float(root.findtext("size/width"))), int(float(root.findtext("size/height")))
        seen = set()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in class_set:
                raise ValueError(f"Unknown GT class {name!r}")
            node = obj.find("bndbox")
            raw = tuple(float(node.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
            signature = (name, *raw)
            if signature in seen or not all(math.isfinite(v) for v in raw):
                continue
            seen.add(signature)
            box = (
                max(0.0, min(raw[0], width)), max(0.0, min(raw[1], height)),
                max(0.0, min(raw[2], width)), max(0.0, min(raw[3], height)),
            )
            if box[2] > box[0] and box[3] > box[1]:
                rows.append({
                    "image_id": Path(manifest_row["image_path"]).name, "class_name": name,
                    "box": box, "size_bucket": size_bucket(box, width * height),
                })
    return rows, {Path(row["image_path"]).name for row in manifest_rows}


def load_predictions(path: Path, expected_images: set[str]):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("metadata", {}).get("classes") not in (None, ["defect"]):
        raise ValueError("Expected class-agnostic predictions with class 'defect'")
    records = {}
    for record in payload["images"]:
        image_id = Path(record["image_id"]).name
        if image_id not in expected_images or image_id in records:
            raise ValueError(f"Unexpected or duplicate prediction image: {image_id}")
        proposals = []
        for pred in record.get("predictions", []):
            score, box = float(pred["score"]), tuple(float(v) for v in pred["bbox_xyxy"])
            if math.isfinite(score) and len(box) == 4 and all(math.isfinite(v) for v in box):
                if box[2] > box[0] and box[3] > box[1]:
                    proposals.append((score, box))
        records[image_id] = sorted(proposals, reverse=True)
    missing = expected_images - set(records)
    if missing:
        raise ValueError(f"Predictions miss {len(missing)} images")
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classes = yaml.safe_load(args.classes.read_text(encoding="utf-8"))["names"]
    gt_rows, image_ids = load_gt(args.manifest, args.split, classes)
    predictions = load_predictions(args.predictions, image_ids)
    summary_rows, group_rows, detail_rows = [], [], []
    for score_threshold in sorted(set(args.score_thresholds)):
        best_ious = []
        proposal_counts = []
        for image_id in image_ids:
            kept = [item for item in predictions[image_id] if item[0] >= score_threshold]
            kept = kept[:args.max_proposals_per_image]
            proposal_counts.append(len(kept))
        for gt_index, gt in enumerate(gt_rows):
            kept = [item for item in predictions[gt["image_id"]] if item[0] >= score_threshold]
            kept = kept[:args.max_proposals_per_image]
            best = max((iou(gt["box"], box) for _, box in kept), default=0.0)
            best_ious.append(best)
            detail_rows.append({
                "score_threshold": score_threshold, "gt_index": gt_index,
                "image_id": gt["image_id"], "class_name": gt["class_name"],
                "size_bucket": gt["size_bucket"], "best_iou": round(best, 8),
            })
        for iou_threshold in sorted(set(args.iou_thresholds)):
            covered = sum(value >= iou_threshold for value in best_ious)
            summary_rows.append({
                "score_threshold": score_threshold, "iou_threshold": iou_threshold,
                "gt_count": len(gt_rows), "covered": covered,
                "recall": round(covered / len(gt_rows), 8),
                "avg_proposals_per_image": round(sum(proposal_counts) / len(proposal_counts), 8),
            })
            for group_type, values in (
                ("class", classes),
                ("size", ["tiny_lt_0.1pct", "small_0.1_to_1pct", "medium_1_to_10pct", "large_ge_10pct"]),
            ):
                for value in values:
                    indices = [
                        index for index, gt in enumerate(gt_rows)
                        if gt["class_name" if group_type == "class" else "size_bucket"] == value
                    ]
                    group_covered = sum(best_ious[index] >= iou_threshold for index in indices)
                    group_rows.append({
                        "score_threshold": score_threshold, "iou_threshold": iou_threshold,
                        "group_type": group_type, "group": value, "gt_count": len(indices),
                        "covered": group_covered,
                        "recall": round(group_covered / len(indices), 8) if indices else 0.0,
                    })
    write_csv(args.output_dir / "coverage_summary.csv", summary_rows)
    write_csv(args.output_dir / "coverage_by_group.csv", group_rows)
    write_csv(args.output_dir / "gt_best_iou.csv", detail_rows)
    print(json.dumps({"gt_count": len(gt_rows), "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
