#!/usr/bin/env python3
"""Describe weak-class GT failures across raw and official fused candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_predictions import box_iou, load_classes, load_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True,
                        help="Official whole/tile candidates after .001 NMS.")
    parser.add_argument("--gt-diagnostics", type=Path, required=True,
                        help="Output CSV from diagnose_weak_classes.py.")
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--target-classes", nargs="+", default=["huashang", "qilie"])
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def geometry_bucket(area_ratio: float, aspect: float) -> tuple[str, str]:
    area = "tiny" if area_ratio < 0.001 else "small" if area_ratio < 0.01 else "medium" if area_ratio < 0.1 else "large"
    shape = "compact" if aspect < 2 else "elongated" if aspect < 5 else "line" if aspect < 10 else "extreme_line"
    return area, shape


def best(candidates: list[dict], gt_box: list[float], class_id: int | None, *, by_score: bool):
    matched = [item for item in candidates if (class_id is None or item["class_id"] == class_id)
               and box_iou(gt_box, item["bbox_xyxy"]) >= 0.5]
    if not matched:
        return None
    if by_score:
        return max(matched, key=lambda item: item["score"])
    return max(matched, key=lambda item: box_iou(gt_box, item["bbox_xyxy"]))


def candidate_fields(prefix: str, item: dict | None, gt_box: list[float], classes: list[str]) -> dict:
    if item is None:
        return {f"{prefix}_{name}": "" for name in ("class", "score", "iou", "source", "border")}
    return {
        f"{prefix}_class": classes[item["class_id"]],
        f"{prefix}_score": round(float(item["score"]), 6),
        f"{prefix}_iou": round(float(box_iou(gt_box, item["bbox_xyxy"])), 6),
        f"{prefix}_source": item.get("source", ""),
        f"{prefix}_border": bool(item.get("touches_internal_border", False)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(args.classes)
    class_to_id = {name: index for index, name in enumerate(classes)}
    gt, sizes, _ = load_ground_truth(args.manifest, args.split, class_to_id)
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    candidates = {item["image_id"]: item["predictions"] for item in raw["images"]}
    diagnostics = list(csv.DictReader(args.gt_diagnostics.open(encoding="utf-8-sig", newline="")))

    rows: list[dict] = []
    summary: dict[str, Counter] = defaultdict(Counter)
    for item in diagnostics:
        name = item["true_class"]
        if name not in args.target_classes:
            continue
        image_id, gt_index = item["image_id"], int(item["gt_index"])
        class_id = class_to_id[name]
        gt_box = list(gt[class_id][image_id][gt_index])
        width, height = sizes[image_id]
        box_w, box_h = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
        area_ratio = box_w * box_h / (width * height)
        aspect = max(box_w / max(box_h, 1e-6), box_h / max(box_w, 1e-6))
        area_bucket, shape_bucket = geometry_bucket(area_ratio, aspect)
        image_candidates = candidates.get(image_id, [])
        same_score = best(image_candidates, gt_box, class_id, by_score=True)
        any_score = best(image_candidates, gt_box, None, by_score=True)
        same_iou = best(image_candidates, gt_box, class_id, by_score=False)
        row = {
            "image_id": image_id,
            "true_class": name,
            "gt_index": gt_index,
            "status": item["status"],
            "confusion_label": item["confusion_label"],
            "area_ratio": round(area_ratio, 8),
            "area_bucket": area_bucket,
            "aspect_ratio": round(aspect, 4),
            "shape_bucket": shape_bucket,
            **candidate_fields("same_score_match", same_score, gt_box, classes),
            **candidate_fields("any_score_match", any_score, gt_box, classes),
            **candidate_fields("same_best_iou", same_iou, gt_box, classes),
        }
        rows.append(row)
        prefix = name
        summary[prefix][f"status:{row['status']}"] += 1
        summary[prefix][f"area:{area_bucket}|status:{row['status']}"] += 1
        summary[prefix][f"shape:{shape_bucket}|status:{row['status']}"] += 1
        summary[prefix][f"same_source:{row['same_score_match_source'] or 'none'}|status:{row['status']}"] += 1
        if row["same_score_match_score"] != "":
            score = float(row["same_score_match_score"])
            band = "lt_0.05" if score < .05 else "0.05_0.10" if score < .10 else "0.10_0.20" if score < .20 else "0.20_0.30" if score < .30 else "ge_0.30"
            summary[prefix][f"same_score_band:{band}|status:{row['status']}"] += 1

    fields = list(rows[0]) if rows else []
    with (args.output_dir / "lifecycle_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {name: dict(counter) for name, counter in summary.items()}
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
