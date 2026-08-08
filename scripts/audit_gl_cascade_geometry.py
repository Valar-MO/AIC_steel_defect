#!/usr/bin/env python3
"""Audit labeled AIC geometry before committing to GL-Cascade-Full.

This is intentionally CPU-only.  It reads VOC annotations and evaluates the
same sliding-window geometry proposed for the local detector.  The training
split is the only split permitted to produce recommendations; validation can
be reported separately as a holdout diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--global-input", type=int, default=1024)
    parser.add_argument("--local-input", type=int, default=1536)
    parser.add_argument("--anchor-clusters", type=int, default=5)
    parser.add_argument("--report-splits", nargs="+", default=["train", "val"])
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid classes file: {path}")
    return [str(name) for name in names]


def read_voc(path: Path, class_to_id: dict[str, int]) -> tuple[int, int, list[tuple[int, tuple[float, ...]]]]:
    root = ET.parse(path).getroot()
    width, height = int(float(root.findtext("size/width", "0"))), int(float(root.findtext("size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {path}")
    boxes, seen = [], set()
    for obj in root.findall("object"):
        name, node = (obj.findtext("name") or "").strip(), obj.find("bndbox")
        if name not in class_to_id or node is None:
            continue
        try:
            x1, y1, x2, y2 = (float(node.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        box = (max(0.0, min(x1, width)), max(0.0, min(y1, height)), max(0.0, min(x2, width)), max(0.0, min(y2, height)))
        signature = (name, *box)
        if box[2] <= box[0] or box[3] <= box[1] or signature in seen:
            continue
        seen.add(signature)
        boxes.append((class_to_id[name], box))
    return width, height, boxes


def starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    result = list(range(0, length - size + 1, stride))
    last = length - size
    if result[-1] != last:
        result.append(last)
    return result


def windows(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    stride = tile_size - overlap
    return [(x, y, min(width, x + tile_size), min(height, y + tile_size))
            for y in starts(height, tile_size, stride)
            for x in starts(width, tile_size, stride)]


def intersection_fraction(box: tuple[float, ...], view: tuple[int, int, int, int]) -> float:
    x1, y1 = max(box[0], view[0]), max(box[1], view[1])
    x2, y2 = min(box[2], view[2]), min(box[3], view[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1) / ((box[2] - box[0]) * (box[3] - box[1]))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {"count": len(values), **{f"p{int(q * 100):02d}": None if (value := percentile(values, q)) is None else round(value, 4)
                                     for q in (0.05, 0.25, 0.50, 0.75, 0.90, 0.95)},
            "min": None if not values else round(min(values), 4),
            "max": None if not values else round(max(values), 4),
            "mean": None if not values else round(statistics.fmean(values), 4)}


def weighted_kmeans_1d(values: list[float], weights: list[float], clusters: int) -> list[float]:
    """Deterministic weighted 1-D k-means over log(w / h)."""
    if not values:
        return []
    clusters = min(clusters, len(values))
    centers = [percentile(values, (index + 0.5) / clusters) for index in range(clusters)]
    assert all(center is not None for center in centers)
    for _ in range(100):
        groups: list[list[int]] = [[] for _ in centers]
        for index, value in enumerate(values):
            nearest = min(range(len(centers)), key=lambda item: (abs(value - centers[item]), item))
            groups[nearest].append(index)
        updated = []
        for center, group in zip(centers, groups):
            if not group:
                updated.append(center)
                continue
            total = sum(weights[index] for index in group)
            updated.append(sum(values[index] * weights[index] for index in group) / total)
        if max(abs(a - b) for a, b in zip(centers, updated)) < 1e-6:
            break
        centers = updated
    return sorted(centers)


def rounded_unique(values: list[float], digits: int = 3) -> list[float]:
    result = []
    for value in values:
        rounded = round(value, digits)
        if not result or abs(result[-1] - rounded) > 10 ** (-digits):
            result.append(rounded)
    return result


def audit_split(rows: list[dict[str, str]], classes: list[str], tile_size: int, overlap: int,
                global_input: int, local_input: int) -> tuple[dict, list[dict], list[dict]]:
    class_to_id = {name: index for index, name in enumerate(classes)}
    by_class: dict[int, list[dict]] = defaultdict(list)
    gt_rows, crop_rows = [], []
    tile_count = positive_tile_count = 0
    image_count = 0
    for row in rows:
        width, height, boxes = read_voc(Path(row["xml_path"]), class_to_id)
        image_count += 1
        views = windows(width, height, tile_size, overlap)
        tile_count += len(views)
        labels_per_view = [0] * len(views)
        for class_id, box in boxes:
            visible = [intersection_fraction(box, view) for view in views]
            complete_count = sum(value >= 0.999999 for value in visible)
            best_visible = max(visible, default=0.0)
            label_views = sum(value >= 0.50 for value in visible)
            for index, value in enumerate(visible):
                labels_per_view[index] += int(value >= 0.50)
            bw, bh = box[2] - box[0], box[3] - box[1]
            area, image_area = bw * bh, width * height
            aspect = bw / bh
            base = {
                "class": classes[class_id], "image_width": width, "image_height": height,
                "box_width": round(bw, 4), "box_height": round(bh, 4), "area": round(area, 4),
                "area_ratio": area / image_area, "aspect_w_over_h": aspect,
                "long_short_ratio": max(aspect, 1 / aspect), "best_visible_fraction": best_visible,
                "complete_crop_count": complete_count, "label_crop_count_at_50pct": label_views,
                "global_box_width": bw * global_input / width,
                "global_box_height": bh * global_input / height,
                "local_box_width": bw * local_input / tile_size,
                "local_box_height": bh * local_input / tile_size,
                "local_sqrt_area": math.sqrt(area) * local_input / tile_size,
            }
            by_class[class_id].append(base)
            gt_rows.append(base)
        positive_tile_count += sum(count > 0 for count in labels_per_view)
        crop_rows.append({"image_id": Path(row["image_path"]).name, "tile_count": len(views),
                          "positive_tile_count": sum(count > 0 for count in labels_per_view),
                          "empty_tile_count": sum(count == 0 for count in labels_per_view)})

    class_summary = []
    for class_id, name in enumerate(classes):
        values = by_class[class_id]
        areas = [item["area_ratio"] for item in values]
        long_short = [item["long_short_ratio"] for item in values]
        local_short = [min(item["local_box_width"], item["local_box_height"]) for item in values]
        global_short = [min(item["global_box_width"], item["global_box_height"]) for item in values]
        complete = sum(item["complete_crop_count"] > 0 for item in values)
        row = {"class": name, "gt_count": len(values), "complete_crop_gt": complete,
               "complete_crop_rate": complete / len(values) if values else None,
               "multi_complete_crop_gt": sum(item["complete_crop_count"] >= 2 for item in values),
               "best_visible_lt_50": sum(item["best_visible_fraction"] < .50 for item in values),
               "best_visible_50_70": sum(.50 <= item["best_visible_fraction"] < .70 for item in values),
               "best_visible_70_99": sum(.70 <= item["best_visible_fraction"] < .999999 for item in values),
               "tiny_area_lt_0_1pct": sum(value < .001 for value in areas),
               "small_area_0_1_to_1pct": sum(.001 <= value < .01 for value in areas),
               "line_ratio_ge_5": sum(value >= 5 for value in long_short),
               "line_ratio_ge_10": sum(value >= 10 for value in long_short),
               **{f"area_ratio_{key}": value for key, value in summary(areas).items() if key != "count"},
               **{f"long_short_{key}": value for key, value in summary(long_short).items() if key != "count"},
               **{f"global_short_{key}": value for key, value in summary(global_short).items() if key != "count"},
               **{f"local_short_{key}": value for key, value in summary(local_short).items() if key != "count"}}
        class_summary.append(row)
    total_gt = len(gt_rows)
    report = {"images": image_count, "ground_truth": total_gt, "tile_count": tile_count,
              "positive_tile_count_at_50pct": positive_tile_count,
              "empty_tile_count_at_50pct": tile_count - positive_tile_count,
              "complete_crop_gt": sum(item["complete_crop_count"] > 0 for item in gt_rows),
              "complete_crop_rate": (sum(item["complete_crop_count"] > 0 for item in gt_rows) / total_gt) if total_gt else 0.0,
              "multi_complete_crop_gt": sum(item["complete_crop_count"] >= 2 for item in gt_rows),
              "best_visible_lt_50": sum(item["best_visible_fraction"] < .50 for item in gt_rows),
              "best_visible_50_70": sum(.50 <= item["best_visible_fraction"] < .70 for item in gt_rows),
              "best_visible_70_99": sum(.70 <= item["best_visible_fraction"] < .999999 for item in gt_rows),
              "class_summary": class_summary}
    return report, gt_rows, crop_rows


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("tile-size must be positive and overlap must be in [0, tile-size)")
    if args.global_input <= 0 or args.local_input <= 0 or args.anchor_clusters <= 0:
        raise ValueError("input sizes and anchor clusters must be positive")
    classes = load_classes(args.classes)
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    required = {"image_path", "xml_path", "split"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    unknown = set(args.report_splits) - {row["split"] for row in rows}
    if unknown:
        raise ValueError(f"Unknown requested split(s): {sorted(unknown)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audits, train_gt = {}, None
    for split in args.report_splits:
        split_rows = [row for row in rows if row["split"] == split]
        report, gt_rows, crop_rows = audit_split(split_rows, classes, args.tile_size, args.overlap, args.global_input, args.local_input)
        audits[split] = report
        if split == "train":
            train_gt = gt_rows
        with (output / f"{split}_class_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["class_summary"][0]))
            writer.writeheader(); writer.writerows(report["class_summary"])
        with (output / f"{split}_crop_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "tile_count", "positive_tile_count", "empty_tile_count"])
            writer.writeheader(); writer.writerows(crop_rows)
    if train_gt is None:
        raise ValueError("report-splits must include train: recommendations must not use validation labels")

    class_counts = Counter(item["class"] for item in train_gt)
    log_aspects = [math.log(max(.05, min(20.0, item["aspect_w_over_h"]))) for item in train_gt]
    class_weights = [1.0 / class_counts[item["class"]] for item in train_gt]
    cluster_logs = weighted_kmeans_1d(log_aspects, class_weights, args.anchor_clusters)
    ratios = rounded_unique([math.exp(value) for value in cluster_logs])
    local_scales = [item["local_sqrt_area"] for item in train_gt]
    local_short = [min(item["local_box_width"], item["local_box_height"]) for item in train_gt]
    global_short = [min(item["global_box_width"], item["global_box_height"]) for item in train_gt]
    recommendation = {
        "derived_from_split": "train",
        "geometry": {"global_input": args.global_input, "local_crop_size": args.tile_size,
                     "local_input": args.local_input, "tile_overlap": args.overlap,
                     "tile_stride": args.tile_size - args.overlap},
        "coverage_policy": {"complete_visible_fraction": 1.0, "normal_min_visible_fraction": 0.70,
                            "partial_min_visible_fraction": 0.50, "severe_partial_ignore_below": 0.50},
        "anchor_policy": {"ratio_definition": "width_over_height", "ratio_clusters": len(ratios),
                          "class_balanced_log_kmeans_ratios": ratios,
                          "local_sqrt_area_pixels": summary(local_scales),
                          "local_short_side_pixels": summary(local_short),
                          "global_short_side_pixels": summary(global_short),
                          "fpn_strides": [4, 8, 16, 32, 64]},
        "sampling_observation": {"class_counts": dict(sorted(class_counts.items())),
                                  "tile_stats": {key: audits["train"][key] for key in ("tile_count", "positive_tile_count_at_50pct", "empty_tile_count_at_50pct")}},
        "note": "Only the train split generated this recommendation. Validation is diagnostic-only."}
    (output / "gl_cascade_recommendation.json").write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"inputs": {"manifest": str(args.manifest.resolve()), "classes": str(args.classes.resolve()),
                          "tile_size": args.tile_size, "overlap": args.overlap,
                          "global_input": args.global_input, "local_input": args.local_input},
               "audits": audits, "recommendation": recommendation}
    (output / "audit_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train": {key: audits["train"][key] for key in ("images", "ground_truth", "tile_count", "positive_tile_count_at_50pct", "empty_tile_count_at_50pct", "complete_crop_rate")},
                      "recommendation": recommendation, "output_dir": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
