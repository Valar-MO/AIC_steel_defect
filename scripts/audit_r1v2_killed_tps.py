#!/usr/bin/env python3
"""Audit base-fused TPs that disappear after R1-v2 rescoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fused", type=Path, required=True)
    parser.add_argument("--r1-fused", type=Path, required=True)
    parser.add_argument("--r1-whole-raw", type=Path, required=True)
    parser.add_argument("--r1-tile-raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--same-box-iou", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(names, list):
        raise ValueError(f"Invalid class file: {path}")
    return [str(name) for name in names]


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_area(box) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def aspect_ratio(box) -> float:
    x1, y1, x2, y2 = map(float, box)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return max(w / h, h / w)


def clean_voc_boxes(xml_path: Path, class_to_id: dict[str, int]):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        node = obj.find("bndbox")
        if name not in class_to_id or node is None:
            continue
        raw = [float(node.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
        if all(math.isfinite(value) for value in raw) and raw[2] > raw[0] and raw[3] > raw[1]:
            boxes.append({"class_id": class_to_id[name], "bbox": raw})
    return boxes


def load_gt(manifest: Path, split: str, class_to_id: dict[str, int]):
    gt = {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            gt[Path(row["image_path"]).name] = clean_voc_boxes(Path(row["xml_path"]), class_to_id)
    return gt


def load_predictions(path: Path, threshold: float):
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for record in payload["images"]:
        image_id = Path(str(record["image_id"])).name
        for index, pred in enumerate(record.get("predictions", [])):
            score = float(pred["score"])
            if score < threshold:
                continue
            grouped[image_id].append({
                "record_index": index,
                "class_id": int(pred["class_id"]),
                "class_name": str(pred["category_name"]),
                "score": score,
                "bbox": [float(value) for value in pred["bbox_xyxy"]],
                "source": pred.get("source"),
                "touches_internal_border": bool(pred.get("touches_internal_border")),
                "original_score": float(pred.get("original_score", score)),
            })
    for rows in grouped.values():
        rows.sort(key=lambda row: row["score"], reverse=True)
    return grouped


def match_tps(predictions, gt_by_image, match_iou: float):
    matched = defaultdict(set)
    tps = []
    for image_id, rows in predictions.items():
        for pred in rows:
            best_index, best_iou = -1, 0.0
            for gt_index, gt in enumerate(gt_by_image.get(image_id, [])):
                if gt_index in matched[image_id] or gt["class_id"] != pred["class_id"]:
                    continue
                value = iou_xyxy(pred["bbox"], gt["bbox"])
                if value > best_iou:
                    best_index, best_iou = gt_index, value
            if best_index >= 0 and best_iou >= match_iou:
                matched[image_id].add(best_index)
                tps.append({**pred, "image_id": image_id, "gt_index": best_index,
                            "match_iou": best_iou, "gt_bbox": gt_by_image[image_id][best_index]["bbox"]})
    return tps, matched


def index_raw(path: Path, threshold: float = 0.0):
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for record in payload["images"]:
        image_id = Path(str(record["image_id"])).name
        for pred in record.get("predictions", []):
            score = float(pred["score"])
            if score < threshold:
                continue
            grouped[(image_id, int(pred["class_id"]))].append(pred)
    return grouped


def find_raw_match(row, raw_index, same_box_iou: float):
    source = row.get("source")
    key = (row["image_id"], row["class_id"])
    best = None
    best_iou = 0.0
    for pred in raw_index.get(key, []):
        if source is not None and pred.get("source") != source:
            continue
        value = iou_xyxy(row["bbox"], pred["bbox_xyxy"])
        if value > best_iou:
            best, best_iou = pred, value
    return best if best is not None and best_iou >= same_box_iou else None


def bucket_area(area: float) -> str:
    if area < 32 * 32:
        return "<32^2"
    if area < 64 * 64:
        return "32^2-64^2"
    if area < 128 * 128:
        return "64^2-128^2"
    if area < 256 * 256:
        return "128^2-256^2"
    return ">=256^2"


def bucket_aspect(value: float) -> str:
    if value < 2:
        return "<2"
    if value < 5:
        return "2-5"
    if value < 10:
        return "5-10"
    return ">=10"


def mean(rows, key):
    values = [key(row) for row in rows if key(row) is not None and math.isfinite(float(key(row)))]
    return sum(values) / len(values) if values else None


def main():
    args = parse_args()
    classes = load_classes(args.classes)
    weak = {"huashang", "qilie", "zonglie"}
    gt = load_gt(args.manifest, args.split, {name: index for index, name in enumerate(classes)})
    base_predictions = load_predictions(args.base_fused, args.score_threshold)
    r1_predictions = load_predictions(args.r1_fused, args.score_threshold)
    base_tps, _ = match_tps(base_predictions, gt, args.match_iou)
    r1_tps, r1_matched = match_tps(r1_predictions, gt, args.match_iou)
    raw_index = index_raw(args.r1_whole_raw)
    for key, rows in index_raw(args.r1_tile_raw).items():
        raw_index[key].extend(rows)

    killed = []
    preserved = []
    for row in base_tps:
        still_matched = row["gt_index"] in r1_matched.get(row["image_id"], set())
        target = preserved if still_matched else killed
        raw = find_raw_match(row, raw_index, args.same_box_iou)
        residual = None
        final_raw_score = None
        ranker_base = None
        if raw is not None and raw.get("selection_ranker"):
            residual = float(raw["selection_ranker"]["logit_residual"])
            ranker_base = float(raw["selection_ranker"].get("ranker_base_selection", raw["selection_ranker"]["base_score"]))
            final_raw_score = float(raw["score"])
        area = box_area(row["bbox"])
        item = {
            **row,
            "area": area,
            "area_bucket": bucket_area(area),
            "aspect_ratio": aspect_ratio(row["bbox"]),
            "aspect_bucket": bucket_aspect(aspect_ratio(row["bbox"])),
            "weak_class": row["class_name"] in weak,
            "ranker_logit_residual": residual,
            "ranker_base_selection": ranker_base,
            "r1_raw_score": final_raw_score,
            "dropped_below_threshold": final_raw_score is not None and final_raw_score < args.score_threshold,
            "raw_match_found": raw is not None,
        }
        target.append(item)

    def summarize(rows):
        counters = {
            "by_class": Counter(row["class_name"] for row in rows),
            "by_source": Counter(row.get("source") or "unknown" for row in rows),
            "by_area_bucket": Counter(row["area_bucket"] for row in rows),
            "by_aspect_bucket": Counter(row["aspect_bucket"] for row in rows),
            "weak_class": Counter(str(row["weak_class"]) for row in rows),
            "touches_internal_border": Counter(str(row["touches_internal_border"]) for row in rows),
            "dropped_below_threshold": Counter(str(row["dropped_below_threshold"]) for row in rows),
            "raw_match_found": Counter(str(row["raw_match_found"]) for row in rows),
        }
        return {
            "count": len(rows),
            **{key: dict(value) for key, value in counters.items()},
            "means": {
                "base_fused_score": mean(rows, lambda row: row["score"]),
                "base_original_score": mean(rows, lambda row: row["original_score"]),
                "r1_raw_score": mean(rows, lambda row: row["r1_raw_score"]),
                "ranker_base_selection": mean(rows, lambda row: row["ranker_base_selection"]),
                "ranker_logit_residual": mean(rows, lambda row: row["ranker_logit_residual"]),
                "match_iou": mean(rows, lambda row: row["match_iou"]),
                "area": mean(rows, lambda row: row["area"]),
                "aspect_ratio": mean(rows, lambda row: row["aspect_ratio"]),
            },
        }

    report = {
        "metadata": {
            "base_fused": str(args.base_fused),
            "r1_fused": str(args.r1_fused),
            "score_threshold": args.score_threshold,
            "match_iou": args.match_iou,
            "base_tp_count": len(base_tps),
            "r1_tp_count": len(r1_tps),
        },
        "killed_summary": summarize(killed),
        "preserved_summary": summarize(preserved),
        "killed_examples": sorted(
            killed,
            key=lambda row: (
                row["class_name"] not in weak,
                row.get("source") != "tile",
                row["r1_raw_score"] if row["r1_raw_score"] is not None else 1,
            ),
        )[:80],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "killed_summary": report["killed_summary"],
        "preserved_summary": {
            "count": report["preserved_summary"]["count"],
            "means": report["preserved_summary"]["means"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
