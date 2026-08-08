#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ["jieba", "zonglie", "qilie", "jiaza", "yiwuyaru", "huashang", "mamianmakeng", "yanghuatiepi", "gunyin"]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def load_manifest(path: Path, split: str = "val"):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("split") == split]


def clean_voc_boxes(xml_path: Path, class_to_id: dict[str, int]):
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_to_id:
            continue
        node = obj.find("bndbox")
        if node is None:
            continue
        try:
            x1, y1, x2, y2 = (float(node.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        box = [max(0.0, min(x1, width)), max(0.0, min(y1, height)),
               max(0.0, min(x2, width)), max(0.0, min(y2, height))]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append({"class_id": class_to_id[name], "bbox": box})
    return width, height, boxes


def load_gts(manifest: Path):
    class_to_id = {name: i for i, name in enumerate(CLASSES)}
    gts = {}
    for r in load_manifest(manifest):
        image_id = Path(r["image_path"]).name
        _, _, boxes = clean_voc_boxes(Path(r["xml_path"]), class_to_id)
        gts[image_id] = boxes
    return gts


def best_match(pred, gt_items, same_class=True):
    cls = int(pred["original_class_id"] if "original_class_id" in pred else pred["class_id"])
    box = pred.get("bbox_xyxy", pred.get("bbox"))
    best_iou, best_gt = 0.0, None
    for idx, gt in enumerate(gt_items):
        if same_class and gt["class_id"] != cls:
            continue
        val = iou_xyxy(box, gt["bbox"])
        if val > best_iou:
            best_iou, best_gt = val, idx
    return best_iou, best_gt


def bucket_score(s):
    if s < 0.03:
        return "<0.03"
    if s < 0.05:
        return "0.03-0.05"
    if s < 0.08:
        return "0.05-0.08"
    if s < 0.12:
        return "0.08-0.12"
    if s < 0.2:
        return "0.12-0.20"
    return ">=0.20"


def bucket_area(box):
    x1, y1, x2, y2 = map(float, box)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area < 32 * 32:
        return "<32^2"
    if area < 64 * 64:
        return "32^2-64^2"
    if area < 128 * 128:
        return "64^2-128^2"
    if area < 256 * 256:
        return "128^2-256^2"
    return ">=256^2"


def area(box):
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def touches_edge(box, eps=2.0):
    x1, y1, x2, y2 = map(float, box)
    return x1 <= eps or y1 <= eps


def mean(items, key):
    vals = [float(key(x)) for x in items if key(x) is not None and math.isfinite(float(key(x)))]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    pred_path = Path(sys.argv[1])
    manifest = Path(sys.argv[2])
    score_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    iou_thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    gts = load_gts(manifest)

    suppressed = []
    for rec in payload["images"]:
        image_id = Path(str(rec["image_id"])).name
        for p in rec.get("predictions", []):
            v = p.get("dual_head_verifier") or {}
            if v.get("background_suppressed"):
                suppressed.append((image_id, p))

    # Independent original-score matching only for suppressed boxes.
    # This asks: if the detector prediction existed before suppression, did it align
    # with any same-class GT at IoU>=threshold?
    total = len(suppressed)
    originally_above = []
    killed = []
    true_like = []
    fp_like = []
    weak_iou_or_wrong_class = []

    by_class = defaultdict(lambda: Counter())
    by_score = defaultdict(lambda: Counter())
    by_area = defaultdict(lambda: Counter())
    examples_killed_tp = []
    examples_suppressed_fp = []
    ious_tp = []

    for image_id, p in suppressed:
        old_score = float(p.get("original_score", p.get("score", 0.0)))
        new_score = float(p.get("score", 0.0))
        cls_id = int(p.get("original_class_id", p.get("class_id")))
        cls_name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else str(cls_id)
        box = p.get("bbox_xyxy", p.get("bbox"))
        same_iou, same_gt = best_match(p, gts.get(image_id, []), same_class=True)
        any_iou, any_gt = best_match(p, gts.get(image_id, []), same_class=False)
        is_tp_like = same_iou >= iou_thr
        was_counted = old_score >= score_thr
        killed_by_threshold = old_score >= score_thr and new_score < score_thr
        item = {
            "image_id": image_id,
            "class": cls_name,
            "old_score": old_score,
            "new_score": new_score,
            "defect_probability": (p.get("dual_head_verifier") or {}).get("defect_probability"),
            "same_class_iou": same_iou,
            "any_class_iou": any_iou,
            "bbox": box,
            "area": area(box),
            "touches_top_or_left_edge": touches_edge(box),
        }
        if was_counted:
            originally_above.append(item)
        if is_tp_like:
            true_like.append(item)
            ious_tp.append(same_iou)
        else:
            fp_like.append(item)
            if any_iou >= iou_thr:
                weak_iou_or_wrong_class.append(item)
        if killed_by_threshold:
            killed.append(item)
            label = "killed_tp" if is_tp_like else "killed_fp"
            by_class[cls_name][label] += 1
            by_score[bucket_score(old_score)][label] += 1
            by_area[bucket_area(box)][label] += 1
            if is_tp_like and len(examples_killed_tp) < 20:
                examples_killed_tp.append(item)
            if (not is_tp_like) and len(examples_suppressed_fp) < 20:
                examples_suppressed_fp.append(item)

    killed_tp = [x for x in killed if x["same_class_iou"] >= iou_thr]
    killed_fp = [x for x in killed if x["same_class_iou"] < iou_thr]
    report = {
        "prediction_file": str(pred_path),
        "score_threshold": score_thr,
        "match_iou": iou_thr,
        "suppressed_total": total,
        "suppressed_originally_score_ge_threshold": len(originally_above),
        "suppressed_killed_by_score_threshold": len(killed),
        "suppressed_tp_like_all": len(true_like),
        "suppressed_fp_like_all": len(fp_like),
        "killed_tp_like": len(killed_tp),
        "killed_fp_like": len(killed_fp),
        "killed_fp_over_killed": len(killed_fp) / len(killed) if killed else 0,
        "killed_tp_over_killed": len(killed_tp) / len(killed) if killed else 0,
        "suppressed_wrong_class_or_localized_other_class": len(weak_iou_or_wrong_class),
        "killed_by_class": {k: dict(v) for k, v in sorted(by_class.items())},
        "killed_by_original_score_bin": {k: dict(v) for k, v in sorted(by_score.items())},
        "killed_by_bbox_area_bin": {k: dict(v) for k, v in sorted(by_area.items())},
        "killed_tp_iou_mean": sum(ious_tp) / len(ious_tp) if ious_tp else 0,
        "killed_tp_summary": {
            "old_score_mean": mean(killed_tp, lambda x: x["old_score"]),
            "defect_probability_mean": mean(killed_tp, lambda x: x["defect_probability"]),
            "area_mean": mean(killed_tp, lambda x: x["area"]),
            "touch_top_or_left_edge_count": sum(1 for x in killed_tp if x["touches_top_or_left_edge"]),
        },
        "killed_fp_summary": {
            "old_score_mean": mean(killed_fp, lambda x: x["old_score"]),
            "defect_probability_mean": mean(killed_fp, lambda x: x["defect_probability"]),
            "area_mean": mean(killed_fp, lambda x: x["area"]),
            "touch_top_or_left_edge_count": sum(1 for x in killed_fp if x["touches_top_or_left_edge"]),
        },
        "examples_killed_tp": examples_killed_tp,
        "examples_killed_fp": examples_suppressed_fp,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
