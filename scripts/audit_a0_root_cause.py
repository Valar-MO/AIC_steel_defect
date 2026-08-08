#!/usr/bin/env python3
"""Replay A0 whole+tile fusion and diagnose every validation GT outcome.

The audit deliberately starts from the saved raw ``.001`` whole/tile outputs.
It applies the official tile-border penalty, class-wise NMS, and operating
threshold before assigning a mutually exclusive failure mode to every GT.
It also exports high-score pure-background FPs and paired FP/difficult-TP
image crops for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evaluate_predictions import box_iou, load_classes, load_ground_truth
from merge_predictions import nms


WEAK_CLASSES = {"huashang", "qilie", "zonglie"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--candidate-min-score", type=float, default=0.001)
    parser.add_argument("--operating-threshold", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.80)
    parser.add_argument("--border-factor", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--visual-pairs", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def size_bucket(box, width, height):
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    fraction = area / max(float(width * height), 1.0)
    if fraction < 0.001:
        return "tiny_lt_0.1pct", fraction
    if fraction < 0.01:
        return "small_0.1_to_1pct", fraction
    if fraction < 0.10:
        return "medium_1_to_10pct", fraction
    return "large_ge_10pct", fraction


def aspect_ratio(box):
    width, height = max(box[2] - box[0], 1e-6), max(box[3] - box[1], 1e-6)
    return max(width / height, height / width)


def load_raw(path, source, border_factor, minimum_score):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for record in payload["images"]:
        image_id = Path(record["image_id"]).name
        rows = []
        for raw_index, prediction in enumerate(record.get("predictions", [])):
            raw_score = float(prediction["score"])
            border = source == "tile" and bool(prediction.get("touches_internal_border"))
            score = raw_score * (border_factor if border else 1.0)
            if score < minimum_score:
                continue
            rows.append({
                "candidate_id": f"{source}:{raw_index}", "source": source,
                "raw_index": raw_index, "class_id": int(prediction["class_id"]),
                "class_name": str(prediction["category_name"]),
                "box": tuple(float(value) for value in prediction["bbox_xyxy"]),
                "raw_score": raw_score, "score": score,
                "touches_internal_border": border,
            })
        records[image_id] = {
            "width": int(record["width"]), "height": int(record["height"]),
            "source_path": str(record.get("source_path", "")), "rows": rows,
        }
    return records


def apply_official_protocol(rows, threshold, nms_iou, max_det):
    """Return NMS survivors and a candidate_id -> suppressor map."""
    grouped = defaultdict(list)
    for row in rows:
        if row["score"] >= threshold:
            grouped[row["class_id"]].append(row)
    survivors, suppressed_by = [], {}
    for same_class in grouped.values():
        order = sorted(range(len(same_class)), key=lambda index: same_class[index]["score"], reverse=True)
        kept = []
        for index in order:
            candidate = same_class[index]
            suppressor = next((winner for winner in kept
                               if box_iou(candidate["box"], winner["box"]) > nms_iou), None)
            if suppressor is None:
                kept.append(candidate)
            else:
                suppressed_by[candidate["candidate_id"]] = suppressor
        survivors.extend(kept)
    survivors.sort(key=lambda row: row["score"], reverse=True)
    for row in survivors[max_det:]:
        suppressed_by[row["candidate_id"]] = {"candidate_id": "max_det", "score": None}
    return survivors[:max_det], suppressed_by


def match_final(survivors, gt, classes):
    """Match official survivors using the evaluator's classwise score order."""
    assigned, matched_gt = {}, set()
    by_class = defaultdict(list)
    for row in survivors:
        by_class[row["class_id"]].append(row)
    for class_id, rows in by_class.items():
        used = set()
        for row in sorted(rows, key=lambda item: item["score"], reverse=True):
            candidates = gt[class_id].get(row["image_id"], [])
            best_iou, best_index = -1.0, -1
            for index, gt_box in enumerate(candidates):
                if index in used:
                    continue
                iou = box_iou(row["box"], gt_box)
                if iou > best_iou:
                    best_iou, best_index = iou, index
            if best_iou >= 0.5:
                used.add(best_index)
                key = (class_id, row["image_id"], best_index)
                assigned[row["candidate_id"]] = key
                matched_gt.add(key)
    return assigned, matched_gt


def choose_best(gt_box, rows, class_id=None, min_iou=0.0):
    choices = []
    for row in rows:
        if class_id is not None and row["class_id"] != class_id:
            continue
        iou = box_iou(gt_box, row["box"])
        if iou >= min_iou:
            choices.append((iou, row["score"], row))
    return max(choices, default=(0.0, 0.0, None), key=lambda item: (item[0], item[1]))


def gt_status(gt_box, class_id, gt_key, rows, survivor_ids, suppressed_by, matched_gt):
    any_iou, _, best_any = choose_best(gt_box, rows)
    same_iou, _, best_same = choose_best(gt_box, rows, class_id)
    if gt_key in matched_gt:
        return "final_tp", best_same or best_any, max(any_iou, same_iou)
    if any_iou < 0.30:
        return "no_localized_candidate", best_any, any_iou
    if any_iou < 0.50:
        return "poor_localization", best_any, any_iou
    if same_iou < 0.50:
        return "wrong_class", best_any, any_iou
    correct = [row for row in rows if row["class_id"] == class_id and box_iou(gt_box, row["box"]) >= 0.50]
    if not any(row["score"] >= 0.30 for row in correct):
        return "low_selection", best_same, same_iou
    if not any(row["candidate_id"] in survivor_ids for row in correct):
        return "nms_suppressed", best_same, same_iou
    return "duplicate_or_competing", best_same, same_iou


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def crop_with_box(image_path, box, label):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1, 96.0) * 2.0
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    right, bottom = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
    crop = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)
    draw.rectangle((x1 - left, y1 - top, x2 - left, y2 - top), outline=(255, 70, 70), width=3)
    draw.rectangle((0, 0, min(crop.width, 400), 20), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255), font=ImageFont.load_default())
    crop.thumbnail((440, 300))
    panel = Image.new("RGB", (440, 324), "white")
    panel.paste(crop, ((440 - crop.width) // 2, 24))
    return panel


def make_pairs(rows, output_dir, limit):
    by_key = defaultdict(lambda: {"fp": [], "tp": []})
    for row in rows:
        key = (row["class_name"], row["source"], row["size_bucket"])
        if row["row_type"] == "background_fp":
            by_key[key]["fp"].append(row)
        elif row["row_type"] == "difficult_tp":
            by_key[key]["tp"].append(row)
    pair_dir = output_dir / "visual_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for key, groups in sorted(by_key.items()):
        if not groups["fp"] or not groups["tp"]:
            continue
        fps = sorted(groups["fp"], key=lambda row: row["score"], reverse=True)[:limit]
        tps = sorted(groups["tp"], key=lambda row: row["score"])[:limit]
        count = min(len(fps), len(tps))
        canvas = Image.new("RGB", (880, 324 * count), "white")
        for index in range(count):
            fp, tp = fps[index], tps[index]
            try:
                left = crop_with_box(fp["source_path"], fp["box"], f"FP {fp['class_name']} {fp['source']} s={fp['score']:.3f}")
                right = crop_with_box(tp["source_path"], tp["box"], f"TP {tp['class_name']} {tp['source']} s={tp['score']:.3f}")
            except (FileNotFoundError, OSError):
                continue
            canvas.paste(left, (0, index * 324)); canvas.paste(right, (440, index * 324))
        name = "__".join(key) + ".jpg"
        canvas.save(pair_dir / name, quality=92)
        manifest.append({"file": name, "class_name": key[0], "source": key[1], "size_bucket": key[2], "pairs": count})
    write_csv(output_dir / "visual_pair_manifest.csv", manifest)


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classes = load_classes(args.classes)
    class_to_id = {name: index for index, name in enumerate(classes)}
    gt, sizes, image_ids = load_ground_truth(args.manifest, args.split, class_to_id)
    whole = load_raw(args.whole, "whole", args.border_factor, args.candidate_min_score)
    tiles = load_raw(args.tiles, "tile", args.border_factor, args.candidate_min_score)
    if set(whole) != image_ids or set(tiles) != image_ids:
        raise ValueError("Raw prediction image IDs do not match requested split")

    gt_rows, visual_rows, fp_rows = [], [], []
    counts, by_class, by_source, by_size = Counter(), Counter(), Counter(), Counter()
    for image_id in sorted(image_ids):
        info = whole[image_id]
        rows = [dict(row, image_id=image_id) for row in info["rows"] + tiles[image_id]["rows"]]
        survivors, suppressed_by = apply_official_protocol(rows, args.operating_threshold, args.nms_iou, args.max_det)
        survivor_ids = {row["candidate_id"] for row in survivors}
        _, matched_gt = match_final(survivors, gt, classes)
        for class_id in range(len(classes)):
            for gt_index, gt_box in enumerate(gt[class_id].get(image_id, [])):
                gt_key = (class_id, image_id, gt_index)
                status, candidate, best_iou = gt_status(gt_box, class_id, gt_key, rows, survivor_ids, suppressed_by, matched_gt)
                bucket, area_fraction = size_bucket(gt_box, info["width"], info["height"])
                row = {
                    "image_id": image_id, "class_name": classes[class_id], "weak_class": int(classes[class_id] in WEAK_CLASSES),
                    "status": status, "size_bucket": bucket, "area_fraction": round(area_fraction, 8),
                    "gt_aspect_ratio": round(aspect_ratio(gt_box), 4), "best_iou": round(best_iou, 6),
                    "candidate_source": candidate["source"] if candidate else "", "candidate_score": round(candidate["score"], 6) if candidate else "",
                    "candidate_raw_score": round(candidate["raw_score"], 6) if candidate else "",
                    "candidate_class": candidate["class_name"] if candidate else "",
                    "candidate_border": int(candidate["touches_internal_border"]) if candidate else "",
                    "candidate_aspect_ratio": round(aspect_ratio(candidate["box"]), 4) if candidate else "",
                    "candidate_nms_suppressor": suppressed_by.get(candidate["candidate_id"], {}).get("candidate_id", "") if candidate else "",
                }
                gt_rows.append(row); counts[status] += 1
                by_class[(classes[class_id], status)] += 1
                by_source[(row["candidate_source"] or "none", status)] += 1
                by_size[(bucket, status)] += 1
                if status == "final_tp" and (classes[class_id] in WEAK_CLASSES or bucket in {"tiny_lt_0.1pct", "small_0.1_to_1pct"} or candidate["score"] < .45):
                    visual_rows.append({"row_type": "difficult_tp", "source_path": info["source_path"], "box": candidate["box"],
                                        "class_name": classes[class_id], "source": candidate["source"], "size_bucket": bucket, "score": candidate["score"]})
        for survivor in survivors:
            nearest_iou = max((box_iou(survivor["box"], gt_box)
                               for class_boxes in gt.values() for gt_box in class_boxes.get(image_id, [])), default=0.0)
            if nearest_iou < 0.10:
                bucket, area_fraction = size_bucket(survivor["box"], info["width"], info["height"])
                row = {"image_id": image_id, "class_name": survivor["class_name"], "source": survivor["source"],
                       "score": round(survivor["score"], 6), "raw_score": round(survivor["raw_score"], 6),
                       "touches_internal_border": int(survivor["touches_internal_border"]), "size_bucket": bucket,
                       "area_fraction": round(area_fraction, 8), "aspect_ratio": round(aspect_ratio(survivor["box"]), 4),
                       "nearest_gt_iou": round(nearest_iou, 6), "box": list(survivor["box"])}
                fp_rows.append(row)
                visual_rows.append({"row_type": "background_fp", "source_path": info["source_path"], "box": survivor["box"],
                                    "class_name": survivor["class_name"], "source": survivor["source"], "size_bucket": bucket, "score": survivor["score"]})
    write_csv(args.output_dir / "gt_outcomes.csv", gt_rows)
    write_csv(args.output_dir / "high_score_background_fp.csv", sorted(fp_rows, key=lambda row: row["score"], reverse=True))
    make_pairs(visual_rows, args.output_dir, args.visual_pairs)
    summary = {
        "protocol": {"candidate_min_score": args.candidate_min_score, "operating_threshold": args.operating_threshold,
                     "nms_iou": args.nms_iou, "tile_border_factor": args.border_factor},
        "ground_truth": len(gt_rows), "gt_status": dict(counts), "pure_background_fp": len(fp_rows),
        "by_class": {name: {status: by_class[(name, status)] for status in sorted(counts)} for name in classes},
        "by_candidate_source": {source: {status: by_source[(source, status)] for status in sorted(counts)}
                                for source in sorted({key[0] for key in by_source})},
        "by_gt_size": {bucket: {status: by_size[(bucket, status)] for status in sorted(counts)}
                       for bucket in sorted({key[0] for key in by_size})},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
