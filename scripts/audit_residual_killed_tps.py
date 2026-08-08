#!/usr/bin/env python3
"""Audit base final TPs that become false negatives after a score residual.

Both sides are replayed with the official whole+tile protocol.  Candidate IDs
use source+roi_cache_index so legacy tile query-ID collisions cannot affect the
comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_predictions import box_iou, load_classes, load_ground_truth
from merge_predictions import nms


WEAK_CLASSES = {"huashang", "qilie", "zonglie"}


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-whole", type=Path, required=True)
    p.add_argument("--base-tiles", type=Path, required=True)
    p.add_argument("--refined-whole", type=Path, required=True)
    p.add_argument("--refined-tiles", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--threshold", type=float, default=.30)
    p.add_argument("--nms-iou", type=float, default=.80)
    p.add_argument("--border-factor", type=float, default=.50)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_candidates(path, source, border_factor):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for record in payload["images"]:
        image_id = Path(record["image_id"]).name
        rows = []
        for prediction in record.get("predictions", []):
            index = int(prediction["roi_cache_index"])
            raw_score = float(prediction["score"])
            border = source == "tile" and bool(prediction.get("touches_internal_border"))
            rows.append({
                "candidate_id": f"{source}:{index}", "source": source, "roi_cache_index": index,
                "class_id": int(prediction["class_id"]), "box": [float(v) for v in prediction["bbox_xyxy"]],
                "raw_score": raw_score, "effective_score": raw_score * (border_factor if border else 1.),
                "touches_internal_border": border,
                "residual": float(prediction.get("roi_selection_residual", 0.)),
            })
        records[image_id] = {"width": int(record["width"]), "height": int(record["height"]), "rows": rows}
    return records


def final_candidates(candidates, threshold, nms_iou):
    grouped = defaultdict(list)
    for candidate in candidates:
        if candidate["effective_score"] >= threshold:
            grouped[candidate["class_id"]].append(candidate)
    output = []
    for rows in grouped.values():
        keep = nms([row["box"] for row in rows], [row["effective_score"] for row in rows], nms_iou)
        output.extend(rows[index] for index in keep)
    return sorted(output, key=lambda row: row["effective_score"], reverse=True)


def match_final(final, gt_by_class):
    assignment, present = {}, set()
    grouped = defaultdict(list)
    for row in final: grouped[row["class_id"]].append(row)
    for class_id, rows in grouped.items():
        used = set()
        for row in sorted(rows, key=lambda item: item["effective_score"], reverse=True):
            best_iou, best_index = -1., -1
            for index, gt_box in enumerate(gt_by_class.get(class_id, [])):
                if index in used: continue
                value = box_iou(row["box"], gt_box)
                if value > best_iou: best_iou, best_index = value, index
            if best_iou >= .5:
                used.add(best_index)
                assignment[row["candidate_id"]] = (class_id, best_index)
                present.add((class_id, best_index))
    return assignment, present


def logit(value):
    value = min(max(value, 1e-6), 1 - 1e-6)
    return math.log(value / (1 - value))


def band(score):
    if score < .35: return ".30-.35"
    if score < .45: return ".35-.45"
    if score < .60: return ".45-.60"
    return ">=.60"


def group_stats(rows, key):
    grouped = defaultdict(list)
    for row in rows: grouped[str(row[key])].append(row)
    return {
        name: {
            "count": len(values),
            "mean_residual": sum(row["residual"] for row in values) / len(values),
            "mean_aspect_ratio": sum(row["aspect_ratio"] for row in values) / len(values),
            "mean_area_fraction": sum(row["area_fraction"] for row in values) / len(values),
            "mean_base_effective_score": sum(row["base_effective_score"] for row in values) / len(values),
        }
        for name, values in sorted(grouped.items())
    }


def main():
    opt = args()
    if opt.output_dir.exists() and any(opt.output_dir.iterdir()) and not opt.overwrite:
        raise FileExistsError(opt.output_dir)
    classes = load_classes(opt.classes); class_to_id = {name: index for index, name in enumerate(classes)}
    gt, _, image_ids = load_ground_truth(opt.manifest, opt.split, class_to_id)
    base_whole = load_candidates(opt.base_whole, "whole", opt.border_factor)
    base_tiles = load_candidates(opt.base_tiles, "tile", opt.border_factor)
    refined_whole = load_candidates(opt.refined_whole, "whole", opt.border_factor)
    refined_tiles = load_candidates(opt.refined_tiles, "tile", opt.border_factor)
    rows, counts = [], Counter()
    for image_id in sorted(image_ids):
        base_info = base_whole[image_id]
        base_rows = base_info["rows"] + base_tiles[image_id]["rows"]
        refined_rows = refined_whole[image_id]["rows"] + refined_tiles[image_id]["rows"]
        refined_by_id = {row["candidate_id"]: row for row in refined_rows}
        gt_by_class = {class_id: gt[class_id].get(image_id, []) for class_id in range(len(classes))}
        base_final = final_candidates(base_rows, opt.threshold, opt.nms_iou)
        base_assignment, _ = match_final(base_final, gt_by_class)
        refined_final = final_candidates(refined_rows, opt.threshold, opt.nms_iou)
        refined_assignment, refined_present = match_final(refined_final, gt_by_class)
        refined_final_ids = {row["candidate_id"] for row in refined_final}
        for candidate_id, target in base_assignment.items():
            if target in refined_present: continue
            base = next(row for row in base_rows if row["candidate_id"] == candidate_id)
            refined = refined_by_id[candidate_id]
            mode = "matching_replaced"
            suppressor = None
            if refined["effective_score"] < opt.threshold:
                mode = "below_threshold"
            elif candidate_id not in refined_final_ids:
                rivals = [row for row in refined_final if row["class_id"] == refined["class_id"] and
                          box_iou(row["box"], refined["box"]) > opt.nms_iou]
                if rivals:
                    suppressor = max(rivals, key=lambda row: row["effective_score"])
                    mode = "nms_replaced"
                else:
                    mode = "nms_or_protocol_removed"
            class_name = classes[base["class_id"]]
            x1, y1, x2, y2 = base["box"]
            width, height = max(x2 - x1, 0.), max(y2 - y1, 0.)
            row = {
                "image_id": image_id, "class_name": class_name, "class_id": base["class_id"],
                "weak_class": class_name in WEAK_CLASSES, "source": base["source"],
                "area_fraction": width * height / max(base_info["width"] * base_info["height"], 1),
                "aspect_ratio": max(width / max(height, 1e-6), height / max(width, 1e-6)),
                "touches_internal_border": base["touches_internal_border"], "drop_mode": mode,
                "base_score": base["raw_score"], "base_effective_score": base["effective_score"],
                "refined_score": refined["raw_score"], "refined_effective_score": refined["effective_score"],
                "residual": logit(refined["raw_score"]) - logit(base["raw_score"]),
                "base_score_band": band(base["effective_score"]),
                "suppressor_source": suppressor["source"] if suppressor else None,
                "suppressor_score": suppressor["effective_score"] if suppressor else None,
                "suppressor_class": classes[suppressor["class_id"]] if suppressor else None,
            }
            rows.append(row)
            counts["killed"] += 1; counts[f"class:{class_name}"] += 1
            counts[f"source:{row['source']}"] += 1; counts[f"mode:{mode}"] += 1
            counts[f"band:{row['base_score_band']}"] += 1
            if row["weak_class"]: counts["weak_class"] += 1
            if row["touches_internal_border"]: counts["touches_internal_border"] += 1
    opt.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with (opt.output_dir / "killed_base_tps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    (opt.output_dir / "killed_base_tps.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"base_final_tp": sum(1 for image_id in image_ids for _ in
                                      match_final(final_candidates(base_whole[image_id]["rows"] + base_tiles[image_id]["rows"], opt.threshold, opt.nms_iou),
                                                  {class_id: gt[class_id].get(image_id, []) for class_id in range(len(classes))})[0]),
               "killed_count": len(rows), "counts": dict(counts),
               "mean_residual": sum(row["residual"] for row in rows) / max(len(rows), 1),
               "mean_aspect_ratio": sum(row["aspect_ratio"] for row in rows) / max(len(rows), 1),
               "by_class": group_stats(rows, "class_name"),
               "by_source": group_stats(rows, "source"),
               "by_score_band": group_stats(rows, "base_score_band"),
               "by_drop_mode": group_stats(rows, "drop_mode"),
               "line_like": {
                   "aspect_ratio_ge_5": sum(row["aspect_ratio"] >= 5 for row in rows),
                   "aspect_ratio_ge_10": sum(row["aspect_ratio"] >= 10 for row in rows),
                   "tile_and_aspect_ratio_ge_5": sum(row["source"] == "tile" and row["aspect_ratio"] >= 5 for row in rows),
                   "huashang_tile_and_aspect_ratio_ge_5": sum(row["class_name"] == "huashang" and row["source"] == "tile" and row["aspect_ratio"] >= 5 for row in rows),
               }}
    (opt.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
