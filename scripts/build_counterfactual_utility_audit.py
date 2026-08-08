#!/usr/bin/env python3
"""Build final-protocol counterfactual utility labels for ROI candidates.

This is an offline audit only.  It replays the exact whole+tile score penalty,
class-wise NMS and .30 threshold; it never trains a model and never consumes
test labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_predictions import box_iou, load_classes, load_ground_truth
from merge_predictions import nms


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--whole", type=Path, required=True)
    p.add_argument("--tiles", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--threshold", type=float, default=.30)
    p.add_argument("--nms-iou", type=float, default=.80)
    p.add_argument("--border-factor", type=float, default=.50)
    p.add_argument("--residual-max", type=float, default=.35,
                   help="Maximum positive residual in logit space for potential-FP replay.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sigmoid(value):
    import math
    return 1.0 / (1.0 + math.exp(-value))


def logit(value):
    import math
    value = min(max(value, 1e-6), 1 - 1e-6)
    return math.log(value / (1 - value))


def load_candidates(path, source, border_factor):
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = defaultdict(list)
    for record in payload["images"]:
        image_id = Path(record["image_id"]).name
        for pred in record.get("predictions", []):
            # Older cache exports used mode:image:query, which collides across
            # tile views.  roi_cache_index is immutable and unique per source.
            if pred.get("roi_cache_index") is None:
                raise ValueError(f"Missing roi_cache_index in {path}")
            candidate_id = f"{source}:{int(pred['roi_cache_index'])}"
            score = float(pred["score"])
            border = source == "tile" and bool(pred.get("touches_internal_border"))
            output[image_id].append({
                "candidate_id": candidate_id, "roi_cache_index": int(pred["roi_cache_index"]),
                "source": source, "class_id": int(pred["class_id"]), "score": score,
                "effective_score": score * (border_factor if border else 1.0),
                "border": border, "box": [float(x) for x in pred["bbox_xyxy"]],
            })
    return output


def final_candidates(candidates, threshold, nms_iou, overrides=None, removed=None):
    """Replay the official candidate scoring/NMS for one original image."""
    overrides, removed = overrides or {}, removed or set()
    grouped = defaultdict(list)
    for candidate in candidates:
        if candidate["candidate_id"] in removed: continue
        item = dict(candidate)
        item["effective_score"] = overrides.get(item["candidate_id"], item["effective_score"])
        if item["effective_score"] >= threshold: grouped[item["class_id"]].append(item)
    output = []
    for _, items in grouped.items():
        keep = nms([item["box"] for item in items], [item["effective_score"] for item in items], nms_iou)
        output.extend(items[index] for index in keep)
    return sorted(output, key=lambda item: item["effective_score"], reverse=True)


def match_final(final, gt_by_class):
    """Return candidate->GT assignment using the evaluator's per-image ordering."""
    assignment, present = {}, set()
    by_class = defaultdict(list)
    for item in final: by_class[item["class_id"]].append(item)
    for class_id, items in by_class.items():
        used = set()
        for item in sorted(items, key=lambda row: row["effective_score"], reverse=True):
            best_iou, best_index = -1.0, -1
            for index, box in enumerate(gt_by_class.get(class_id, [])):
                if index in used: continue
                value = box_iou(item["box"], box)
                if value > best_iou: best_iou, best_index = value, index
            if best_iou >= .5:
                used.add(best_index); assignment[item["candidate_id"]] = (class_id, best_index)
                present.add((class_id, best_index))
    return assignment, present


def main():
    opt = parse_args()
    if opt.output.exists() and not opt.overwrite: raise FileExistsError(opt.output)
    classes = load_classes(opt.classes); class_to_id = {name: i for i, name in enumerate(classes)}
    gt, _, image_ids = load_ground_truth(opt.manifest, opt.split, class_to_id)
    whole = load_candidates(opt.whole, "whole", opt.border_factor)
    tiles = load_candidates(opt.tiles, "tile", opt.border_factor)
    rows, role_counts, role_by_class, role_by_source = [], Counter(), Counter(), Counter()
    for image_id in sorted(image_ids):
        candidates = whole[image_id] + tiles[image_id]
        gt_by_class = {class_id: gt[class_id].get(image_id, []) for class_id in range(len(classes))}
        baseline = final_candidates(candidates, opt.threshold, opt.nms_iou)
        assignment, present = match_final(baseline, gt_by_class)
        max_iou, nearest_label, nearest_index = {}, {}, {}
        for candidate in candidates:
            best, label, gt_index = 0.0, -1, -1
            for class_id, boxes in gt_by_class.items():
                for index, box in enumerate(boxes):
                    value = box_iou(candidate["box"], box)
                    if value > best: best, label, gt_index = value, class_id, index
            max_iou[candidate["candidate_id"]] = best; nearest_label[candidate["candidate_id"]] = label
            nearest_index[candidate["candidate_id"]] = gt_index
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            role = "harmless_background"
            assigned = assignment.get(candidate_id)
            if assigned is not None:
                without = final_candidates(candidates, opt.threshold, opt.nms_iou, removed={candidate_id})
                _, present_without = match_final(without, gt_by_class)
                role = "essential_tp" if assigned not in present_without else "replaceable_tp"
            elif max_iou[candidate_id] >= .5 and nearest_label[candidate_id] == candidate["class_id"]:
                role = "duplicate_or_replaceable"
            elif max_iou[candidate_id] >= .5:
                role = "wrong_class_high_iou"
            elif max_iou[candidate_id] >= .1:
                role = "near_or_localization"
            elif max_iou[candidate_id] < .1 and any(item["candidate_id"] == candidate_id for item in baseline):
                role = "final_harmful_background"
            elif max_iou[candidate_id] < .1:
                max_effective = sigmoid(logit(candidate["score"]) + opt.residual_max)
                if candidate["border"]: max_effective *= opt.border_factor
                promoted = final_candidates(candidates, opt.threshold, opt.nms_iou,
                                            overrides={candidate_id: max_effective})
                if any(item["candidate_id"] == candidate_id for item in promoted):
                    role = "potential_harmful_background"
            row = {"image_id": image_id, **candidate, "max_iou": max_iou[candidate_id],
                   "nearest_gt_label": nearest_label[candidate_id],
                   "nearest_gt_index": nearest_index[candidate_id], "utility_role": role,
                   "base_final_tp": assigned is not None}
            rows.append(row); role_counts[role] += 1
            role_by_class[f"{classes[candidate['class_id']]}:{role}"] += 1
            role_by_source[f"{candidate['source']}:{role}"] += 1
    payload = {"metadata": {"split": opt.split, "threshold": opt.threshold, "nms_iou": opt.nms_iou,
                              "border_factor": opt.border_factor, "residual_max": opt.residual_max,
                              "classes": classes, "candidate_count": len(rows)}, "candidates": rows}
    opt.output.parent.mkdir(parents=True, exist_ok=True); opt.output.write_text(json.dumps(payload), encoding="utf-8")
    report = {**payload["metadata"], "role_counts": dict(role_counts),
              "role_by_class": dict(role_by_class), "role_by_source": dict(role_by_source)}
    report_path = opt.report or opt.output.with_name(opt.output.stem + "_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
