#!/usr/bin/env python3
"""Read-only e9 ROI decision audit and optional e4/e9 final-FN overlap."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_gl_cascade_fn_path import (  # noqa: E402
    greedy_matched_gt, init_evidence, load_classes, load_predictions, primary_cause, scalar_iou, update_tile_evidence, visible_targets,
)
from src.gl_cascade import GLCascadeDetector, OfficialGridGlobalLocalDataset, gl_cascade_collate  # noqa: E402
from src.gl_cascade.boxes import box_iou  # noqa: E402
from src.gl_cascade.roi import _roi_levels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--comparison-predictions", type=Path, help="Optional e4 merged predictions for FN identity overlap.")
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--global-input-size", type=int, default=1024)
    parser.add_argument("--local-input-size", type=int, default=1536)
    parser.add_argument("--score-threshold", type=float, default=.30)
    parser.add_argument("--match-iou", type=float, default=.50)
    parser.add_argument("--high-iou", type=float, default=.60)
    return parser.parse_args()


def summary(rows: list[dict]) -> dict:
    fields = ("final_iou", "p_fg_aux", "true_minus_background", "true_minus_other_defect", "quality_probability", "short_side_feature_pixels")
    result = {"count": len(rows), "fpn_levels": dict(Counter(str(row["fpn_level"]) for row in rows))}
    for field in fields:
        values = sorted(row[field] for row in rows)
        if not values:
            result[field] = {"mean": 0., "median": 0., "p25": 0., "p75": 0.}
            continue
        at = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1)))]
        result[field] = {"mean": sum(values) / len(values), "median": at(.5), "p25": at(.25), "p75": at(.75)}
    return result


def main() -> None:
    opt = parse_args()
    classes = load_classes(opt.classes)
    checkpoint = torch.load(opt.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    dataset = OfficialGridGlobalLocalDataset(opt.manifest, opt.classes, split="val", global_input_size=opt.global_input_size,
                                             local_input_size=opt.local_input_size)
    canonical, evidence = init_evidence(dataset)
    for item in evidence.values():
        item["best_decision"] = None
    sizes = {record.image_path.name: (record.width, record.height) for record in dataset.records}
    predictions, _, _, _ = load_predictions(opt.predictions, classes, sizes, set(canonical), allow_missing=False)
    matched = greedy_matched_gt(predictions, canonical, opt.score_threshold, opt.match_iou)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GLCascadeDetector(backbone_name=args.get("backbone", "r50_dcn"), pretrained_backbone=False,
                              internimage_root=args.get("internimage_root"), backbone_checkpointing=bool(args.get("backbone_checkpointing", False)),
                              classifier_mode=args.get("classifier_mode", "softmax_background")).to(device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=device.type == "cuda",
                        persistent_workers=opt.workers > 0, collate_fn=gl_cascade_collate)
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            local, global_image = batch["local"].to(device, non_blocking=True), batch["global"].to(device, non_blocking=True)
            target, record = batch["targets"][0], dataset.records[index - 1]
            view, image_size = target["view_xyxy"].unsqueeze(0).to(device), target["image_size"].unsqueeze(0).to(device)
            features = model.fpn(model.backbone.forward_features(local), model.backbone.forward_features(global_image), view, image_size)
            rpn = model.rpn(features, tuple(local.shape[-2:]), targets=None)
            boxes = rpn["proposals"]
            for stage in model.roi_heads.stages[:-1]:
                output = stage(features, boxes)
                boxes = model.roi_heads._split_and_refine(boxes, output["box_delta"], tuple(local.shape[-2:]))
            stage3_input = boxes[0]
            final = model.roi_heads.stages[-1](features, boxes)
            final_boxes = model.roi_heads._split_and_refine(boxes, final["box_delta"], tuple(local.shape[-2:]))[0]
            detections = model._detections({"boxes": [final_boxes], "final": final})[0]
            update_tile_evidence(evidence, target["image_id"], record, rpn["proposals"][0], final_boxes, final["class_logits"],
                                 final["foreground"], detections, opt.local_input_size, opt.match_iou, len(classes), opt.score_threshold)
            levels = _roi_levels(stage3_input)
            for gt_index, label, gt_box in visible_targets(record, opt.local_input_size):
                if not len(final_boxes):
                    continue
                ious = box_iou(gt_box.unsqueeze(0).to(final_boxes), final_boxes).squeeze(0)
                candidate = int(ious.argmax())
                iou = float(ious[candidate])
                item = evidence[(target["image_id"], gt_index)]
                if item["best_decision"] is not None and iou <= item["best_decision"]["final_iou"]:
                    continue
                logits = final["class_logits"][candidate]
                competitors = logits[:len(classes)].clone(); competitors[label] = float("-inf")
                width, height = (stage3_input[candidate, 2] - stage3_input[candidate, 0]).item(), (stage3_input[candidate, 3] - stage3_input[candidate, 1]).item()
                level = int(levels[candidate])
                item["best_decision"] = {"final_iou": iou, "p_fg_aux": float(final["foreground"][candidate].sigmoid()),
                                         "true_minus_background": float(logits[label] - logits[len(classes)]),
                                         "true_minus_other_defect": float(logits[label] - competitors.max()),
                                         "quality_probability": float(final["quality"][candidate].sigmoid()), "fpn_level": level,
                                         "short_side_feature_pixels": min(width, height) / (2 ** level), "label": label}
            if index % 100 == 0:
                print(json.dumps({"tiles": index, "total": len(dataset)}, ensure_ascii=False), flush=True)

    def has_final_candidate(key, item):
        return any(score >= opt.score_threshold and image_id == key[0] and scalar_iou(box, tuple(item["box"])) >= opt.match_iou
                   for score, image_id, box in predictions[item["label"]])
    groups, per_class_rows = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for key, item in evidence.items():
        row = item["best_decision"]
        if row is None:
            continue
        name = classes[item["label"]]
        if key in matched:
            group = "final_tp"
        elif primary_cause(item, has_final_candidate(key, item)) == "roi_classified_background" and row["final_iou"] >= opt.high_iou:
            group = "high_iou_background_fn"
        else:
            continue
        groups[group].append({"image_id": key[0], "gt_index": key[1], **row})
        per_class_rows[name][group].append({"image_id": key[0], "gt_index": key[1], **row})
    overlap = None
    if opt.comparison_predictions:
        other, _, _, _ = load_predictions(opt.comparison_predictions, classes, sizes, set(canonical), allow_missing=False)
        other_matched = greedy_matched_gt(other, canonical, opt.score_threshold, opt.match_iou)
        all_keys = set(evidence)
        current_fn, other_fn = all_keys - matched, all_keys - other_matched
        overlap = {"current_fn": len(current_fn), "comparison_fn": len(other_fn), "intersection": len(current_fn & other_fn),
                   "current_fraction_persistent": len(current_fn & other_fn) / max(1, len(current_fn)),
                   "jaccard": len(current_fn & other_fn) / max(1, len(current_fn | other_fn))}
    output = {"checkpoint": str(opt.checkpoint.resolve()), "high_iou": opt.high_iou, "groups": {name: summary(rows) for name, rows in groups.items()},
              "per_class": {name: {group: summary(rows) for group, rows in values.items()} for name, values in per_class_rows.items()},
              "examples": {name: rows[:100] for name, rows in groups.items()}, "fn_overlap": overlap,
              "selection_note": "Each GT uses the local stage-3 candidate with its best final refined-box IoU across official tiles."}
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(opt.output.resolve()), "groups": {key: len(value) for key, value in groups.items()}, "fn_overlap": overlap}, ensure_ascii=False))


if __name__ == "__main__":
    main()
