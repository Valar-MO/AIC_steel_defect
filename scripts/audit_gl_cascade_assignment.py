#!/usr/bin/env python3
"""Audit Cascade classification assignment against official IoU=.50 recall.

This is read-only: it replays a validation checkpoint without changing model
weights.  Per original-image GT it aggregates evidence from every official
overlapping tile, then reports (1) the raw final-box IoU distribution of
background-classified false negatives and (2) stage-wise positive proposal
supply under both IoU=.50 and the current Cascade classification thresholds.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_gl_cascade_fn_path import (  # noqa: E402
    greedy_matched_gt, init_evidence, load_classes, load_predictions,
    primary_cause, scalar_iou, update_tile_evidence, visible_targets,
)
from src.gl_cascade import GLCascadeDetector, OfficialGridGlobalLocalDataset, gl_cascade_collate  # noqa: E402
from src.gl_cascade.boxes import box_iou  # noqa: E402


IOU_BUCKETS = ((.50, .55), (.55, .60), (.60, .70), (.70, .80), (.80, float("inf")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--global-input-size", type=int, default=1024)
    parser.add_argument("--local-input-size", type=int, default=1536)
    parser.add_argument("--score-threshold", type=float, default=.30)
    parser.add_argument("--match-iou", type=float, default=.50)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": 0., "median": 0., "p10": 0., "p25": 0., "zero": 0}

    def percentile(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]

    return {"count": len(ordered), "mean": sum(ordered) / len(ordered), "median": percentile(.50),
            "p10": percentile(.10), "p25": percentile(.25), "zero": sum(value == 0 for value in ordered)}


def _bucket(iou: float) -> str:
    for low, high in IOU_BUCKETS:
        if low <= iou < high:
            return f"{low:.2f}-{high:.2f}" if high != float("inf") else ">=0.80"
    return "<0.50"


def _final_true_candidate(predictions: dict[int, list[tuple[float, str, tuple[float, ...]]]], canonical: dict,
                          key: tuple[str, int], label: int, gt_box: tuple[float, ...], score_threshold: float,
                          match_iou: float) -> bool:
    image_id, _ = key
    return any(score >= score_threshold and candidate_image == image_id and scalar_iou(box, gt_box) >= match_iou
               for score, candidate_image, box in predictions[label])


def main() -> None:
    opt = parse_args()
    class_names = load_classes(opt.classes)
    checkpoint = torch.load(opt.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    dataset = OfficialGridGlobalLocalDataset(opt.manifest, opt.classes, split="val", global_input_size=opt.global_input_size,
                                             local_input_size=opt.local_input_size)
    canonical, evidence = init_evidence(dataset)
    for item in evidence.values():
        item["stage_best_iou"] = [0., 0., 0.]
        item["stage_iou50_count"] = [0, 0, 0]
        item["stage_assigned_positive_count"] = [0, 0, 0]
    sizes = {record.image_path.name: (record.width, record.height) for record in dataset.records}
    predictions, _, _, _ = load_predictions(opt.predictions, class_names, sizes, set(canonical), allow_missing=False)
    matched = greedy_matched_gt(predictions, canonical, opt.score_threshold, opt.match_iou)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GLCascadeDetector(backbone_name=train_args.get("backbone", "r50_dcn"), pretrained_backbone=False,
                              internimage_root=train_args.get("internimage_root"),
                              backbone_checkpointing=bool(train_args.get("backbone_checkpointing", False)),
                              classifier_mode=train_args.get("classifier_mode", "softmax_background")).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=device.type == "cuda",
                        persistent_workers=opt.workers > 0, collate_fn=gl_cascade_collate)

    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            local, global_image = batch["local"].to(device, non_blocking=True), batch["global"].to(device, non_blocking=True)
            target, record = batch["targets"][0], dataset.records[index - 1]
            view, image_size = target["view_xyxy"].unsqueeze(0).to(device), target["image_size"].unsqueeze(0).to(device)
            local_features, global_features = model.backbone.forward_features(local), model.backbone.forward_features(global_image)
            features = model.fpn(local_features, global_features, view, image_size)
            rpn = model.rpn(features, tuple(local.shape[-2:]), targets=None)

            # This is the same inference cascade, while retaining each stage's
            # incoming proposals before refinement for assignment accounting.
            boxes, stage_boxes, final = rpn["proposals"], [], None
            for stage, threshold in zip(model.roi_heads.stages, model.roi_heads.thresholds):
                stage_boxes.append(boxes[0])
                output = stage(features, boxes)
                boxes = model.roi_heads._split_and_refine(boxes, output["box_delta"], tuple(local.shape[-2:]))
                final = output
            local_detections = model._detections({"boxes": boxes, "final": final})[0]
            update_tile_evidence(evidence, target["image_id"], record, rpn["proposals"][0], boxes[0],
                                 final["class_logits"], final["foreground"], local_detections, opt.local_input_size,
                                 opt.match_iou, len(class_names), opt.score_threshold)

            for gt_index, _, gt_box in visible_targets(record, opt.local_input_size):
                item = evidence[(target["image_id"], gt_index)]
                for stage_index, (proposal_boxes, threshold) in enumerate(zip(stage_boxes, model.roi_heads.thresholds)):
                    ious = box_iou(gt_box.unsqueeze(0).to(proposal_boxes), proposal_boxes).squeeze(0) if len(proposal_boxes) else gt_box.new_zeros((0,))
                    if len(ious):
                        item["stage_best_iou"][stage_index] = max(item["stage_best_iou"][stage_index], float(ious.max()))
                        item["stage_iou50_count"][stage_index] += int((ious >= .50).sum())
                        item["stage_assigned_positive_count"][stage_index] += int((ious >= threshold).sum())
            if index % 100 == 0:
                print(json.dumps({"tiles": index, "total": len(dataset)}, ensure_ascii=False), flush=True)

    final_candidate = {key: _final_true_candidate(predictions, canonical, key, value["label"], tuple(value["box"]),
                                                   opt.score_threshold, opt.match_iou)
                       for key, value in evidence.items()}
    per_class, background_fn_buckets, all_background_fn_buckets = {}, {}, Counter()
    for label, name in enumerate(class_names):
        items = [(key, value) for key, value in evidence.items() if value["label"] == label]
        misses = [(key, value) for key, value in items if key not in matched]
        background_misses = [(key, value) for key, value in misses
                             if primary_cause(value, final_candidate[key]) == "roi_classified_background"]
        bucket_counts = Counter(_bucket(value["stage_best_iou"][-1]) for _, value in background_misses)
        background_fn_buckets[name] = dict(bucket_counts)
        all_background_fn_buckets.update(bucket_counts)
        stages = {}
        for stage_index, threshold in enumerate(model.roi_heads.thresholds):
            stages[f"stage{stage_index + 1}"] = {
                "classification_positive_iou": threshold,
                "best_iou": _summary([value["stage_best_iou"][stage_index] for _, value in items]),
                "iou50_candidates_per_gt": _summary([float(value["stage_iou50_count"][stage_index]) for _, value in items]),
                "assigned_positive_candidates_per_gt": _summary(
                    [float(value["stage_assigned_positive_count"][stage_index]) for _, value in items]),
            }
        per_class[name] = {"gt": len(items), "final_fn": len(misses), "background_fn": len(background_misses), "stages": stages}

    output = {
        "checkpoint": str(opt.checkpoint.resolve()), "predictions": str(opt.predictions.resolve()),
        "scope": "Inference-proposal supply before the training sampler. Training also injects visible GT boxes, so this audit measures the deployment proposal evidence and assignment gap rather than stochastic sampled counts.",
        "classification_thresholds": list(model.roi_heads.thresholds), "background_fn_final_raw_iou_buckets": dict(all_background_fn_buckets),
        "per_class_background_fn_final_raw_iou_buckets": background_fn_buckets, "per_class": per_class,
    }
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(opt.output.resolve()), "background_fn_final_raw_iou_buckets": dict(all_background_fn_buckets)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
