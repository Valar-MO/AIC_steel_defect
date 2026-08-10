#!/usr/bin/env python3
"""Attribute GL-Cascade validation false negatives to their earliest failed stage.

The audit is deliberately read-only.  It replays official validation tiles from
a checkpoint, then joins their evidence with the already merged predictions
used by the evaluator.  A ground-truth instance is keyed in original-image
coordinates, so appearances in overlapping tiles are evidence for one GT, not
multiple independent misses.
"""
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
from evaluate_predictions import load_classes, load_predictions  # noqa: E402
from src.gl_cascade import GLCascadeDetector, OfficialGridGlobalLocalDataset, gl_cascade_collate  # noqa: E402
from src.gl_cascade.boxes import box_iou  # noqa: E402
from src.gl_cascade.merge import map_tile_to_original  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Merged validation predictions produced from this checkpoint.")
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--global-input-size", type=int, default=1024)
    parser.add_argument("--local-input-size", type=int, default=1536)
    parser.add_argument("--score-threshold", type=float, default=.30)
    parser.add_argument("--match-iou", type=float, default=.50)
    return parser.parse_args()


def visible_targets(record, local_size: int) -> list[tuple[int, int, torch.Tensor]]:
    """Return original GT index, label, and visible local box for one official tile."""
    x1, y1, x2, y2 = record.view
    scale_x, scale_y = local_size / (x2 - x1), local_size / (y2 - y1)
    targets = []
    for gt_index, (label, box) in enumerate(record.boxes):
        ix1, iy1, ix2, iy2 = max(box[0], x1), max(box[1], y1), min(box[2], x2), min(box[3], y2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        fraction = (ix2 - ix1) * (iy2 - iy1) / ((box[2] - box[0]) * (box[3] - box[1]))
        if fraction < .5:
            continue
        targets.append((gt_index, label, torch.tensor(((ix1 - x1) * scale_x, (iy1 - y1) * scale_y,
                                                        (ix2 - x1) * scale_x, (iy2 - y1) * scale_y))))
    return targets


def greedy_matched_gt(predictions: dict[int, list[tuple[float, str, tuple[float, ...]]]],
                      canonical: dict[str, list[tuple[int, tuple[float, ...]]]],
                      score_threshold: float, iou_threshold: float) -> set[tuple[str, int]]:
    """Mirror the evaluator's classwise score-ranked matching at the point threshold."""
    matched: set[tuple[str, int]] = set()
    for label, items in predictions.items():
        for score, image_id, box in items:
            if score < score_threshold:
                continue
            best_index, best_iou = -1, -1.0
            for index, (gt_label, gt_box) in enumerate(canonical[image_id]):
                if gt_label != label or (image_id, index) in matched:
                    continue
                iou = scalar_iou(box, gt_box)
                if iou > best_iou:
                    best_index, best_iou = index, iou
            if best_index >= 0 and best_iou >= iou_threshold:
                matched.add((image_id, best_index))
    return matched


def scalar_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def init_evidence(dataset: OfficialGridGlobalLocalDataset) -> tuple[dict[str, list[tuple[int, tuple[float, ...]]]], dict]:
    canonical: dict[str, list[tuple[int, tuple[float, ...]]]] = {}
    evidence = {}
    for record in dataset.records:
        image_id = record.image_path.name
        if image_id in canonical:
            continue
        canonical[image_id] = record.boxes
        for index, (label, box) in enumerate(record.boxes):
            evidence[(image_id, index)] = {"label": label, "box": list(box), "visible_tiles": 0,
                                            "rpn_covered": False, "roi_covered": False,
                                            "true_class": False, "background_class": False,
                                            "wrong_defect_class": False, "foreground_veto": False,
                                            "local_true_detection": False, "local_true_score": False}
    return canonical, evidence


def update_tile_evidence(evidence: dict, image_id: str, record, rpn_boxes: torch.Tensor, roi_boxes: torch.Tensor,
                         logits: torch.Tensor, foreground: torch.Tensor, local_detections: dict,
                         local_size: int, match_iou: float, background_index: int, score_threshold: float) -> None:
    for gt_index, label, gt_box in visible_targets(record, local_size):
        item = evidence[(image_id, gt_index)]
        item["visible_tiles"] += 1
        rpn_iou = box_iou(gt_box.unsqueeze(0).to(rpn_boxes), rpn_boxes).max().item() if len(rpn_boxes) else 0.
        item["rpn_covered"] |= rpn_iou >= match_iou
        if not len(roi_boxes):
            continue
        ious = box_iou(gt_box.unsqueeze(0).to(roi_boxes), roi_boxes).squeeze(0)
        covered = torch.where(ious >= match_iou)[0]
        if not len(covered):
            continue
        item["roi_covered"] = True
        probabilities = logits[covered].softmax(dim=1)
        all_labels = probabilities.argmax(dim=1)
        defect_labels = probabilities[:, :background_index].argmax(dim=1)
        item["true_class"] |= bool((all_labels == label).any())
        item["background_class"] |= bool((all_labels == background_index).any())
        item["wrong_defect_class"] |= bool(((all_labels != label) & (all_labels != background_index)).any())
        matching = covered[defect_labels == label]
        if len(matching):
            item["foreground_veto"] |= bool((foreground[matching].sigmoid() < .05).any())
        if len(local_detections["boxes"]):
            det_iou = box_iou(gt_box.unsqueeze(0).to(local_detections["boxes"]), local_detections["boxes"]).squeeze(0)
            true = (local_detections["labels"] == label) & (det_iou >= match_iou)
            item["local_true_detection"] |= bool(true.any())
            item["local_true_score"] |= bool((true & (local_detections["scores"] >= score_threshold)).any())


def primary_cause(item: dict, has_final_true_candidate: bool) -> str:
    if not item["rpn_covered"]:
        return "rpn_no_iou50_proposal"
    if not item["roi_covered"]:
        return "cascade_localization_lost"
    if not item["true_class"]:
        if item["background_class"] and not item["wrong_defect_class"]:
            return "roi_classified_background"
        return "roi_wrong_defect_class"
    if not item["local_true_detection"]:
        return "foreground_veto_or_local_nms"
    if not item["local_true_score"]:
        return "local_score_below_threshold"
    if not has_final_true_candidate:
        return "merge_border_or_softnms_score_loss"
    return "final_matching_conflict_duplicate"


def main() -> None:
    opt = parse_args()
    if not 0 <= opt.score_threshold <= 1:
        raise ValueError("score threshold must be in [0, 1]")
    class_names = load_classes(opt.classes)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    checkpoint = torch.load(opt.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    dataset = OfficialGridGlobalLocalDataset(opt.manifest, opt.classes, split="val", global_input_size=opt.global_input_size,
                                             local_input_size=opt.local_input_size)
    canonical, evidence = init_evidence(dataset)
    sizes = {image_id: (record.width, record.height) for record in dataset.records for image_id in [record.image_path.name]}
    predictions, _, _, _ = load_predictions(opt.predictions, class_names, sizes, set(canonical), allow_missing=False)
    matched = greedy_matched_gt(predictions, canonical, opt.score_threshold, opt.match_iou)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GLCascadeDetector(backbone_name=train_args.get("backbone", "r50_dcn"), pretrained_backbone=False,
                              internimage_root=train_args.get("internimage_root"),
                              backbone_checkpointing=bool(train_args.get("backbone_checkpointing", False)),
                              classifier_mode=train_args.get("classifier_mode", "softmax_background")).to(device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=device.type == "cuda",
                        persistent_workers=opt.workers > 0, collate_fn=gl_cascade_collate)
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            local, global_image = batch["local"].to(device, non_blocking=True), batch["global"].to(device, non_blocking=True)
            target, record = batch["targets"][0], dataset.records[index - 1]
            view, image_size = target["view_xyxy"].unsqueeze(0).to(device), target["image_size"].unsqueeze(0).to(device)
            local_features = model.backbone.forward_features(local)
            global_features = model.backbone.forward_features(global_image)
            features = model.fpn(local_features, global_features, view, image_size)
            rpn = model.rpn(features, tuple(local.shape[-2:]), targets=None)
            roi = model.roi_heads(features, rpn["proposals"], tuple(local.shape[-2:]), targets=None)
            local_detections = model._detections(roi)[0]
            update_tile_evidence(evidence, target["image_id"], record, rpn["proposals"][0], roi["boxes"][0],
                                 roi["final"]["class_logits"], roi["final"]["foreground"], local_detections,
                                 opt.local_input_size, opt.match_iou, len(class_names), opt.score_threshold)
            if index % 100 == 0:
                print(json.dumps({"tiles": index, "total": len(dataset)}, ensure_ascii=False), flush=True)
    final_true_candidate = {}
    for image_id, items in canonical.items():
        for gt_index, (label, gt_box) in enumerate(items):
            final_true_candidate[(image_id, gt_index)] = any(
                score >= opt.score_threshold and candidate_image == image_id and scalar_iou(box, gt_box) >= opt.match_iou
                for score, candidate_image, box in predictions[label])
    per_class, examples = {}, {}
    for label, name in enumerate(class_names):
        items = [(key, value) for key, value in evidence.items() if value["label"] == label]
        misses = [(key, value) for key, value in items if key not in matched]
        causes = Counter(primary_cause(value, final_true_candidate[key]) for key, value in misses)
        per_class[name] = {"gt": len(items), "final_tp": len(items) - len(misses), "final_fn": len(misses),
                           "fn_primary_causes": dict(causes),
                           "rpn_iou50_coverage": sum(value["rpn_covered"] for _, value in items),
                           "roi_iou50_coverage": sum(value["roi_covered"] for _, value in items),
                           "roi_true_class_when_covered": sum(value["true_class"] for _, value in items if value["roi_covered"])}
        examples[name] = [{"image_id": key[0], "gt_index": key[1], "box": value["box"],
                           "cause": primary_cause(value, final_true_candidate[key]),
                           "visible_tiles": value["visible_tiles"]}
                          for key, value in misses]
    output = {"checkpoint": str(opt.checkpoint.resolve()), "predictions": str(opt.predictions.resolve()),
              "score_threshold": opt.score_threshold, "match_iou": opt.match_iou,
              "matching_note": "Final TP/FN uses evaluator-equivalent classwise score-ranked matching.",
              "per_class": per_class, "false_negative_examples": examples}
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(opt.output.resolve()), "weak": {name: per_class[name] for name in ("huashang", "qilie", "zonglie")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
