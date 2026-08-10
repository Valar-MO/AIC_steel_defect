#!/usr/bin/env python3
"""Audit GL-Cascade sampling, EQL state, and raw ROI class ranks on validation tiles."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gl_cascade import GLCascadeDetector, OfficialGridBalancedSampler, OfficialGridGlobalLocalDataset, gl_cascade_collate
from src.gl_cascade.boxes import box_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--global-input-size", type=int, default=1024)
    parser.add_argument("--local-input-size", type=int, default=1536)
    parser.add_argument("--match-iou", type=float, default=.50)
    return parser.parse_args()


def sampler_exposure(dataset: OfficialGridGlobalLocalDataset, class_names: list[str]) -> dict:
    sampler = OfficialGridBalancedSampler(dataset)
    expected_tile_draws = torch.zeros(len(class_names), dtype=torch.float64)
    expected_instance_draws = torch.zeros(len(class_names), dtype=torch.float64)
    available_tiles = torch.zeros(len(class_names), dtype=torch.long)
    available_instances = torch.zeros(len(class_names), dtype=torch.long)
    for weight, labels in zip(sampler.weights, dataset.tile_labels):
        for label in set(labels):
            available_tiles[label] += 1
            expected_tile_draws[label] += weight
        for label in labels:
            available_instances[label] += 1
            expected_instance_draws[label] += weight
    draws = len(dataset)
    return {
        "tiles_per_epoch": draws,
        "empty_tile_probability": sampler.empty_fraction,
        "per_class": {
            name: {
                "available_positive_tiles": int(available_tiles[index]),
                "available_visible_gt_instances": int(available_instances[index]),
                "expected_positive_tile_draws_per_epoch": float(expected_tile_draws[index] * draws),
                "expected_visible_gt_instance_draws_per_epoch": float(expected_instance_draws[index] * draws),
            }
            for index, name in enumerate(class_names)
        },
    }


def eql_state(state_dict: dict[str, torch.Tensor], class_names: list[str]) -> dict:
    if "roi_heads.eql.positive_grad" not in state_dict:
        weights = state_dict["roi_heads.class_loss.class_weights"].tolist()
        return {"mode": "softmax_background", "class_weights": {"background": weights[-1],
                **{name: weights[index] for index, name in enumerate(class_names)}}}
    positive = state_dict["roi_heads.eql.positive_grad"].double()
    negative = state_dict["roi_heads.eql.negative_grad"].double()
    ratio = positive / negative.clamp_min(1e-6)
    negative_weight = torch.sigmoid(12.0 * (ratio - .8))
    return {
        name: {
            "positive_grad": float(positive[index]),
            "negative_grad": float(negative[index]),
            "positive_negative_ratio": float(ratio[index]),
            "current_negative_weight": float(negative_weight[index]),
            "current_positive_weight": float(1 + 4 * (1 - negative_weight[index])),
        }
        for index, name in enumerate(class_names)
    }


def raw_roi_ranks(model: GLCascadeDetector, dataset: OfficialGridGlobalLocalDataset, class_names: list[str],
                  device: torch.device, workers: int, match_iou: float) -> dict:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda",
                        persistent_workers=workers > 0, collate_fn=gl_cascade_collate)
    stats = {name: defaultdict(float) for name in class_names}
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            local = batch["local"].to(device, non_blocking=True)
            global_image = batch["global"].to(device, non_blocking=True)
            target = batch["targets"][0]
            view = target["view_xyxy"].unsqueeze(0).to(device)
            image_size = target["image_size"].unsqueeze(0).to(device)
            local_features = model.backbone.forward_features(local)
            global_features = model.backbone.forward_features(global_image)
            features = model.fpn(local_features, global_features, view, image_size)
            rpn = model.rpn(features, tuple(local.shape[-2:]), targets=None)
            roi = model.roi_heads(features, rpn["proposals"], tuple(local.shape[-2:]), targets=None)
            boxes, logits = roi["boxes"][0], roi["final"]["class_logits"]
            gt_boxes, gt_labels = target["boxes"].to(device), target["labels"].to(device)
            if len(gt_boxes):
                ious = box_iou(gt_boxes, boxes) if len(boxes) else gt_boxes.new_zeros((len(gt_boxes), 0))
                for gt_index, label in enumerate(gt_labels.tolist()):
                    item = stats[class_names[label]]
                    item["visible_gt_instances"] += 1
                    if not len(boxes):
                        continue
                    best_iou, proposal_index = ious[gt_index].max(dim=0)
                    item["best_iou_sum"] += float(best_iou)
                    if best_iou < match_iou:
                        continue
                    item["proposal_covered_iou50"] += 1
                    true_logit = logits[proposal_index, label]
                    rank = int((logits[proposal_index] > true_logit).sum().item()) + 1
                    item["true_class_rank_sum"] += rank
                    item["top1"] += int(rank <= 1)
                    item["top3"] += int(rank <= 3)
                    item["top5"] += int(rank <= 5)
            if index % 100 == 0:
                print(json.dumps({"tiles": index, "total": len(dataset)}), flush=True)
    result = {}
    for name, item in stats.items():
        visible, covered = item["visible_gt_instances"], item["proposal_covered_iou50"]
        result[name] = {
            "visible_gt_instances": int(visible),
            "proposal_covered_iou50": int(covered),
            "proposal_coverage_recall": item["proposal_covered_iou50"] / max(1, visible),
            "mean_best_iou": item["best_iou_sum"] / max(1, visible),
            "true_class_mean_rank_when_covered": item["true_class_rank_sum"] / max(1, covered),
            "true_class_top1_when_covered": item["top1"] / max(1, covered),
            "true_class_top3_when_covered": item["top3"] / max(1, covered),
            "true_class_top5_when_covered": item["top5"] / max(1, covered),
        }
    return result


def main() -> None:
    opt = parse_args()
    class_names = (yaml.safe_load(opt.classes.read_text(encoding="utf-8")) or {})["names"]
    checkpoint = torch.load(opt.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = OfficialGridGlobalLocalDataset(opt.manifest, opt.classes, split="train",
                                                   global_input_size=opt.global_input_size, local_input_size=opt.local_input_size)
    val_dataset = OfficialGridGlobalLocalDataset(opt.manifest, opt.classes, split="val",
                                                 global_input_size=opt.global_input_size, local_input_size=opt.local_input_size)
    model = GLCascadeDetector(backbone_name=train_args.get("backbone", "r50_dcn"), pretrained_backbone=False,
                              internimage_root=train_args.get("internimage_root"),
                              backbone_checkpointing=bool(train_args.get("backbone_checkpointing", False)),
                              classifier_mode=train_args.get("classifier_mode", "legacy_eql")).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    report = {
        "checkpoint": str(opt.checkpoint.resolve()),
        "match_iou": opt.match_iou,
        "sampler_expected_exposure": sampler_exposure(train_dataset, class_names),
        "eql_state": eql_state(checkpoint["model"], class_names),
        "val_raw_roi_rank": raw_roi_ranks(model, val_dataset, class_names, device, opt.workers, opt.match_iou),
    }
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(opt.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
