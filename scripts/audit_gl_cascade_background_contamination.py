#!/usr/bin/env python3
"""Measure train-only semantic contamination in Cascade background samples.

The audit replays one deterministic sampler trace with the same GT injection,
positive fraction, and per-stage assignment used during training.  It reports
IoG/IoR for valid background candidates before sampling and for candidates
actually selected by that trace; it never updates weights.
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
from audit_gl_cascade_fn_path import load_classes  # noqa: E402
from src.gl_cascade import GLCascadeDetector, OfficialGridGlobalLocalDataset, gl_cascade_collate  # noqa: E402
from src.gl_cascade.boxes import box_iou  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--global-input-size", type=int, default=1024)
    parser.add_argument("--local-input-size", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def bucket(value: float) -> str:
    if value >= .8:
        return ">=0.80"
    if value >= .5:
        return "0.50-0.80"
    if value >= .2:
        return "0.20-0.50"
    return "<0.20"


def intersection_metrics(boxes: torch.Tensor, gt_boxes: torch.Tensor, matched: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = gt_boxes[matched]
    x1, y1 = torch.maximum(boxes[:, 0], target[:, 0]), torch.maximum(boxes[:, 1], target[:, 1])
    x2, y2 = torch.minimum(boxes[:, 2], target[:, 2]), torch.minimum(boxes[:, 3], target[:, 3])
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    gt_area = (target[:, 2] - target[:, 0]).clamp(min=1) * (target[:, 3] - target[:, 1]).clamp(min=1)
    roi_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=1) * (boxes[:, 3] - boxes[:, 1]).clamp(min=1)
    return intersection / gt_area, intersection / roi_area


def main() -> None:
    opt = parse_args(); torch.manual_seed(opt.seed)
    classes = load_classes(opt.classes)
    checkpoint = torch.load(opt.checkpoint, map_location="cpu", weights_only=False); args = checkpoint.get("args", {})
    dataset = OfficialGridGlobalLocalDataset(opt.manifest, opt.classes, split="train", global_input_size=opt.global_input_size,
                                             local_input_size=opt.local_input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GLCascadeDetector(backbone_name=args.get("backbone", "r50_dcn"), pretrained_backbone=False,
                              internimage_root=args.get("internimage_root"), backbone_checkpointing=bool(args.get("backbone_checkpointing", False)),
                              classifier_mode=args.get("classifier_mode", "softmax_background")).to(device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=device.type == "cuda",
                        persistent_workers=opt.workers > 0, collate_fn=gl_cascade_collate)
    stats = defaultdict(lambda: {"pool": Counter(), "sampled": Counter(), "iog_sum": 0., "ior_sum": 0., "count": 0})
    with torch.inference_mode():
        processed = 0
        for batch in loader:
            local, global_image = batch["local"].to(device, non_blocking=True), batch["global"].to(device, non_blocking=True)
            targets = [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()}
                       for target in batch["targets"]]
            view, image_size = torch.stack([target["view_xyxy"] for target in targets]), torch.stack([target["image_size"] for target in targets])
            features = model.fpn(model.backbone.forward_features(local), model.backbone.forward_features(global_image), view, image_size)
            rpn = model.rpn(features, tuple(local.shape[-2:]), targets=None)
            boxes = [torch.cat((proposal, target["boxes"]), dim=0) for proposal, target in zip(rpn["proposals"], targets)]
            for stage_index, (stage, threshold) in enumerate(zip(model.roi_heads.stages, model.roi_heads.thresholds), start=1):
                train_boxes = []
                for offset, (current, target) in enumerate(zip(boxes, targets)):
                    matched, positive, _, weight = model.roi_heads._assign(current, target, threshold)
                    background = (~positive) & (weight > 0)
                    torch.manual_seed(opt.seed + processed + offset)
                    selected = model.roi_heads._sample(positive, weight)
                    selected_background = torch.zeros(len(current), dtype=torch.bool, device=device); selected_background[selected] = background[selected]
                    if background.any() and len(target["boxes"]):
                        indices = torch.where(background)[0]
                        iog, ior = intersection_metrics(current[indices], target["boxes"], matched[indices])
                        labels = target["labels"][matched[indices]]
                        for local_index, label in enumerate(labels.tolist()):
                            key = (f"stage{stage_index}", classes[label])
                            stats[key]["pool"][bucket(float(iog[local_index]))] += 1
                            stats[key]["iog_sum"] += float(iog[local_index]); stats[key]["ior_sum"] += float(ior[local_index]); stats[key]["count"] += 1
                            if selected_background[indices[local_index]]:
                                stats[key]["sampled"][bucket(float(iog[local_index]))] += 1
                    train_boxes.append(current[selected])
                output = stage(features, train_boxes)
                boxes = model.roi_heads._split_and_refine(train_boxes, output["box_delta"], tuple(local.shape[-2:]))
            processed += len(targets)
            if processed % 100 == 0:
                print(json.dumps({"tiles": processed, "total": len(dataset)}, ensure_ascii=False), flush=True)
    output = {"checkpoint": str(opt.checkpoint.resolve()), "split": "train", "sampler_trace_seed": opt.seed,
              "note": "Background means the current Cascade assignment's valid non-positive candidate. Pool is before sampler; sampled is one deterministic trace.", "stages": {}}
    for stage in ("stage1", "stage2", "stage3"):
        output["stages"][stage] = {}
        for name in classes:
            value = stats[(stage, name)]
            output["stages"][stage][name] = {"pool_iog_buckets": dict(value["pool"]), "sampled_iog_buckets": dict(value["sampled"]),
                                               "pool_count": value["count"], "mean_iog": value["iog_sum"] / max(1, value["count"]),
                                               "mean_ior": value["ior_sum"] / max(1, value["count"])}
    opt.output.parent.mkdir(parents=True, exist_ok=True); opt.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(opt.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
