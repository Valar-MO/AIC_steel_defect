#!/usr/bin/env python3
"""Run GL-Cascade over official local tiles and merge them in original coordinates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gl_cascade import GLCascadeDetector, OfficialGridGlobalLocalDataset, gl_cascade_collate
from src.gl_cascade.merge import group_and_merge


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--global-input-size", type=int, default=1024)
    parser.add_argument("--local-input-size", type=int, default=1536)
    parser.add_argument("--min-detection-score", type=float, default=.001)
    parser.add_argument("--border-penalty", type=float, default=.5)
    parser.add_argument("--soft-nms-iou", type=float, default=.80)
    return parser.parse_args()


def main() -> None:
    opt = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = (yaml.safe_load(opt.classes.read_text(encoding="utf-8")) or {})["names"]
    dataset = OfficialGridGlobalLocalDataset(opt.manifest, opt.classes, split=opt.split, global_input_size=opt.global_input_size,
                                             local_input_size=opt.local_input_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=device.type == "cuda",
                        persistent_workers=opt.workers > 0, collate_fn=gl_cascade_collate)
    checkpoint = torch.load(opt.checkpoint, map_location="cpu", weights_only=False)
    model = GLCascadeDetector(pretrained_backbone=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    records = []
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            local, global_image = batch["local"].to(device, non_blocking=True), batch["global"].to(device, non_blocking=True)
            target = batch["targets"][0]
            output = model(local, global_image, target["view_xyxy"].unsqueeze(0).to(device), target["image_size"].unsqueeze(0).to(device))
            records.append((target["image_id"], output.detections[0], target["view_xyxy"].to(device), target["image_size"].to(device)))
            if index % 100 == 0:
                print(json.dumps({"tiles": index, "total": len(dataset)}, ensure_ascii=False), flush=True)
    merged = group_and_merge(records, opt.local_input_size, opt.border_penalty, opt.soft_nms_iou)
    images = []
    for image_id in sorted(merged):
        prediction = merged[image_id]
        keep = prediction["scores"] >= opt.min_detection_score
        images.append({"image_id": image_id, "predictions": [{"category_name": names[int(label)], "class_id": int(label),
                       "bbox_xyxy": [round(float(value), 4) for value in box], "score": round(float(score), 8),
                       "ranking_logit": round(float(rank), 8)} for box, label, score, rank in zip(
                           prediction["boxes"][keep].cpu(), prediction["labels"][keep].cpu(), prediction["scores"][keep].cpu(), prediction["ranking_logits"][keep].cpu())]})
    payload = {"metadata": {"classes": names, "split": opt.split, "global_input_size": opt.global_input_size,
                            "local_input_size": opt.local_input_size, "border_penalty": opt.border_penalty,
                            "soft_nms_iou": opt.soft_nms_iou, "score_mode": "detection_score_ranking_separated"}, "images": images}
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(images), "output": str(opt.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
