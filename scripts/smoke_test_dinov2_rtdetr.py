#!/usr/bin/env python3
"""Run an end-to-end inference smoke test for DINOv2 + RT-DETR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import build_dinov2_rtdetr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dinov2-weights",
        type=Path,
        default=Path("/root/autodl-tmp/weights/dinov2-small"),
    )
    parser.add_argument("--rtdetr-weights", type=Path)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = build_dinov2_rtdetr(
        args.dinov2_weights,
        num_classes=args.num_classes,
        freeze_backbone=True,
        rtdetr_weights=args.rtdetr_weights,
    ).to(device).eval()
    pixel_values = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    pixel_mask = torch.ones(1, args.imgsz, args.imgsz, dtype=torch.bool, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    report = {
        "logits": list(outputs.logits.shape),
        "pred_boxes": list(outputs.pred_boxes.shape),
        "finite_logits": bool(torch.isfinite(outputs.logits).all()),
        "finite_boxes": bool(torch.isfinite(outputs.pred_boxes).all()),
        "peak_gpu_gib": (
            round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            if device.type == "cuda"
            else None
        ),
    }
    print(json.dumps(report, indent=2))
    expected_logits = [1, model.config.num_queries, args.num_classes]
    expected_boxes = [1, model.config.num_queries, 4]
    if report["logits"] != expected_logits or report["pred_boxes"] != expected_boxes:
        raise RuntimeError(f"Unexpected detector output: {report}")
    if not report["finite_logits"] or not report["finite_boxes"]:
        raise RuntimeError("Detector produced non-finite outputs")
    print("End-to-end smoke test passed.")


if __name__ == "__main__":
    main()
