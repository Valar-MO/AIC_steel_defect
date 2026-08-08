#!/usr/bin/env python3
"""Smoke-test DINOv2 token extraction, feature pyramid, and gradients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import Dinov2FeaturePyramid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("/root/autodl-tmp/weights/dinov2-small"),
    )
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unfreeze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_dir():
        raise FileNotFoundError(f"DINOv2 weights not found: {args.weights}")
    if args.imgsz <= 0 or args.batch <= 0 or args.channels <= 0:
        raise ValueError("imgsz, batch, and channels must be positive")

    device = torch.device(args.device)
    model = Dinov2FeaturePyramid(
        args.weights,
        out_channels=args.channels,
        freeze_backbone=not args.unfreeze,
    ).to(device)
    model.train()

    inputs = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        features = model(inputs)
        loss = sum(feature.float().square().mean() for feature in features.values())
    loss.backward()

    adapter_has_grad = any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    )
    backbone_has_grad = any(
        parameter.grad is not None for parameter in model.backbone.parameters()
    )
    report = {
        "weights": str(args.weights),
        "input": list(inputs.shape),
        "patch_size": model.patch_size,
        "selected_layers": list(model.selected_layers),
        "features": {name: list(value.shape) for name, value in features.items()},
        "freeze_backbone": model.freeze_backbone,
        "adapter_has_grad": adapter_has_grad,
        "backbone_has_grad": backbone_has_grad,
        "peak_gpu_gib": (
            round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            if device.type == "cuda"
            else None
        ),
    }
    print(json.dumps(report, indent=2))

    expected_grid = args.imgsz // model.patch_size
    expected = {
        "p3": [args.batch, args.channels, expected_grid * 2, expected_grid * 2],
        "p4": [args.batch, args.channels, expected_grid, expected_grid],
        "p5": [args.batch, args.channels, expected_grid // 2, expected_grid // 2],
    }
    actual = {name: list(value.shape) for name, value in features.items()}
    if actual != expected:
        raise RuntimeError(f"Unexpected feature shapes: expected {expected}, got {actual}")
    if not adapter_has_grad:
        raise RuntimeError("Adapter parameters did not receive gradients")
    if model.freeze_backbone and backbone_has_grad:
        raise RuntimeError("Frozen backbone unexpectedly received gradients")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
