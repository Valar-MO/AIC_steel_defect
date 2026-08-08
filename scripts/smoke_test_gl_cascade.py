#!/usr/bin/env python3
"""CPU/GPU smoke test for the Global-Local Cascade forward and losses."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gl_cascade.model import GLCascadeDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=256, help="Small spatial size for a fast structural smoke test.")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = GLCascadeDetector(pretrained_backbone=False).to(device)
    model.train()
    local = torch.randn(1, 3, args.size, args.size, device=device)
    global_image = torch.randn(1, 3, args.size, args.size, device=device)
    target = {"boxes": torch.tensor([[32., 40., 130., 90.]], device=device), "labels": torch.tensor([5], device=device),
              "visible_fraction": torch.tensor([1.], device=device), "boundary_weight": torch.tensor([1.], device=device),
              "ignore_boxes": torch.zeros((0, 4), device=device)}
    output = model(local, global_image, torch.tensor([[0., 0., 2048., 2048.]], device=device),
                   torch.tensor([[4096., 4096.]], device=device), [target])
    loss = sum(output.losses.values())
    loss.backward()
    print({"losses": {key: round(float(value.detach()), 5) for key, value in output.losses.items()},
           "proposal_count": len(output.proposals[0]), "loss_finite": bool(torch.isfinite(loss))})
    model.eval()
    with torch.no_grad():
        prediction = model(local, global_image, torch.tensor([[0., 0., 2048., 2048.]], device=device),
                           torch.tensor([[4096., 4096.]], device=device))
    print({"detections": len(prediction.detections[0]), "score_keys": sorted(prediction.detections[0])})


if __name__ == "__main__":
    main()
