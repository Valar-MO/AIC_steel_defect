#!/usr/bin/env python3
"""Measure one real full-resolution GL-Cascade optimization step."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gl_cascade import GLCascadeDetector, OfficialGridGlobalLocalDataset, gl_cascade_collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--global-input-size", type=int, default=1024)
    parser.add_argument("--local-input-size", type=int, default=1536)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    dataset = OfficialGridGlobalLocalDataset(args.manifest, args.classes, split="train", global_input_size=args.global_input_size,
                                             local_input_size=args.local_input_size)
    index = next(index for index, labels in enumerate(dataset.tile_labels) if labels)
    batch = gl_cascade_collate([dataset[index]])
    device = torch.device("cuda")
    model = GLCascadeDetector(pretrained_backbone=False).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    local, global_image = batch["local"].to(device), batch["global"].to(device)
    targets = [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch["targets"][0].items()}]
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.amp.autocast("cuda"):
        output = model(local, global_image, targets[0]["view_xyxy"].unsqueeze(0), targets[0]["image_size"].unsqueeze(0), targets)
        loss = sum(output.losses.values())
    loss.backward(); optimizer.step(); torch.cuda.synchronize()
    print({"loss": float(loss.detach()), "seconds": round(time.perf_counter() - start, 2),
           "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
           "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)})


if __name__ == "__main__":
    main()
