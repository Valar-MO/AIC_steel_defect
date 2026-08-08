#!/usr/bin/env python3
"""Train the Global-Local Cascade detector on fixed official local windows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gl_cascade import GLCascadeDetector, OfficialGridGlobalLocalDataset, gl_cascade_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--global-input-size", type=int, default=1024)
    parser.add_argument("--local-input-size", type=int, default=1536)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--limit-steps", type=int, help="Smoke-test only: stop early without changing data geometry.")
    return parser.parse_args()


def _to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [{key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in target.items()}
            for target in targets]


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("The first GL-Cascade run is deliberately batch-size 1; use --accumulate for its effective batch.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OfficialGridGlobalLocalDataset(args.manifest, args.classes, split="train", global_input_size=args.global_input_size,
                                             local_input_size=args.local_input_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0, collate_fn=gl_cascade_collate)
    model = GLCascadeDetector(pretrained_backbone=args.pretrained_backbone).to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_config.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); running = {}
        for step, batch in enumerate(loader, start=1):
            local, global_image = batch["local"].to(device, non_blocking=True), batch["global"].to(device, non_blocking=True)
            targets = _to_device(batch["targets"], device)
            views = torch.stack([target["view_xyxy"] for target in targets])
            image_sizes = torch.stack([target["image_size"] for target in targets])
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(local, global_image, views, image_sizes, targets)
                loss = sum(output.losses.values()) / args.accumulate
            scaler.scale(loss).backward()
            if step % args.accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            for name, value in output.losses.items():
                running[name] = running.get(name, 0.) + float(value.detach())
            if step % 20 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "loss": round(sum(running.values()) / step, 5),
                                  "components": {name: round(value / step, 5) for name, value in running.items()}}, ensure_ascii=False), flush=True)
            if args.limit_steps and step >= args.limit_steps:
                break
        if step % args.accumulate:
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)}
        torch.save(checkpoint, args.output_dir / f"epoch_{epoch + 1:02d}.pth")
        print(json.dumps({"epoch": epoch + 1, "mean_losses": {name: value / step for name, value in running.items()},
                          "checkpoint": str(args.output_dir / f'epoch_{epoch + 1:02d}.pth')}, ensure_ascii=False), flush=True)
        if args.limit_steps:
            return


if __name__ == "__main__":
    main()
