#!/usr/bin/env python3
"""Train the Global-Local Cascade detector on fixed official local windows."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gl_cascade import GLCascadeDetector, OfficialGridBalancedSampler, OfficialGridGlobalLocalDataset, gl_cascade_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=.10)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-lr-mult", type=float, default=.05)
    parser.add_argument("--empty-fraction", type=float, default=.55)
    parser.add_argument("--global-input-size", type=int, default=1024)
    parser.add_argument("--local-input-size", type=int, default=1536)
    parser.add_argument("--backbone", choices=("internimage_s", "r50_dcn"), default="internimage_s")
    parser.add_argument("--internimage-root", type=Path, default=Path("/root/autodl-tmp/InternImage"))
    parser.add_argument("--internimage-checkpoint", type=Path, default=Path("/root/autodl-tmp/weights/internimage_s/internimage_s_1k_224.pth"))
    parser.add_argument("--backbone-checkpointing", action="store_true")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--val-workers", type=int, default=4)
    parser.add_argument("--limit-steps", type=int, help="Smoke-test only: stop early without changing data geometry.")
    return parser.parse_args()


def _to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [{key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in target.items()}
            for target in targets]


def _parameter_groups(model: GLCascadeDetector, base_lr: float, backbone_lr_mult: float,
                      pretrained_backbone: bool) -> list[dict]:
    pretrained, new_backbone, heads = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.") and not name.startswith("backbone.adapters."):
            (pretrained if pretrained_backbone else new_backbone).append(parameter)
        elif name.startswith("backbone.adapters."):
            new_backbone.append(parameter)
        else:
            heads.append(parameter)
    groups = [{"params": pretrained, "lr": base_lr * backbone_lr_mult, "name": "pretrained_backbone"},
              {"params": new_backbone, "lr": base_lr, "name": "new_backbone"},
              {"params": heads, "lr": base_lr, "name": "detector_heads"}]
    return [group for group in groups if group["params"]]


def _scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_lr_mult: float):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_mult + (1 - min_lr_mult) * .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _validate(opt: argparse.Namespace, checkpoint: Path, epoch: int) -> dict:
    epoch_dir = opt.output_dir / "val" / f"epoch_{epoch:02d}"
    predictions, report = epoch_dir / "merged_predictions.json", epoch_dir / "report.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "predict_gl_cascade.py"), "--checkpoint", str(checkpoint),
                    "--manifest", str(opt.manifest), "--classes", str(opt.classes), "--split", "val", "--output", str(predictions),
                    "--workers", str(opt.val_workers), "--global-input-size", str(opt.global_input_size),
                    "--local-input-size", str(opt.local_input_size)], check=True)
    subprocess.run([sys.executable, str(root / "scripts" / "evaluate_gl_cascade.py"), "--predictions", str(predictions),
                    "--manifest", str(opt.manifest), "--classes", str(opt.classes), "--split", "val", "--score-threshold", ".30",
                    "--output", str(report)], check=True)
    return json.loads(report.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.accumulate <= 0:
        raise ValueError("batch-size and accumulate must be positive")
    if args.backbone == "r50_dcn" and not args.pretrained_backbone:
        raise ValueError("Refusing to train randomly initialized r50_dcn. Pass --pretrained-backbone or use internimage_s with its official checkpoint.")
    if args.backbone == "internimage_s" and not args.internimage_checkpoint.is_file():
        raise FileNotFoundError(f"InternImage-S checkpoint is required but missing: {args.internimage_checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OfficialGridGlobalLocalDataset(args.manifest, args.classes, split="train", global_input_size=args.global_input_size,
                                             local_input_size=args.local_input_size)
    sampler = OfficialGridBalancedSampler(dataset, empty_fraction=args.empty_fraction)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0, collate_fn=gl_cascade_collate)
    use_pretrained = args.pretrained_backbone or args.backbone == "internimage_s"
    model = GLCascadeDetector(backbone_name=args.backbone, pretrained_backbone=use_pretrained,
                              internimage_root=str(args.internimage_root), internimage_checkpoint=str(args.internimage_checkpoint),
                              backbone_checkpointing=args.backbone_checkpointing).to(device)
    optimizer = torch.optim.AdamW(_parameter_groups(model, args.learning_rate, args.backbone_lr_mult, use_pretrained),
                                  weight_decay=args.weight_decay)
    optimizer_steps_per_epoch = math.ceil(len(loader) / args.accumulate)
    scheduler = _scheduler(optimizer, args.epochs * optimizer_steps_per_epoch,
                           math.ceil(args.warmup_epochs * optimizer_steps_per_epoch), args.min_lr_mult)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
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
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            for name, value in output.losses.items():
                running[name] = running.get(name, 0.) + float(value.detach())
            if step % 20 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "loss": round(sum(running.values()) / step, 5),
                                  "components": {name: round(value / step, 5) for name, value in running.items()}}, ensure_ascii=False), flush=True)
            if args.limit_steps and step >= args.limit_steps:
                break
        if step % args.accumulate:
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "args": vars(args)}
        checkpoint_path = args.output_dir / f"epoch_{epoch + 1:02d}.pth"
        torch.save(checkpoint, checkpoint_path)
        summary = {"epoch": epoch + 1, "mean_losses": {name: value / step for name, value in running.items()}, "checkpoint": str(checkpoint_path)}
        if not args.limit_steps and args.validate_every and (epoch + 1) % args.validate_every == 0:
            summary["validation"] = _validate(args, checkpoint_path, epoch + 1)
            f2 = float(summary["validation"]["overall"]["f2"])
            best_path, best_meta = args.output_dir / "best_f2.pth", args.output_dir / "best_f2.json"
            previous = json.loads(best_meta.read_text(encoding="utf-8"))["f2"] if best_meta.exists() else float("-inf")
            if f2 > previous:
                shutil.copy2(checkpoint_path, best_path)
                best_meta.write_text(json.dumps({"epoch": epoch + 1, "f2": f2, "source_checkpoint": str(checkpoint_path),
                                                  "validation_report": str(args.output_dir / "val" / f"epoch_{epoch + 1:02d}" / "report.json")},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
                summary["best_f2_updated"] = True
        summary["learning_rates"] = {group["name"]: group["lr"] for group in optimizer.param_groups}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if args.limit_steps:
            return


if __name__ == "__main__":
    main()
