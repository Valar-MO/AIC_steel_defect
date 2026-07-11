#!/usr/bin/env python3
"""Train DINOv2 + RT-DETR on an Ultralytics-format detection dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import YoloDetectionDataset, detection_collate
from src.models import build_dinov2_rtdetr


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = {}
    if pre_args.config:
        import yaml

        payload = yaml.safe_load(pre_args.config.read_text(encoding="utf-8")) or {}
        mapping = {
            ("model", "backbone_weights"): "dinov2_weights",
            ("model", "detector_weights"): "rtdetr_weights",
            ("model", "unfreeze_last_blocks"): "unfreeze_last_blocks",
            ("model", "selected_layers"): "selected_layers",
            ("model", "out_channels"): "out_channels",
            ("data", "yaml"): "data",
            ("data", "image_size"): "imgsz",
            ("data", "horizontal_flip"): "hflip",
            ("data", "vertical_flip"): "vflip",
            ("data", "brightness"): "brightness",
            ("data", "contrast"): "contrast",
            ("training", "batch_size"): "batch",
            ("training", "epochs"): "epochs",
            ("training", "workers"): "workers",
            ("training", "head_learning_rate"): "lr",
            ("training", "backbone_learning_rate"): "backbone_lr",
            ("training", "weight_decay"): "weight_decay",
            ("training", "grad_clip"): "grad_clip",
            ("training", "warmup_epochs"): "warmup_epochs",
            ("training", "patience"): "patience",
            ("training", "min_delta"): "min_delta",
            ("training", "save_every"): "save_every",
            ("training", "amp_dtype"): "amp_dtype",
            ("sampler", "type"): "sampler",
            ("sampler", "max_weight"): "sampler_max_weight",
            ("experiment", "project"): "project",
            ("experiment", "name"): "name",
            ("experiment", "seed"): "seed",
        }
        for keys, destination in mapping.items():
            section = payload.get(keys[0], {})
            if keys[1] in section:
                defaults[destination] = section[keys[1]]
        if "amp" in payload.get("training", {}):
            defaults["no_amp"] = not bool(payload["training"]["amp"])
        sampler_payload = payload.get("sampler", {})
        if isinstance(sampler_payload, dict):
            if "sample_type_weights" in sampler_payload:
                defaults["sampler_sample_type_weights"] = json.dumps(
                    sampler_payload["sample_type_weights"], ensure_ascii=False
                )
            if "class_weights" in sampler_payload:
                defaults["sampler_class_weights"] = json.dumps(
                    sampler_payload["class_weights"], ensure_ascii=False
                )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dinov2-weights", type=Path)
    parser.add_argument("--rtdetr-weights", type=Path)
    parser.add_argument("--selected-layers", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--out-channels", type=int, default=256)
    parser.add_argument("--project", type=Path, default=Path("/root/autodl-tmp/runs/dinov2_rtdetr"))
    parser.add_argument("--name")
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--hflip", type=float, default=0.5)
    parser.add_argument("--vflip", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-steps", type=int, help="Stop after this many optimizer steps (smoke tests).")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sampler", choices=("none", "class_aware"), default="none")
    parser.add_argument("--sampler-max-weight", type=float, default=4.0)
    parser.add_argument(
        "--sampler-sample-type-weights",
        default='{"whole": 1.0, "regular_positive": 1.0, "rare_center": 2.0, '
                '"random_background": 0.7, "hard_negative": 2.0}',
        help="JSON mapping from sample_manifest.csv sample_type to sampling weight.",
    )
    parser.add_argument(
        "--sampler-class-weights",
        default='{"huashang": 3.0, "qilie": 3.0, "zonglie": 2.0, "yanghuatiepi": 2.0}',
        help="JSON mapping from class name to sampling weight.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    parser.set_defaults(**defaults)
    return parser.parse_args()


def move_labels(labels: list[dict], device: torch.device) -> list[dict]:
    return [
        {"class_labels": item["class_labels"].to(device), "boxes": item["boxes"].to(device)}
        for item in labels
    ]


@torch.no_grad()
def validate(model, loader, device, amp_enabled: bool, amp_dtype: torch.dtype) -> float:
    model.eval()
    total_loss, batches = 0.0, 0
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(
                pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                pixel_mask=batch["pixel_mask"].to(device, non_blocking=True),
                labels=move_labels(batch["labels"], device),
            )
        total_loss += float(output.loss)
        batches += 1
    model.train()
    return total_loss / max(batches, 1)


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, step,
                    best_val, epochs_without_improvement, args) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val": best_val,
            "epochs_without_improvement": epochs_without_improvement,
            "args": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def write_history_outputs(run_dir: Path, records: list[dict]) -> None:
    if not records:
        return
    csv_path = run_dir / "history.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in records]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, [row["train_loss"] for row in records], label="train")
        axis.plot(epochs, [row["val_loss"] for row in records], label="val")
        axis.set(xlabel="Epoch", ylabel="Loss", title="DINOv2 + RT-DETR training")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(run_dir / "loss_curve.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass


def make_scheduler(optimizer, total_steps: int, warmup_steps: int, start_factor: float):
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return start_factor + (1.0 - start_factor) * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, multiplier)


def parse_weight_mapping(payload: str, name: str) -> dict[str, float]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    output = {}
    for key, value in parsed.items():
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{key} must be numeric") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{name}.{key} must be a positive finite number")
        output[str(key)] = weight
    return output


def load_sample_manifest(dataset_yaml: Path) -> dict[str, dict]:
    import yaml

    config = yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8")) or {}
    root = Path(config.get("path", Path(dataset_yaml).parent))
    if not root.is_absolute():
        root = (Path(dataset_yaml).parent / root).resolve()
    manifest = root / "sample_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Class-aware sampler requires mixed dataset sample_manifest.csv: {manifest}"
        )
    rows_by_sample = {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_path = Path(row["sample_path"])
            if not sample_path.is_absolute():
                sample_path = root / sample_path
            rows_by_sample[str(sample_path.resolve())] = row
    if not rows_by_sample:
        raise ValueError(f"Empty sample manifest: {manifest}")
    return rows_by_sample


def build_class_aware_sampler(dataset: YoloDetectionDataset, args, run_dir: Path):
    if args.sampler == "none":
        return None, None
    if args.sampler != "class_aware":
        raise ValueError(f"Unsupported sampler: {args.sampler}")
    if args.sampler_max_weight < 1.0:
        raise ValueError("--sampler-max-weight must be >= 1")

    sample_type_weights = parse_weight_mapping(
        args.sampler_sample_type_weights, "--sampler-sample-type-weights"
    )
    class_weights = parse_weight_mapping(args.sampler_class_weights, "--sampler-class-weights")
    manifest = load_sample_manifest(dataset.dataset_yaml)

    weights, missing = [], []
    sample_type_counts, sample_type_weight_mass = Counter(), Counter()
    class_counts, class_weight_mass = Counter(), Counter()
    for image_path in dataset.images:
        key = str(Path(image_path).resolve())
        row = manifest.get(key)
        if row is None:
            missing.append(key)
            weights.append(1.0)
            continue
        sample_type = row.get("sample_type", "")
        classes = [item for item in row.get("classes", "").split("|") if item]
        weight = sample_type_weights.get(sample_type, 1.0)
        for class_name in classes:
            weight = max(weight, class_weights.get(class_name, 1.0))
        weight = min(float(weight), float(args.sampler_max_weight))
        weights.append(weight)
        sample_type_counts[sample_type] += 1
        sample_type_weight_mass[sample_type] += weight
        for class_name in classes:
            class_counts[class_name] += 1
            class_weight_mass[class_name] += weight

    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"{len(missing)} training images are missing from sample_manifest.csv; "
            f"first examples: {preview}"
        )
    weight_tensor = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=len(weight_tensor),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    summary = {
        "type": "class_aware",
        "num_samples": len(weights),
        "min_weight": round(min(weights), 6),
        "mean_weight": round(sum(weights) / len(weights), 6),
        "max_weight": round(max(weights), 6),
        "sample_type_weights": sample_type_weights,
        "class_weights": class_weights,
        "sample_type_counts": dict(sample_type_counts),
        "sample_type_weight_mass": {key: round(value, 3) for key, value in sample_type_weight_mass.items()},
        "class_counts": dict(class_counts),
        "class_weight_mass": {key: round(value, 3) for key, value in class_weight_mass.items()},
    }
    (run_dir / "sampler_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return sampler, summary


def main() -> None:
    args = parse_args()
    missing = [name for name in ("data", "dinov2_weights", "rtdetr_weights", "name")
               if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}")
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.unfreeze_backbone and args.unfreeze_last_blocks:
        raise ValueError("Choose full unfreeze or --unfreeze-last-blocks, not both")
    if args.imgsz % 14:
        raise ValueError("imgsz must be divisible by DINOv2 patch size 14")
    if min(args.batch, args.epochs, args.workers + 1) <= 0:
        raise ValueError("batch/epochs must be positive and workers non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    use_scaler = amp_enabled and amp_dtype == torch.float16

    run_dir = (args.project / args.name).resolve()
    if run_dir.exists() and not args.exist_ok and args.resume is None:
        raise FileExistsError(f"Run directory exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    train_set = YoloDetectionDataset(
        args.data, "train", args.imgsz,
        horizontal_flip=args.hflip,
        vertical_flip=args.vflip,
        brightness=args.brightness,
        contrast=args.contrast,
        limit=args.max_train_samples,
    )
    val_set = YoloDetectionDataset(
        args.data, "val", args.imgsz,
        limit=args.max_val_samples,
    )
    loader_kwargs = dict(
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
        persistent_workers=args.workers > 0,
    )
    train_sampler, sampler_summary = build_class_aware_sampler(train_set, args, run_dir)
    train_loader = DataLoader(train_set, sampler=train_sampler, shuffle=train_sampler is None, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = build_dinov2_rtdetr(
        args.dinov2_weights,
        num_classes=len(train_set.names),
        out_channels=args.out_channels,
        selected_layers=tuple(args.selected_layers),
        freeze_backbone=not args.unfreeze_backbone,
        rtdetr_weights=args.rtdetr_weights,
    ).to(device)
    model.config.id2label = {index: name for index, name in enumerate(train_set.names)}
    model.config.label2id = {name: index for index, name in enumerate(train_set.names)}
    pyramid = model.model.backbone.model
    if args.unfreeze_last_blocks:
        pyramid.unfreeze_last_blocks(args.unfreeze_last_blocks)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"], strict=True)

    backbone_parameters, other_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".backbone.model.backbone." in name:
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    parameter_groups = [{"params": other_parameters, "lr": args.lr}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.backbone_lr})
    optimizer = AdamW(parameter_groups, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = round(args.warmup_epochs * len(train_loader))
    scheduler = make_scheduler(
        optimizer, max(total_steps, 1), warmup_steps, args.warmup_start_factor
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    start_epoch, global_step, best_val, epochs_without_improvement = 0, 0, math.inf, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["step"])
        best_val = float(checkpoint["best_val"])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))

    print(json.dumps({
        "run_dir": str(run_dir), "train_images": len(train_set), "val_images": len(val_set),
        "classes": train_set.names, "amp": amp_enabled,
        "amp_dtype": args.amp_dtype if amp_enabled else None,
        "sampler": sampler_summary or {"type": "none"},
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }, ensure_ascii=False, indent=2))
    history_path = run_dir / "history.jsonl"
    history_records = []
    if history_path.is_file():
        history_records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
                           if line.strip()]
    stop = False
    try:
      for epoch in range(start_epoch, args.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        running_loss, epoch_batches = 0.0, 0
        epoch_start = time.time()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(
                    pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                    pixel_mask=batch["pixel_mask"].to(device, non_blocking=True),
                    labels=move_labels(batch["labels"], device),
                )
                loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {loss}")
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_batches += 1
            running_loss += float(loss.detach())
            if global_step == 1 or global_step % args.log_interval == 0:
                print(f"epoch={epoch + 1} step={global_step} loss={float(loss.detach()):.6f} "
                      f"avg={running_loss / epoch_batches:.6f} grad_norm={float(grad_norm):.4f}", flush=True)
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

        train_loss = running_loss / max(epoch_batches, 1)
        val_loss = validate(model, val_loader, device, amp_enabled, amp_dtype)
        record = {
            "epoch": epoch + 1, "step": global_step, "train_loss": train_loss,
            "val_loss": val_loss, "lr": scheduler.get_last_lr(),
            "seconds": round(time.time() - epoch_start, 2),
            "peak_gpu_gib": (round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
                             if device.type == "cuda" else None),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        history_records.append(record)
        write_history_outputs(run_dir, history_records)
        improved = val_loss < best_val - args.min_delta
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler,
                        epoch, global_step, best_val, epochs_without_improvement, args)
        if improved:
            save_checkpoint(run_dir / "best_loss.pt", model, optimizer, scheduler, scaler,
                            epoch, global_step, best_val, epochs_without_improvement, args)
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(run_dir / f"epoch_{epoch + 1:03d}.pt", model, optimizer,
                            scheduler, scaler, epoch, global_step, best_val,
                            epochs_without_improvement, args)
        if stop:
            break
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping: val loss did not improve by {args.min_delta} "
                  f"for {args.patience} epochs.")
            break
    except (KeyboardInterrupt, Exception):
        save_checkpoint(run_dir / "interrupted.pt", model, optimizer, scheduler, scaler,
                        max(start_epoch, locals().get("epoch", start_epoch)), global_step,
                        best_val, epochs_without_improvement, args)
        raise
    print(f"Training finished. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
