#!/usr/bin/env python3
"""Train a DINOv2 crop classifier for detector/classifier decoupling.

Expected dataset layout is produced by ``scripts/build_crop_cls_dataset.py``:

    crop_root/
      train/{class_name}/*.jpg
      val/{class_name}/*.jpg

The model is intentionally simple for the first validation experiment:

    DINOv2 backbone -> CLS token -> dropout -> linear 9-class head
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/root/autodl-tmp/datasets/AIC_steel_crop_cls_v1"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--dinov2-weights", type=Path, default=Path("/root/autodl-tmp/weights/dinov2-small"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/runs/crop_cls/dinov2_small_v1"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3, help="Classifier-head learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=0.5,
                        help="Cross-entropy class weights use (max_count/class_count)^power.")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-steps", type=int, help="Smoke-test stop after N optimizer steps.")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid classes file: {path}")
    return [str(name) for name in names]


class CropClassificationDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        classes: list[str],
        *,
        image_size: int,
        train: bool,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.classes = classes
        self.class_to_id = {name: idx for idx, name in enumerate(classes)}
        self.image_size = int(image_size)
        self.train = bool(train)
        samples: list[tuple[Path, int]] = []
        split_root = self.root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_root}")
        for class_name in classes:
            class_dir = split_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, self.class_to_id[class_name]))
        if max_samples is not None and len(samples) > max_samples:
            rng = random.Random(seed)
            rng.shuffle(samples)
            samples = samples[:max_samples]
            samples.sort(key=lambda item: str(item[0]))
        if not samples:
            raise ValueError(f"No images found in {split_root}")
        self.samples = samples
        self.targets = [target for _, target in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: Image.Image) -> Image.Image:
        if not self.train:
            return image
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.10:
            image = ImageOps.flip(image)
        if random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.20))
        if random.random() < 0.20:
            image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.75, 1.5))
        return image

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        if self.train:
            scale = random.uniform(0.85, 1.0)
            resize_to = max(self.image_size, int(round(self.image_size / scale)))
            image = image.resize((resize_to, resize_to), Image.Resampling.BICUBIC)
            if resize_to > self.image_size:
                left = random.randint(0, resize_to - self.image_size)
                top = random.randint(0, resize_to - self.image_size)
                image = image.crop((left, top, left + self.image_size, top + self.image_size))
        else:
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        data = data.view(image.height, image.width, 3).permute(2, 0, 1).float().div_(255.0)
        return (data - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = self._augment(image)
            tensor = self._to_tensor(image)
        return tensor, torch.tensor(target, dtype=torch.long)


class Dinov2CropClassifier(nn.Module):
    def __init__(self, weights: Path, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import Dinov2Model

        self.backbone = Dinov2Model.from_pretrained(str(weights), local_files_only=True)
        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone.requires_grad_(trainable)
        if not trainable:
            self.backbone.eval()

    def unfreeze_last_blocks(self, count: int) -> None:
        layers = self.backbone.encoder.layer
        if not 0 <= count <= len(layers):
            raise ValueError(f"--unfreeze-last-blocks must be in [0, {len(layers)}], got {count}")
        self.backbone.requires_grad_(False)
        if count == 0:
            self.backbone.eval()
            return
        for layer in layers[-count:]:
            layer.requires_grad_(True)
        if hasattr(self.backbone, "layernorm"):
            self.backbone.layernorm.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls_token))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_counts(dataset: CropClassificationDataset, classes: list[str]) -> Counter:
    counts = Counter({name: 0 for name in classes})
    for target in dataset.targets:
        counts[classes[target]] += 1
    return counts


def make_balanced_sampler(dataset: CropClassificationDataset) -> WeightedRandomSampler:
    counts = Counter(dataset.targets)
    weights = [1.0 / counts[target] for target in dataset.targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def make_class_weights(dataset: CropClassificationDataset, classes: list[str], power: float) -> torch.Tensor:
    counts = class_counts(dataset, classes)
    max_count = max(counts.values())
    weights = []
    for name in classes:
        count = max(1, counts[name])
        weights.append((max_count / count) ** power)
    values = torch.tensor(weights, dtype=torch.float32)
    return values / values.mean()


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return max(1e-3, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


@torch.inference_mode()
def evaluate(model, loader, device, num_classes: int, amp: bool, amp_dtype: torch.dtype):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits.float(), targets)
        preds = logits.argmax(dim=1)
        for target, pred in zip(targets.cpu(), preds.cpu()):
            confusion[int(target), int(pred)] += 1
        total_loss += float(loss.item())
        total += int(targets.numel())
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(1, total)
    return metrics, confusion


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, float]:
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    total = confusion.sum().clamp_min(1)
    accuracy = tp.sum() / total
    valid = support > 0
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision[valid].mean()) if valid.any() else 0.0,
        "macro_recall": float(recall[valid].mean()) if valid.any() else 0.0,
        "macro_f1": float(f1[valid].mean()) if valid.any() else 0.0,
    }


def per_class_report(confusion: torch.Tensor, classes: list[str]) -> list[dict]:
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    rows = []
    for idx, name in enumerate(classes):
        rows.append(
            {
                "class_id": idx,
                "class_name": name,
                "support": int(support[idx].item()),
                "predicted": int(predicted[idx].item()),
                "precision": float(precision[idx].item()),
                "recall": float(recall[idx].item()),
                "f1": float(f1[idx].item()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(path: Path, overwrite: bool, exist_ok: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif not exist_ok:
            raise FileExistsError(f"Output directory exists: {path}. Use --overwrite or --exist-ok.")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    classes = load_classes(args.classes)
    output = args.output.resolve()
    prepare_output(output, args.overwrite, args.exist_ok)

    train_dataset = CropClassificationDataset(
        args.data, "train", classes, image_size=args.image_size, train=True,
        max_samples=args.max_train_samples, seed=args.seed,
    )
    val_dataset = CropClassificationDataset(
        args.data, "val", classes, image_size=args.image_size, train=False,
        max_samples=args.max_val_samples, seed=args.seed,
    )
    sampler = None if args.no_balanced_sampler else make_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = Dinov2CropClassifier(args.dinov2_weights, len(classes), dropout=args.dropout).to(device)
    if args.freeze_backbone:
        model.set_backbone_trainable(False)
    if args.unfreeze_last_blocks:
        model.unfreeze_last_blocks(args.unfreeze_last_blocks)

    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    optimizer_groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        optimizer_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = AdamW(optimizer_groups, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(args.warmup_epochs * max(1, len(train_loader)))
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)

    class_weights = make_class_weights(train_dataset, classes, args.class_weight_power).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda" and amp_dtype == torch.float16
    )

    config = {
        "data": str(args.data.resolve()),
        "classes": classes,
        "dinov2_weights": str(args.dinov2_weights.resolve()),
        "image_size": args.image_size,
        "batch": args.batch,
        "epochs": args.epochs,
        "freeze_backbone": args.freeze_backbone,
        "unfreeze_last_blocks": args.unfreeze_last_blocks,
        "balanced_sampler": sampler is not None,
        "class_weight_power": args.class_weight_power,
        "train_counts": dict(class_counts(train_dataset, classes)),
        "val_counts": dict(class_counts(val_dataset, classes)),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))

    history = []
    best_macro_f1 = -1.0
    stale_epochs = 0
    global_step = 0
    stopped = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        running_count = 0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits.float(), targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += float(loss.item()) * int(targets.numel())
            running_count += int(targets.numel())
            if args.log_interval and (step == 1 or step % args.log_interval == 0):
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={float(loss.item()):.6f} avg={running_loss/max(1,running_count):.6f} "
                    f"grad_norm={float(grad_norm):.4f}"
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                stopped = True
                break

        val_metrics, confusion = evaluate(model, val_loader, device, len(classes), args.amp, amp_dtype)
        train_loss = running_loss / max(1, running_count)
        record = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_precision": val_metrics["macro_precision"],
            "lr": [group["lr"] for group in optimizer.param_groups],
            "seconds": round(time.time() - start, 2),
        }
        history.append(record)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        checkpoint = {
            "model": model.state_dict(),
            "classes": classes,
            "config": config,
            "epoch": epoch,
            "metrics": record,
        }
        torch.save(checkpoint, output / "last.pt")
        torch.save(checkpoint, output / f"epoch_{epoch:03d}.pt")
        if val_metrics["macro_f1"] > best_macro_f1 + args.min_delta:
            best_macro_f1 = val_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(checkpoint, output / "best.pt")
            (output / "best_metrics.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            torch.save(confusion, output / "best_confusion_matrix.pt")
            (output / "best_confusion_matrix.json").write_text(
                json.dumps(confusion.tolist(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_csv(output / "best_per_class_report.csv", per_class_report(confusion, classes))
        else:
            stale_epochs += 1

        if stopped:
            break
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping: no macro-F1 improvement for {stale_epochs} epochs.")
            break

    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training finished. Outputs: {output}")


if __name__ == "__main__":
    main()
