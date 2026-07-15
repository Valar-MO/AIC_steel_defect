#!/usr/bin/env python3
"""Train a multi-scale DINOv2 dual-head proposal verifier.

This is a second-stage model for detector/classifier decoupling.  It consumes
proposal crops described by ``build_proposal_cls_dataset.py`` metadata:

    tight crop   = proposal box expanded by --tight-expand
    context crop = proposal box expanded by --context-expand

The shared DINOv2 backbone encodes both crops.  The concatenated features feed:

    objectness head: defect vs background
    class head: 9 defect classes, trained only on positive proposals
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

from train_crop_classifier import IMAGENET_MEAN, IMAGENET_STD, load_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/root/autodl-tmp/datasets/AIC_steel_proposal_bg_cls_v1"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"),
                        help="9 detector/defect classes, without background.")
    parser.add_argument("--background-class", default="background")
    parser.add_argument("--dinov2-weights", type=Path, default=Path("/root/autodl-tmp/weights/dinov2-small"))
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Optional existing multi-scale dual-head checkpoint for hard-example fine-tuning.")
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/runs/crop_cls/dinov2_small_ms_dual_v1"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--tight-expand", type=float, default=1.0)
    parser.add_argument("--context-expand", type=float, default=1.8)
    parser.add_argument("--min-crop-size", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=2)
    parser.add_argument("--lambda-obj", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float,
                        help="Optional BCE positive weight for objectness. Default is neg/pos from train data.")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--hard-positive-weight", type=float, default=1.0,
                        help="Sampler multiplier for low-score true proposals, applied on train split only.")
    parser.add_argument("--hard-negative-weight", type=float, default=1.0,
                        help="Sampler multiplier for low-score false proposals, applied on train split only.")
    parser.add_argument("--hard-score-min", type=float, default=0.05)
    parser.add_argument("--hard-score-max", type=float, default=0.20)
    parser.add_argument("--hard-positive-iou-min", type=float, default=0.50)
    parser.add_argument("--hard-negative-iou-max", type=float, default=0.30)
    parser.add_argument("--hard-positive-classes", nargs="*",
                        default=["zonglie", "yanghuatiepi", "jieba", "huashang", "yiwuyaru"],
                        help="Classes emphasized as hard positives. Pass empty list to allow all classes.")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expanded_crop_box(box, width: int, height: int, expand: float, min_size: int):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid proposal bbox: {box}")
    side = max(max(x2 - x1, y2 - y1) * expand, float(min_size))
    side = min(side, float(max(width, height)))
    crop_w = min(side, float(width))
    crop_h = min(side, float(height))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = min(max(cx - crop_w / 2.0, 0.0), float(width) - crop_w)
    top = min(max(cy - crop_h / 2.0, 0.0), float(height) - crop_h)
    return (
        int(math.floor(left)),
        int(math.floor(top)),
        int(math.ceil(left + crop_w)),
        int(math.ceil(top + crop_h)),
    )


class MultiScaleProposalDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        defect_classes: list[str],
        *,
        background_class: str,
        image_size: int,
        tight_expand: float,
        context_expand: float,
        min_crop_size: int,
        train: bool,
        max_samples: int | None = None,
        seed: int = 42,
        hard_positive_weight: float = 1.0,
        hard_negative_weight: float = 1.0,
        hard_score_min: float = 0.05,
        hard_score_max: float = 0.20,
        hard_positive_iou_min: float = 0.50,
        hard_negative_iou_max: float = 0.30,
        hard_positive_classes: set[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.defect_classes = defect_classes
        self.class_to_id = {name: idx for idx, name in enumerate(defect_classes)}
        self.background_class = background_class
        self.image_size = int(image_size)
        self.tight_expand = float(tight_expand)
        self.context_expand = float(context_expand)
        self.min_crop_size = int(min_crop_size)
        self.train = bool(train)
        self.hard_positive_weight = float(hard_positive_weight)
        self.hard_negative_weight = float(hard_negative_weight)
        self.hard_score_min = float(hard_score_min)
        self.hard_score_max = float(hard_score_max)
        self.hard_positive_iou_min = float(hard_positive_iou_min)
        self.hard_negative_iou_max = float(hard_negative_iou_max)
        self.hard_positive_classes = set(hard_positive_classes or [])

        metadata = self.root / "metadata.csv"
        if not metadata.is_file():
            raise FileNotFoundError(f"Missing metadata.csv: {metadata}")
        rows = []
        with metadata.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != split:
                    continue
                class_name = row["class_name"]
                is_background = class_name == background_class or row.get("sample_type") == "background"
                if not is_background and class_name not in self.class_to_id:
                    raise ValueError(f"Unknown class in metadata: {class_name}")
                rows.append(row)
        if max_samples is not None and len(rows) > max_samples:
            rng = random.Random(seed)
            rng.shuffle(rows)
            rows = rows[:max_samples]
            rows.sort(key=lambda r: (r["image_id"], r["pred_index"], r["class_name"]))
        if not rows:
            raise ValueError(f"No metadata rows for split={split}")
        self.rows = rows
        self.obj_targets = [0 if (r["class_name"] == background_class or r.get("sample_type") == "background") else 1 for r in rows]
        self.class_targets = [
            -100 if obj == 0 else self.class_to_id[rows[i]["class_name"]]
            for i, obj in enumerate(self.obj_targets)
        ]
        self.sample_weights = [self._sample_weight(row) for row in rows]
        self.hard_tags = [self._hard_tag(row) for row in rows]

    def _row_score_iou(self, row: dict) -> tuple[float, float]:
        try:
            score = float(row.get("det_score") or 0.0)
        except ValueError:
            score = 0.0
        try:
            iou = float(row.get("iou") or 0.0)
        except ValueError:
            iou = 0.0
        return score, iou

    def _hard_tag(self, row: dict) -> str:
        if not self.train:
            return "normal"
        score, iou = self._row_score_iou(row)
        in_score_band = self.hard_score_min <= score <= self.hard_score_max
        if not in_score_band:
            return "normal"
        class_name = row["class_name"]
        is_background = class_name == self.background_class or row.get("sample_type") == "background"
        if is_background and iou <= self.hard_negative_iou_max:
            return "hard_negative"
        allow_class = not self.hard_positive_classes or class_name in self.hard_positive_classes
        if (not is_background) and allow_class and iou >= self.hard_positive_iou_min:
            return "hard_positive"
        return "normal"

    def _sample_weight(self, row: dict) -> float:
        tag = self._hard_tag(row)
        if tag == "hard_positive":
            return self.hard_positive_weight
        if tag == "hard_negative":
            return self.hard_negative_weight
        return 1.0

    def __len__(self) -> int:
        return len(self.rows)

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
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        data = data.view(image.height, image.width, 3).permute(2, 0, 1).float().div_(255.0)
        return (data - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, index: int):
        row = self.rows[index]
        source = Path(row["source_image"])
        proposal = [float(row[k]) for k in ("proposal_x1", "proposal_y1", "proposal_x2", "proposal_y2")]
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image = self._augment(image)
            tight = image.crop(expanded_crop_box(proposal, image.width, image.height, self.tight_expand, self.min_crop_size))
            context = image.crop(expanded_crop_box(proposal, image.width, image.height, self.context_expand, self.min_crop_size))
        return (
            self._to_tensor(tight),
            self._to_tensor(context),
            torch.tensor(self.obj_targets[index], dtype=torch.float32),
            torch.tensor(self.class_targets[index], dtype=torch.long),
        )


class MultiScaleDualHeadVerifier(nn.Module):
    def __init__(self, weights: Path, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import Dinov2Model

        self.backbone = Dinov2Model.from_pretrained(str(weights), local_files_only=True)
        hidden_size = int(self.backbone.config.hidden_size)
        fused_size = hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.objectness = nn.Linear(fused_size, 1)
        self.classifier = nn.Linear(fused_size, num_classes)

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

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        return outputs.last_hidden_state[:, 0]

    def forward(self, tight: torch.Tensor, context: torch.Tensor):
        # One backbone call over the concatenated batch keeps shared weights and
        # avoids duplicating DINOv2 modules.
        both = torch.cat([tight, context], dim=0)
        features = self.encode(both)
        tight_feat, context_feat = features.chunk(2, dim=0)
        fused = self.dropout(torch.cat([tight_feat, context_feat], dim=1))
        return {
            "objectness_logits": self.objectness(fused).squeeze(1),
            "class_logits": self.classifier(fused),
        }


def make_balanced_sampler(dataset: MultiScaleProposalDataset) -> WeightedRandomSampler:
    keys = [(int(o), int(c)) for o, c in zip(dataset.obj_targets, dataset.class_targets)]
    counts = Counter(keys)
    weights = [(1.0 / counts[key]) * float(dataset.sample_weights[i]) for i, key in enumerate(keys)]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)


def class_weights(dataset: MultiScaleProposalDataset, num_classes: int, power: float) -> torch.Tensor:
    counts = Counter(c for c in dataset.class_targets if c >= 0)
    max_count = max(counts.values()) if counts else 1
    weights = [(max_count / max(1, counts[i])) ** power for i in range(num_classes)]
    values = torch.tensor(weights, dtype=torch.float32)
    return values / values.mean()


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def macro_f1_from_confusion(confusion: torch.Tensor) -> dict[str, float]:
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    valid = support > 0
    return {
        "macro_precision": float(precision[valid].mean()) if valid.any() else 0.0,
        "macro_recall": float(recall[valid].mean()) if valid.any() else 0.0,
        "macro_f1": float(f1[valid].mean()) if valid.any() else 0.0,
    }


@torch.inference_mode()
def evaluate(model, loader, device, num_classes, obj_criterion, cls_criterion, lambda_obj, lambda_cls, amp, amp_dtype):
    model.eval()
    total_loss = total_obj_loss = total_cls_loss = 0.0
    total = 0
    obj_tp = obj_fp = obj_fn = obj_tn = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for tight, context, obj_targets, class_targets in loader:
        tight = tight.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        obj_targets = obj_targets.to(device, non_blocking=True)
        class_targets = class_targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp and device.type == "cuda"):
            outputs = model(tight, context)
            obj_loss = obj_criterion(outputs["objectness_logits"].float(), obj_targets)
            positive = class_targets >= 0
            cls_loss = cls_criterion(outputs["class_logits"][positive].float(), class_targets[positive]) if positive.any() else outputs["class_logits"].sum() * 0
            loss = lambda_obj * obj_loss + lambda_cls * cls_loss
        total_loss += float(loss.item()) * int(obj_targets.numel())
        total_obj_loss += float(obj_loss.item()) * int(obj_targets.numel())
        total_cls_loss += float(cls_loss.item()) * int(max(1, positive.sum().item()))
        total += int(obj_targets.numel())
        obj_pred = torch.sigmoid(outputs["objectness_logits"]) >= 0.5
        obj_true = obj_targets >= 0.5
        obj_tp += int((obj_pred & obj_true).sum().item())
        obj_fp += int((obj_pred & ~obj_true).sum().item())
        obj_fn += int((~obj_pred & obj_true).sum().item())
        obj_tn += int((~obj_pred & ~obj_true).sum().item())
        if positive.any():
            pred_cls = outputs["class_logits"][positive].argmax(dim=1).cpu()
            true_cls = class_targets[positive].cpu()
            for t, p in zip(true_cls, pred_cls):
                confusion[int(t), int(p)] += 1
    obj = binary_metrics(obj_tp, obj_fp, obj_fn)
    cls = macro_f1_from_confusion(confusion)
    obj_acc = (obj_tp + obj_tn) / max(1, obj_tp + obj_tn + obj_fp + obj_fn)
    return {
        "loss": total_loss / max(1, total),
        "obj_loss": total_obj_loss / max(1, total),
        "cls_loss": total_cls_loss / max(1, int(confusion.sum().item())),
        "objectness_accuracy": obj_acc,
        "objectness_precision": obj["precision"],
        "objectness_recall": obj["recall"],
        "objectness_f1": obj["f1"],
        **{f"class_{k}": v for k, v in cls.items()},
        "selection_score": 0.5 * obj["f1"] + 0.5 * cls["macro_f1"],
    }, confusion


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def per_class_report(confusion: torch.Tensor, classes: list[str]) -> list[dict]:
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return [
        {
            "class_id": i,
            "class_name": name,
            "support": int(support[i].item()),
            "predicted": int(predicted[i].item()),
            "precision": float(precision[i].item()),
            "recall": float(recall[i].item()),
            "f1": float(f1[i].item()),
        }
        for i, name in enumerate(classes)
    ]


def prepare_output(path: Path, overwrite: bool, exist_ok: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif not exist_ok:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite or --exist-ok.")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    classes = load_classes(args.classes)
    output = args.output.resolve()
    prepare_output(output, args.overwrite, args.exist_ok)

    train_dataset = MultiScaleProposalDataset(
        args.data, "train", classes, background_class=args.background_class,
        image_size=args.image_size, tight_expand=args.tight_expand,
        context_expand=args.context_expand, min_crop_size=args.min_crop_size,
        train=True, max_samples=args.max_train_samples, seed=args.seed,
        hard_positive_weight=args.hard_positive_weight,
        hard_negative_weight=args.hard_negative_weight,
        hard_score_min=args.hard_score_min,
        hard_score_max=args.hard_score_max,
        hard_positive_iou_min=args.hard_positive_iou_min,
        hard_negative_iou_max=args.hard_negative_iou_max,
        hard_positive_classes=set(args.hard_positive_classes or []),
    )
    val_dataset = MultiScaleProposalDataset(
        args.data, "val", classes, background_class=args.background_class,
        image_size=args.image_size, tight_expand=args.tight_expand,
        context_expand=args.context_expand, min_crop_size=args.min_crop_size,
        train=False, max_samples=args.max_val_samples, seed=args.seed,
    )
    sampler = None if args.no_balanced_sampler else make_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=sampler is None,
                              sampler=sampler, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = MultiScaleDualHeadVerifier(args.dinov2_weights, len(classes), dropout=args.dropout).to(device)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location="cpu")
        ckpt_classes = payload.get("classes")
        if ckpt_classes is not None and list(ckpt_classes) != classes:
            raise ValueError("Checkpoint classes differ from --classes")
        missing, unexpected = model.load_state_dict(payload.get("model", payload), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Failed to load checkpoint cleanly. missing={missing}, unexpected={unexpected}")
        print(f"Loaded init checkpoint: {args.init_checkpoint}")
    if args.freeze_backbone:
        model.set_backbone_trainable(False)
    if args.unfreeze_last_blocks:
        model.unfreeze_last_blocks(args.unfreeze_last_blocks)

    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = AdamW(groups, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps, int(args.warmup_epochs * max(1, len(train_loader))))

    pos = sum(train_dataset.obj_targets)
    neg = len(train_dataset.obj_targets) - pos
    pos_weight = args.pos_weight if args.pos_weight is not None else neg / max(1, pos)
    obj_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    cls_criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_dataset, len(classes), args.class_weight_power).to(device),
        label_smoothing=args.label_smoothing,
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda" and amp_dtype == torch.float16)

    train_counts = Counter(train_dataset.class_targets)
    val_counts = Counter(val_dataset.class_targets)
    hard_counts = Counter(train_dataset.hard_tags)
    config = {
        "data": str(args.data.resolve()),
        "classes": classes,
        "background_class": args.background_class,
        "dinov2_weights": str(args.dinov2_weights.resolve()),
        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
        "image_size": args.image_size,
        "tight_expand": args.tight_expand,
        "context_expand": args.context_expand,
        "batch": args.batch,
        "epochs": args.epochs,
        "freeze_backbone": args.freeze_backbone,
        "unfreeze_last_blocks": args.unfreeze_last_blocks,
        "lambda_obj": args.lambda_obj,
        "lambda_cls": args.lambda_cls,
        "pos_weight": pos_weight,
        "balanced_sampler": sampler is not None,
        "hard_positive_weight": args.hard_positive_weight,
        "hard_negative_weight": args.hard_negative_weight,
        "hard_score_min": args.hard_score_min,
        "hard_score_max": args.hard_score_max,
        "hard_positive_iou_min": args.hard_positive_iou_min,
        "hard_negative_iou_max": args.hard_negative_iou_max,
        "hard_positive_classes": args.hard_positive_classes,
        "hard_tag_counts": dict(hard_counts),
        "train_positive": int(pos),
        "train_background": int(neg),
        "train_class_counts": {classes[i]: train_counts[i] for i in range(len(classes))},
        "val_class_counts": {classes[i]: val_counts[i] for i in range(len(classes))},
        "val_background": val_counts[-100],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))

    history = []
    best_score = -1.0
    stale = 0
    global_step = 0
    stopped = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running = 0.0
        count = 0
        for step, (tight, context, obj_targets, class_targets) in enumerate(train_loader, start=1):
            tight = tight.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            obj_targets = obj_targets.to(device, non_blocking=True)
            class_targets = class_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                outputs = model(tight, context)
                obj_loss = obj_criterion(outputs["objectness_logits"].float(), obj_targets)
                positive = class_targets >= 0
                cls_loss = cls_criterion(outputs["class_logits"][positive].float(), class_targets[positive]) if positive.any() else outputs["class_logits"].sum() * 0
                loss = args.lambda_obj * obj_loss + args.lambda_cls * cls_loss
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
            running += float(loss.item()) * int(obj_targets.numel())
            count += int(obj_targets.numel())
            if args.log_interval and (step == 1 or step % args.log_interval == 0):
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={float(loss.detach()):.6f} avg={running/max(1,count):.6f} grad_norm={float(grad_norm):.4f}")
            if args.max_steps is not None and global_step >= args.max_steps:
                stopped = True
                break

        val_metrics, confusion = evaluate(
            model, val_loader, device, len(classes), obj_criterion, cls_criterion,
            args.lambda_obj, args.lambda_cls, args.amp, amp_dtype,
        )
        record = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": running / max(1, count),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": [g["lr"] for g in optimizer.param_groups],
            "seconds": round(time.time() - start, 2),
        }
        history.append(record)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        checkpoint = {
            "model": model.state_dict(),
            "classes": classes,
            "background_class": args.background_class,
            "config": config,
            "epoch": epoch,
            "metrics": record,
            "model_type": "multiscale_dual_head_verifier",
        }
        torch.save(checkpoint, output / "last.pt")
        torch.save(checkpoint, output / f"epoch_{epoch:03d}.pt")
        score = val_metrics["selection_score"]
        if score > best_score + args.min_delta:
            best_score = score
            stale = 0
            torch.save(checkpoint, output / "best.pt")
            (output / "best_metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            torch.save(confusion, output / "best_class_confusion_matrix.pt")
            (output / "best_class_confusion_matrix.json").write_text(json.dumps(confusion.tolist(), ensure_ascii=False, indent=2), encoding="utf-8")
            write_csv(output / "best_per_class_report.csv", per_class_report(confusion, classes))
        else:
            stale += 1
        if stopped:
            break
        if args.patience > 0 and stale >= args.patience:
            print(f"Early stopping: no selection score improvement for {stale} epochs.")
            break

    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training finished. Outputs: {output}")


if __name__ == "__main__":
    main()
