#!/usr/bin/env python3
"""Train query, DINOv2 crop, or query+DINO fixed-output verifier variants."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from dfine_dinov2_teacher import VerifierCandidateDataset, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("query", "dino", "query_dino"), required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument(
        "--dinov2-weights",
        type=Path,
        default=Path("/root/autodl-tmp/weights/dinov2-base"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--tight-expand", type=float, default=1.0)
    parser.add_argument("--context-expand", type=float, default=1.8)
    parser.add_argument("--min-crop-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--head-only-epochs", type=int, default=1)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--residual-weight", type=float, default=1e-3)
    parser.add_argument("--positive-iou", type=float, default=0.5)
    parser.add_argument("--background-iou", type=float, default=0.1)
    parser.add_argument("--hard-negative-power", type=float, default=2.0)
    parser.add_argument("--weak-class-positive-boost", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help="Diagnostic-only cap for smoke tests and physical-batch benchmarking.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def binary_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    order = scores.argsort(descending=True)
    positives = targets[order].double()
    total_positive = int(positives.sum())
    if not total_positive:
        return 0.0
    precision = positives.cumsum(0) / torch.arange(
        1, len(positives) + 1, dtype=torch.double
    )
    return float((precision * positives).sum() / total_positive)


def binary_operating_point(
    scores: torch.Tensor, targets: torch.Tensor, target_recall: float
) -> dict:
    order = scores.argsort(descending=True)
    ranked = targets[order].long()
    required = math.ceil(float(targets.sum()) * target_recall - 1e-12)
    cumulative = ranked.cumsum(0)
    locations = torch.where(cumulative >= required)[0]
    if not len(locations):
        return {"reachable": False}
    count = int(locations[0]) + 1
    tp = int(ranked[:count].sum())
    fp = count - tp
    return {
        "reachable": True,
        "prediction_count": count,
        "tp": tp,
        "fp": fp,
        "precision": tp / count,
        "recall": tp / max(1, int(targets.sum())),
        "threshold": float(scores[order[count - 1]]),
    }


def save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_sampler(cache: dict, valid_indices: torch.Tensor, args) -> WeightedRandomSampler:
    iou = cache["max_iou"][valid_indices].float()
    positive = iou >= args.positive_iou
    negative = iou < args.background_iou
    base_probability = cache["selection_logits"][valid_indices].float().sigmoid()
    weights = torch.zeros(len(valid_indices), dtype=torch.double)
    positive_count, negative_count = int(positive.sum()), int(negative.sum())
    weights[positive] = 0.5 / max(1, positive_count)
    hard = base_probability[negative].clamp_min(1e-4).pow(args.hard_negative_power)
    hard = hard / hard.sum().clamp_min(1e-12)
    weights[negative] = 0.5 * hard.double()
    nearest = cache["nearest_gt_label"][valid_indices].long()
    weak_positive = positive & ((nearest == 2) | (nearest == 5))
    weights[weak_positive] *= args.weak_class_positive_boost
    weights /= weights.sum()
    return WeightedRandomSampler(
        weights,
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def loss_function(output: dict, target: torch.Tensor, args) -> tuple[torch.Tensor, dict]:
    logit = output["logit"].float()
    bce = F.binary_cross_entropy_with_logits(logit, target)
    positive_logits = logit[target.bool()]
    negative_logits = logit[~target.bool()]
    if len(positive_logits) and len(negative_logits):
        # Hardest negatives and weakest positives dominate the finite batch ranking.
        negative_logits = negative_logits.topk(min(16, len(negative_logits))).values
        positive_logits = positive_logits.topk(
            min(16, len(positive_logits)), largest=False
        ).values
        rank = F.softplus(
            args.rank_margin
            - positive_logits.unsqueeze(1)
            + negative_logits.unsqueeze(0)
        ).mean()
    else:
        rank = logit.sum() * 0
    residual = output["delta"].float().square().mean()
    total = bce + args.rank_weight * rank + args.residual_weight * residual
    return total, {"bce": bce, "rank": rank, "residual": residual}


@torch.inference_mode()
def evaluate(model, loader, device, dtype, args) -> dict:
    model.eval()
    score_rows, target_rows, iou_rows, index_rows = [], [], [], []
    for step, batch in enumerate(loader, 1):
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda",
        ):
            output = model(**batch)
        score_rows.append(output["logit"].float().cpu())
        target_rows.append((batch["max_iou"] >= args.positive_iou).long().cpu())
        iou_rows.append(batch["max_iou"].float().cpu())
        index_rows.append(batch["candidate_index"].long().cpu())
        if args.max_val_batches is not None and step >= args.max_val_batches:
            break
    scores = torch.cat(score_rows) if score_rows else torch.empty(0)
    targets = torch.cat(target_rows) if target_rows else torch.empty(0, dtype=torch.long)
    return {
        "binary_ap": binary_average_precision(scores, targets),
        "at_positive_recall_080": binary_operating_point(scores, targets, 0.8),
        "positive_count": int(targets.sum()),
        "background_count": int(len(targets) - targets.sum()),
        "scores": scores,
        "indices": torch.cat(index_rows) if index_rows else torch.empty(0, dtype=torch.long),
        "max_iou": torch.cat(iou_rows) if iou_rows else torch.empty(0),
    }


def main() -> None:
    args = parse_args()
    if args.batch < 1 or args.workers < 0 or args.samples_per_epoch < 1:
        raise ValueError("Invalid batch/workers/samples-per-epoch")
    if not 0 <= args.background_iou < args.positive_iou <= 1:
        raise ValueError("Require background-iou < positive-iou")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"{args.output} exists; pass --overwrite or --resume")
    if args.overwrite and args.output.exists() and not args.resume:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_cache = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    if train_cache["classes"] != val_cache["classes"]:
        raise ValueError("Train/val class mismatch")
    if train_cache["checkpoint"] != val_cache["checkpoint"]:
        raise ValueError("Train/val caches came from different D-FINE checkpoints")

    train_iou = train_cache["max_iou"].float()
    train_valid = torch.where(
        (train_iou >= args.positive_iou) | (train_iou < args.background_iou)
    )[0]
    val_iou = val_cache["max_iou"].float()
    val_valid = torch.where(
        (val_iou >= args.positive_iou) | (val_iou < args.background_iou)
    )[0]
    need_crops = args.mode != "query"
    dataset_kwargs = {
        "need_crops": need_crops,
        "image_size": args.image_size,
        "tight_expand": args.tight_expand,
        "context_expand": args.context_expand,
        "min_crop_size": args.min_crop_size,
    }
    train_dataset = VerifierCandidateDataset(
        args.train_cache, train_valid, train=True, **dataset_kwargs
    )
    val_dataset = VerifierCandidateDataset(
        args.val_cache, val_valid, train=False, **dataset_kwargs
    )
    sampler = make_sampler(train_cache, train_valid, args)
    common = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        common["prefetch_factor"] = 1
    train_loader = DataLoader(
        train_dataset, sampler=sampler, drop_last=True, **common
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **common
    )

    device = torch.device(args.device)
    model = build_model(
        args.mode,
        query_dim=int(train_cache["feature_dim"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        dinov2_weights=args.dinov2_weights,
    ).to(device)
    if hasattr(model, "set_trainable_blocks"):
        model.set_trainable_blocks(0)

    backbone_parameters, other_parameters = [], []
    for name, parameter in model.named_parameters():
        if ".backbone." in name:
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    optimizer = AdamW(
        [
            {"params": other_parameters, "lr": args.lr, "name": "heads"},
            {"params": backbone_parameters, "lr": args.backbone_lr, "name": "backbone"},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=min(args.lr, args.backbone_lr) * 0.1
    )
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and dtype == torch.float16
    )

    config = {
        **vars(args),
        "train_candidates": len(train_cache["selection_logits"]),
        "train_valid": len(train_valid),
        "val_candidates": len(val_cache["selection_logits"]),
        "val_valid": len(val_valid),
        "classes": train_cache["classes"],
        "dfine_checkpoint": train_cache["checkpoint"],
    }
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in config.items()}
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    start_epoch, best_ap, best_recall80_fp, stale, history = (
        1, -1.0, math.inf, 0, []
    )
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_ap = float(state["best_ap"])
        best_recall80_fp = float(state.get("best_recall80_fp", math.inf))
        stale = int(state.get("stale", 0))
        history_path = args.output / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(model, "set_trainable_blocks"):
            blocks = args.unfreeze_last_blocks if epoch > args.head_only_epochs else 0
            model.set_trainable_blocks(blocks)
        else:
            blocks = 0
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        totals = {"loss": 0.0, "bce": 0.0, "rank": 0.0, "residual": 0.0}
        count = 0
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            batch = move_batch(batch, device)
            target = (batch["max_iou"] >= args.positive_iou).float()
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda",
            ):
                output = model(**batch)
                loss, parts = loss_function(output, target, args)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            batch_count = len(target)
            totals["loss"] += float(loss.detach()) * batch_count
            for key, value in parts.items():
                totals[key] += float(value.detach()) * batch_count
            count += batch_count
            if step % 50 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={totals['loss']/count:.4f}",
                    flush=True,
                )
            if args.max_steps is not None and step >= args.max_steps:
                break
        scheduler.step()
        validation = evaluate(model, val_loader, device, dtype, args)
        seconds = time.time() - started
        peak = (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else 0.0
        )
        record = {
            "epoch": epoch,
            "unfrozen_backbone_blocks": blocks,
            **{f"train_{key}": value / max(1, count)
               for key, value in totals.items()},
            "val_binary_ap": validation["binary_ap"],
            "val_at_positive_recall_080": validation["at_positive_recall_080"],
            "seconds": seconds,
            "peak_cuda_gib": peak,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }
        history.append(record)
        (args.output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        improved_ap = validation["binary_ap"] > best_ap + 1e-8
        recall80 = validation["at_positive_recall_080"]
        recall80_fp = (
            int(recall80["fp"]) if recall80.get("reachable") else math.inf
        )
        improved_recall80 = recall80_fp < best_recall80_fp
        if improved_ap:
            best_ap = validation["binary_ap"]
        if improved_recall80:
            best_recall80_fp = recall80_fp
        stale = 0 if (improved_ap or improved_recall80) else stale + 1
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_ap": best_ap,
            "best_recall80_fp": best_recall80_fp,
            "stale": stale,
            "mode": args.mode,
            "config": config,
            "record": record,
        }
        save_atomic(checkpoint, args.output / "last.pt")
        if improved_ap:
            save_atomic(checkpoint, args.output / "best.pt")
            torch.save(
                {
                    "indices": validation["indices"],
                    "scores": validation["scores"],
                    "max_iou": validation["max_iou"],
                },
                args.output / "best_binary_val_scores.pt",
            )
        if improved_recall80:
            save_atomic(checkpoint, args.output / "best_recall80.pt")
            torch.save(
                {
                    "indices": validation["indices"],
                    "scores": validation["scores"],
                    "max_iou": validation["max_iou"],
                },
                args.output / "best_recall80_binary_val_scores.pt",
            )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.max_steps is not None:
            break
        if stale >= args.patience:
            print(f"EARLY_STOP epoch={epoch} stale={stale}", flush=True)
            break

    print(json.dumps({
        "mode": args.mode,
        "best_binary_ap": best_ap,
        "best_recall80_fp": best_recall80_fp,
        "output": str(args.output.resolve()),
        "completed_epoch": history[-1]["epoch"] if history else start_epoch - 1,
    }, indent=2))


if __name__ == "__main__":
    main()
