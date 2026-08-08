#!/usr/bin/env python3
"""Train an R1 Selection residual ranker on cached D-FINE query candidates."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.ops import box_iou

from hierarchical_query_classifier import SelectionResidualRanker, selection_ranking_loss


KIND_TO_TIER = {
    "background": 0,
    "wrong_class": 1,
    "localized": 2,
    "duplicate": 3,
    "winner": 4,
}
KIND_TO_TARGET = {
    "background": 0.0,
    "wrong_class": 0.15,
    "localized": 0.35,
    "duplicate": 0.65,
    "winner": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-ann", type=Path, required=True,
                        help="COCO annotation JSON for the cached train split")
    parser.add_argument("--val-ann", type=Path, required=True,
                        help="COCO annotation JSON for the cached val split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--pairwise-weight", type=float, default=0.75)
    parser.add_argument("--pairwise-margin", type=float, default=0.2)
    parser.add_argument("--preserve-weight", type=float, default=0.0)
    parser.add_argument("--preserve-margin", type=float, default=0.05)
    parser.add_argument("--hard-score-threshold", type=float, default=0.30)
    parser.add_argument("--hard-bg-weight", type=float, default=4.0)
    parser.add_argument("--duplicate-weight", type=float, default=1.5)
    parser.add_argument("--winner-weight", type=float, default=3.0)
    parser.add_argument("--nms-iou", type=float, default=0.80)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--localized-iou", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def jsonable_args(args) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def logit(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float().clamp(1e-5, 1 - 1e-5)
    return torch.logit(scores)


def normalized_prediction_view(cache: dict) -> dict:
    """Return a common candidate view for cache_hierarchical or view caches."""
    if "prediction_features" in cache:
        return {
            "features": cache["prediction_features"],
            "scores": cache["prediction_scores"],
            "labels": cache["prediction_labels"],
            "boxes": cache["prediction_boxes"],
            "image_names": list(cache["prediction_image_names"]),
        }
    if "query_features" in cache:
        image_rows = cache.get("images")
        image_indices = cache.get("candidate_image_index")
        if image_rows is None or image_indices is None:
            raise ValueError("View candidate cache missing images/candidate_image_index")
        class_logits = cache["class_logits"].float()
        return {
            "features": cache["query_features"],
            "scores": cache["selection_logits"].float().sigmoid(),
            "labels": class_logits.argmax(dim=1).long(),
            "boxes": cache["boxes_xyxy"],
            "image_names": [
                Path(str(image_rows[int(index)]["image_id"])).name
                for index in image_indices.tolist()
            ],
        }
    raise ValueError("Unsupported candidate cache schema")


def load_coco_gt(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_names = {int(row["id"]): Path(row["file_name"]).name for row in payload["images"]}
    grouped = defaultdict(lambda: {"boxes": [], "labels": []})
    for ann in payload["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in image_names:
            continue
        x, y, w, h = [float(value) for value in ann["bbox"]]
        if w <= 0 or h <= 0:
            continue
        grouped[image_names[image_id]]["boxes"].append([x, y, x + w, y + h])
        grouped[image_names[image_id]]["labels"].append(int(ann["category_id"]))
    return {
        name: {
            "boxes": torch.tensor(rows["boxes"], dtype=torch.float32),
            "labels": torch.tensor(rows["labels"], dtype=torch.long),
        }
        for name, rows in grouped.items()
    }


def nms_indices(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    order = scores.argsort(descending=True)
    keep = []
    while len(order):
        current = order[0]
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        ix1 = torch.maximum(x1[current], x1[rest])
        iy1 = torch.maximum(y1[current], y1[rest])
        ix2 = torch.minimum(x2[current], x2[rest])
        iy2 = torch.minimum(y2[current], y2[rest])
        inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
        union = areas[current] + areas[rest] - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        order = rest[iou <= threshold]
    return torch.stack(keep).long()


def validate_cache(cache: dict, name: str) -> None:
    required = {"classes", "feature_dim"}
    missing = required - cache.keys()
    if missing:
        raise ValueError(f"{name} cache missing keys: {sorted(missing)}")
    view = normalized_prediction_view(cache)
    if len(view["features"]) != len(view["scores"]):
        raise ValueError(f"{name} prediction feature/score count mismatch")


def label_candidates(cache: dict, gt: dict[str, dict[str, torch.Tensor]], args) -> dict:
    view = normalized_prediction_view(cache)
    boxes = view["boxes"].float()
    labels = view["labels"].long()
    scores = view["scores"].float()
    names = view["image_names"]
    kinds = ["background"] * len(scores)
    max_ious = torch.zeros(len(scores), dtype=torch.float32)
    hard_nms_survivor = torch.zeros(len(scores), dtype=torch.bool)
    by_image = defaultdict(list)
    for index, name in enumerate(names):
        by_image[name].append(index)

    for name, indices in by_image.items():
        indices_t = torch.tensor(indices, dtype=torch.long)
        image_gt = gt.get(name, {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)})
        gt_boxes = image_gt["boxes"]
        gt_labels = image_gt["labels"]
        if len(gt_boxes):
            ious = box_iou(boxes[indices_t], gt_boxes)
            max_iou, best_gt = ious.max(dim=1)
            max_ious[indices_t] = max_iou
            winner_local = set()
            for gt_index, gt_label in enumerate(gt_labels.tolist()):
                same = torch.where((labels[indices_t] == gt_label) & (ious[:, gt_index] >= args.match_iou))[0]
                if len(same):
                    best_local = same[ious[same, gt_index].argmax()]
                    winner_local.add(int(best_local))
            for local, global_index in enumerate(indices):
                same_class_iou = 0.0
                if len(gt_boxes):
                    same_gt = torch.where(gt_labels == labels[global_index])[0]
                    if len(same_gt):
                        same_class_iou = float(ious[local, same_gt].max())
                if local in winner_local:
                    kinds[global_index] = "winner"
                elif same_class_iou >= args.match_iou:
                    kinds[global_index] = "duplicate"
                elif float(max_iou[local]) >= args.match_iou:
                    kinds[global_index] = "wrong_class"
                elif float(max_iou[local]) >= args.localized_iou:
                    kinds[global_index] = "localized"

        for class_id in labels[indices_t].unique().tolist():
            class_mask = indices_t[labels[indices_t] == int(class_id)]
            scored = class_mask[scores[class_mask] >= args.hard_score_threshold]
            keep = nms_indices(boxes[scored], scores[scored], args.nms_iou)
            if len(keep):
                hard_nms_survivor[scored[keep]] = True

    tiers = torch.tensor([KIND_TO_TIER[kind] for kind in kinds], dtype=torch.long)
    targets = torch.tensor([KIND_TO_TARGET[kind] for kind in kinds], dtype=torch.float32)
    weights = torch.ones(len(scores), dtype=torch.float32)
    weights[tiers == KIND_TO_TIER["winner"]] = float(args.winner_weight)
    weights[tiers == KIND_TO_TIER["duplicate"]] = float(args.duplicate_weight)
    hard_bg = (tiers == KIND_TO_TIER["background"]) & hard_nms_survivor
    weights[hard_bg] = float(args.hard_bg_weight)
    return {
        "targets": targets,
        "tiers": tiers,
        "weights": weights,
        "kinds": kinds,
        "max_ious": max_ious,
        "hard_nms_survivor": hard_nms_survivor,
        "kind_counts": dict(Counter(kinds)),
        "hard_bg_count": int(hard_bg.sum()),
    }


@torch.inference_mode()
def evaluate(model, cache, labels, device) -> dict:
    model.eval()
    view = normalized_prediction_view(cache)
    features = view["features"].float().to(device)
    base_logits = logit(view["scores"]).to(device)
    final_logits = model(features, base_logits).cpu()
    base_scores = view["scores"].float()
    final_scores = final_logits.sigmoid()
    tiers = labels["tiers"]
    out = {
        "base_mean_score_by_kind": {},
        "final_mean_score_by_kind": {},
        "kind_counts": labels["kind_counts"],
        "hard_bg_count": labels["hard_bg_count"],
    }
    for kind, tier in KIND_TO_TIER.items():
        mask = tiers == tier
        if mask.any():
            out["base_mean_score_by_kind"][kind] = float(base_scores[mask].mean())
            out["final_mean_score_by_kind"][kind] = float(final_scores[mask].mean())
    winner = tiers == KIND_TO_TIER["winner"]
    hard_bg = (tiers == KIND_TO_TIER["background"]) & labels["hard_nms_survivor"]
    if winner.any() and hard_bg.any():
        out["winner_vs_hard_bg_margin"] = float(final_scores[winner].mean() - final_scores[hard_bg].mean())
        out["base_winner_vs_hard_bg_margin"] = float(base_scores[winner].mean() - base_scores[hard_bg].mean())
    return out


def main():
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_cache = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    validate_cache(train_cache, "train")
    validate_cache(val_cache, "val")
    if train_cache["classes"] != val_cache["classes"]:
        raise ValueError("Train and val classes differ")
    if train_cache["feature_dim"] != val_cache["feature_dim"]:
        raise ValueError("Train and val feature dimensions differ")
    train_labels = label_candidates(train_cache, load_coco_gt(args.train_ann), args)
    val_labels = label_candidates(val_cache, load_coco_gt(args.val_ann), args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SelectionResidualRanker(
        train_cache["feature_dim"],
        hidden_dim=args.hidden_dim,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_view = normalized_prediction_view(train_cache)
    x_train = train_view["features"].float()
    base_train = logit(train_view["scores"])
    y_train = train_labels["targets"]
    tiers_train = train_labels["tiers"]
    weights_train = train_labels["weights"]
    best_score = -float("inf")
    stale = 0
    history = []
    checkpoint_path = args.output / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        totals = Counter()
        steps = 0
        for indices in order.split(args.batch):
            final_logits = model(x_train[indices].to(device), base_train[indices].to(device))
            loss, components = selection_ranking_loss(
                final_logits,
                base_train[indices].to(device),
                y_train[indices].to(device),
                tiers_train[indices].to(device),
                weights_train[indices].to(device),
                margin=args.pairwise_margin,
                pairwise_weight=args.pairwise_weight,
                preserve_weight=args.preserve_weight,
                preserve_margin=args.preserve_margin,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["bce"] += float(components["bce"].detach())
            totals["pairwise"] += float(components["pairwise"].detach())
            totals["preserve"] += float(components["preserve"].detach())
            steps += 1
        val_metrics = evaluate(model, val_cache, val_labels, device)
        score = val_metrics.get("winner_vs_hard_bg_margin", 0.0)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / steps,
            "bce": totals["bce"] / steps,
            "pairwise": totals["pairwise"] / steps,
            "preserve": totals["preserve"] / steps,
            "val_winner_vs_hard_bg_margin": score,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "classes": train_cache["classes"],
                "feature_dim": train_cache["feature_dim"],
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "residual_scale": args.residual_scale,
                "epoch": epoch,
                "config": jsonable_args(args),
                "train_labels": {k: v for k, v in train_labels.items() if k.endswith("count") or k == "kind_counts"},
                "val_metrics": val_metrics,
            }, checkpoint_path)
        else:
            stale += 1
        if stale >= args.patience:
            break

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    report = {
        "config": jsonable_args(args),
        "classes": train_cache["classes"],
        "train_label_summary": {
            "kind_counts": train_labels["kind_counts"],
            "hard_bg_count": train_labels["hard_bg_count"],
        },
        "val_label_summary": {
            "kind_counts": val_labels["kind_counts"],
            "hard_bg_count": val_labels["hard_bg_count"],
        },
        "best_epoch": best["epoch"],
        "best_checkpoint": str(checkpoint_path.resolve()),
        "history": history,
        "val_metrics": evaluate(model, val_cache, val_labels, device),
    }
    (args.output / "selection_ranker_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": report["best_epoch"],
        "val_metrics": report["val_metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
