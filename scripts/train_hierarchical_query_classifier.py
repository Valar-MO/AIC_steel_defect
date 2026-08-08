#!/usr/bin/env python3
"""Train H0/H1/H2 linear query-classifier ablations on frozen D-FINE features."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import torch

from hierarchical_query_classifier import (
    HierarchicalQueryClassifier,
    balanced_sample_weights,
    class_count_dict,
    classification_loss,
    classification_metrics,
)


VARIANTS = {
    "H0": {"balanced": False, "margin": False},
    "H1": {"balanced": True, "margin": False},
    "H2": {"balanced": True, "margin": True},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1.25e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--margin-weight", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_cache(train: dict, val: dict) -> None:
    required = {
        "classes", "feature_dim", "matched_features", "matched_class_targets",
        "detector_class_weight", "detector_class_bias",
    }
    for name, cache in (("train", train), ("val", val)):
        missing = required - cache.keys()
        if missing:
            raise ValueError(f"{name} cache missing keys: {sorted(missing)}")
    if train["classes"] != val["classes"]:
        raise ValueError("Train and val cache classes differ")
    if train["feature_dim"] != val["feature_dim"]:
        raise ValueError("Train and val feature dimensions differ")
    if not torch.equal(train["detector_class_weight"], val["detector_class_weight"]):
        raise ValueError("Train and val caches came from different detector class weights")
    if not torch.equal(train["detector_class_bias"], val["detector_class_bias"]):
        raise ValueError("Train and val caches came from different detector class biases")


def epoch_indices(
    targets: torch.Tensor,
    *,
    balanced: bool,
    num_classes: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if not balanced:
        return torch.randperm(len(targets), generator=generator)
    probabilities = balanced_sample_weights(targets, num_classes)
    return torch.multinomial(
        probabilities, len(targets), replacement=True, generator=generator
    )


@torch.inference_mode()
def evaluate(model, features, targets, classes):
    model.eval()
    return classification_metrics(model(features), targets, classes)


def main():
    args = parse_args()
    if args.batch <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("batch, epochs and patience must be positive")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    validate_cache(train, val)
    classes = train["classes"]
    num_classes = len(classes)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    x_train = train["matched_features"].float().to(device)
    y_train = train["matched_class_targets"].long().to(device)
    x_val = val["matched_features"].float().to(device)
    y_val = val["matched_class_targets"].long().to(device)
    initial_weight = train["detector_class_weight"].float().to(device)
    initial_bias = train["detector_class_bias"].float().to(device)
    if not len(x_train) or not len(x_val):
        raise ValueError("Caches must contain matched positive queries")

    baseline_model = HierarchicalQueryClassifier(train["feature_dim"], num_classes).to(device)
    baseline_model.initialize_from_detector(initial_weight, initial_bias)
    report = {
        "config": {**vars(args),
                   "train_cache": str(args.train_cache), "val_cache": str(args.val_cache),
                   "output": str(args.output)},
        "classes": classes,
        "train_class_counts": class_count_dict(y_train, classes),
        "val_class_counts": class_count_dict(y_val, classes),
        "detector_baseline": evaluate(baseline_model, x_val, y_val, classes),
        "runs": [],
    }

    for variant in args.variants:
        options = VARIANTS[variant]
        for seed in args.seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = HierarchicalQueryClassifier(train["feature_dim"], num_classes).to(device)
            model.initialize_from_detector(initial_weight, initial_bias)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            best_score = float("-inf")
            stale = 0
            history = []
            checkpoint_path = args.output / f"{variant}_seed{seed}.pt"
            for epoch in range(1, args.epochs + 1):
                model.train()
                order = epoch_indices(
                    y_train.cpu(),
                    balanced=options["balanced"],
                    num_classes=num_classes,
                    seed=seed * 1000 + epoch,
                )
                total_loss = total_ce = total_margin = 0.0
                steps = 0
                for indices in order.split(args.batch):
                    indices = indices.to(device)
                    logits = model(x_train[indices])
                    loss, components = classification_loss(
                        logits,
                        y_train[indices],
                        label_smoothing=args.label_smoothing,
                        margin=args.margin,
                        margin_weight=args.margin_weight if options["margin"] else 0.0,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += float(loss.detach())
                    total_ce += float(components["ce"].detach())
                    total_margin += float(components["margin"].detach())
                    steps += 1
                metrics = evaluate(model, x_val, y_val, classes)
                score = metrics["macro_f1"]
                row = {
                    "epoch": epoch,
                    "loss": total_loss / steps,
                    "ce": total_ce / steps,
                    "margin": total_margin / steps,
                    "accuracy": metrics["accuracy"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                }
                history.append(row)
                if score > best_score + args.min_delta:
                    best_score = score
                    stale = 0
                    torch.save({
                        "model": model.state_dict(),
                        "feature_dim": train["feature_dim"],
                        "classes": classes,
                        "variant": variant,
                        "seed": seed,
                        "epoch": epoch,
                        "balanced_sampling": options["balanced"],
                        "margin_enabled": options["margin"],
                        "margin": args.margin,
                        "margin_weight": args.margin_weight,
                        "label_smoothing": args.label_smoothing,
                        "classification_metrics": metrics,
                    }, checkpoint_path)
                else:
                    stale += 1
                print(json.dumps({"variant": variant, "seed": seed, **row}), flush=True)
                if stale >= args.patience:
                    break
            best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(best_checkpoint["model"], strict=True)
            best_metrics = evaluate(model, x_val, y_val, classes)
            report["runs"].append({
                "variant": variant,
                "seed": seed,
                "checkpoint": str(checkpoint_path.resolve()),
                "best_epoch": best_checkpoint["epoch"],
                "classification_metrics": best_metrics,
                "history": history,
            })

    (args.output / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = {
        "detector_baseline": {
            key: report["detector_baseline"][key]
            for key in ("accuracy", "macro_recall", "macro_f1")
        },
        "runs": [{
            "variant": row["variant"],
            "seed": row["seed"],
            "best_epoch": row["best_epoch"],
            "accuracy": row["classification_metrics"]["accuracy"],
            "macro_recall": row["classification_metrics"]["macro_recall"],
            "macro_f1": row["classification_metrics"]["macro_f1"],
        } for row in report["runs"]],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
