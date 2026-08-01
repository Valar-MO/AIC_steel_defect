#!/usr/bin/env python3
"""Train cheap nine-way class-conditional verifier ablations on cached V2.1 features."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalVerifierHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, num_classes))

    def forward(self, features):
        return self.net(features)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variants", nargs="+", default=("conditional", "conditional_margin", "conditional_hard"),
                   choices=("conditional", "conditional_margin", "conditional_hard"))
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--lambda-ce", type=float, default=0.5)
    p.add_argument("--lambda-margin", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--negative-focal-gamma", type=float, default=2.0)
    p.add_argument("--hard-negative-fraction", type=float, default=0.75)
    p.add_argument("--positive-fraction", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def macro_f1(true, pred, num_classes):
    scores = []
    for cls in range(num_classes):
        tp = ((true == cls) & (pred == cls)).sum().item()
        fp = ((true != cls) & (pred == cls)).sum().item()
        fn = ((true == cls) & (pred != cls)).sum().item()
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        scores.append(2 * p * r / max(1e-12, p + r))
    return sum(scores) / len(scores)


@torch.inference_mode()
def evaluate(logits, obj_targets, class_targets):
    probability, prediction = logits.sigmoid().max(dim=1)
    pred_obj = probability >= 0.5; true_obj = obj_targets.bool()
    tp = (pred_obj & true_obj).sum().item(); fp = (pred_obj & ~true_obj).sum().item()
    fn = (~pred_obj & true_obj).sum().item()
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    obj_f1 = 2 * precision * recall / max(1e-12, precision + recall)
    positive = class_targets >= 0
    cls_f1 = macro_f1(class_targets[positive], prediction[positive], logits.shape[1])
    return {"objectness_precision": precision, "objectness_recall": recall,
            "objectness_f1": obj_f1, "class_macro_f1": cls_f1,
            "selection_score": 0.5 * (obj_f1 + cls_f1)}


def conditional_loss(logits, class_targets, class_weights, negative_gamma, lambda_ce,
                     lambda_margin, margin):
    positive = class_targets >= 0
    targets = torch.zeros_like(logits)
    if positive.any():
        targets[positive, class_targets[positive]] = 1.0
    probabilities = logits.sigmoid()
    positive_loss = -targets * F.logsigmoid(logits) * class_weights.unsqueeze(0)
    negative_loss = -(1.0 - targets) * probabilities.pow(negative_gamma) * F.logsigmoid(-logits)
    bce = (positive_loss + negative_loss).sum(dim=1).mean()
    ce = F.cross_entropy(logits[positive], class_targets[positive], weight=class_weights) if positive.any() else bce * 0
    ranking = bce * 0
    if positive.any() and lambda_margin > 0:
        true_logits = logits[positive].gather(1, class_targets[positive, None]).squeeze(1)
        masked = logits[positive].clone()
        masked.scatter_(1, class_targets[positive, None], float("-inf"))
        ranking = F.relu(margin - true_logits + masked.max(dim=1).values).mean()
    return bce + lambda_ce * ce + lambda_margin * ranking, {"bce": bce, "ce": ce, "margin": ranking}


def controlled_indices(obj_targets, class_targets, baseline_score, batch, steps,
                       hard_fraction, positive_fraction, generator):
    positive_indices = torch.where(obj_targets.bool())[0]
    background_indices = torch.where(~obj_targets.bool())[0]
    hard_order = background_indices[baseline_score[background_indices].argsort(descending=True)]
    hard_pool = hard_order[:max(batch, len(hard_order) // 2)]
    by_class = [torch.where(class_targets == cls)[0] for cls in range(9)]
    class_probs = torch.tensor([math.sqrt(max(1, len(x))) for x in by_class], dtype=torch.float32)
    class_probs /= class_probs.sum()
    positive_count = round(batch * positive_fraction); background_count = batch - positive_count
    hard_count = round(background_count * hard_fraction); easy_count = background_count - hard_count
    for _ in range(steps):
        sampled_classes = torch.multinomial(class_probs, positive_count, replacement=True, generator=generator)
        positives = torch.tensor([
            by_class[int(cls)][torch.randint(len(by_class[int(cls)]), (1,), generator=generator)].item()
            for cls in sampled_classes
        ])
        hard = hard_pool[torch.randint(len(hard_pool), (hard_count,), generator=generator)]
        easy = background_indices[torch.randint(len(background_indices), (easy_count,), generator=generator)]
        yield torch.cat([positives, hard, easy])


def main():
    args = parse_args()
    if args.output.exists():
        if not args.overwrite: raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    if train["classes"] != val["classes"]: raise ValueError("Cache classes differ")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    x_train = train["features"].float().to(device)
    y_obj = train["objectness_targets"].to(device)
    y_cls = train["class_targets"].long().to(device)
    x_val = val["features"].float().to(device)
    vy_obj = val["objectness_targets"].to(device)
    vy_cls = val["class_targets"].long().to(device)
    baseline_score = (train["detector_scores"] * train["baseline_objectness_logits"].float().sigmoid()).to(device)
    counts = Counter(y_cls[y_cls >= 0].cpu().tolist()); largest = max(counts.values())
    class_weights = torch.tensor([(largest / max(1, counts[i])) ** 0.5 for i in range(9)], device=device)
    class_weights /= class_weights.mean()
    baseline_logits = val["baseline_class_logits"].float().to(device)
    baseline_obj = val["baseline_objectness_logits"].float().sigmoid().to(device)
    baseline_joint = torch.logit(baseline_obj.clamp(1e-5, 1-1e-5))[:, None] + F.log_softmax(baseline_logits, dim=1)
    report = {"baseline_factorized": evaluate(baseline_joint, vy_obj, vy_cls), "variants": {}}

    for variant_index, variant in enumerate(args.variants):
        torch.manual_seed(args.seed + variant_index)
        model = ConditionalVerifierHead(x_train.shape[1], 9, args.hidden, args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, stale, history = -1.0, 0, []
        use_margin = variant != "conditional"
        controlled = variant == "conditional_hard"
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []
            steps = math.ceil(len(x_train) / args.batch)
            if controlled:
                generator = torch.Generator().manual_seed(args.seed + epoch)
                batches = controlled_indices(y_obj.cpu(), y_cls.cpu(), baseline_score.cpu(), args.batch, steps,
                                             args.hard_negative_fraction, args.positive_fraction, generator)
            else:
                order = torch.randperm(len(x_train), device=device)
                batches = order.split(args.batch)
            for indices in batches:
                indices = indices.to(device)
                logits = model(x_train[indices])
                loss, _ = conditional_loss(
                    logits, y_cls[indices], class_weights, args.negative_focal_gamma,
                    args.lambda_ce, args.lambda_margin if use_margin else 0.0, args.margin,
                )
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode(): metrics = evaluate(model(x_val), vy_obj, vy_cls)
            record = {"epoch": epoch, "loss": sum(losses)/len(losses), **metrics}
            history.append(record)
            improved = metrics["selection_score"] > best + 1e-5
            if improved:
                best, stale = metrics["selection_score"], 0
                torch.save({"model": model.state_dict(), "feature_dim": x_train.shape[1],
                            "hidden": args.hidden, "dropout": args.dropout,
                            "classes": train["classes"], "variant": variant,
                            "metrics": record, "config": vars(args)}, args.output / f"{variant}.pt")
            else:
                stale += 1
            print(json.dumps({"variant": variant, **record}))
            if stale >= args.patience: break
        report["variants"][variant] = {"best_score": best, "history": history,
                                        "best_metrics": max(history, key=lambda x: x["selection_score"])}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v["best_metrics"] for k, v in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
