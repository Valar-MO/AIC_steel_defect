#!/usr/bin/env python3
"""Train a zero-initialized proposal graph transformer on cached V2.1 features."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from proposal_relation import (ProposalRelationTransformer, grouped_indices,
                               load_curated_relation_metadata, pad_group_batch)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-images", type=int, default=16)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-class", type=float, default=0.5)
    parser.add_argument("--lambda-rank", type=float, default=0.25)
    parser.add_argument("--lambda-residual", type=float, default=1e-3)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def macro_f1(targets, predictions, classes=9):
    values = []
    for cls in range(classes):
        true, pred = targets == cls, predictions == cls
        tp = (true & pred).sum().item(); fp = (~true & pred).sum().item(); fn = (true & ~pred).sum().item()
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        values.append(2 * precision * recall / max(1e-12, precision + recall))
    return sum(values) / classes


def make_reference_ids(image_ids, references):
    result, mapping = [], {}
    for image_id, reference in zip(image_ids, references):
        if not reference:
            result.append(-1)
            continue
        key = (image_id, reference)
        if key not in mapping:
            mapping[key] = len(mapping)
        result.append(mapping[key])
    return torch.tensor(result, dtype=torch.long)


def limit_group(group, maximum, objectness, baseline_score, hard_fraction, training, generator):
    if len(group) <= maximum:
        return group
    positive = [i for i in group if bool(objectness[i])]
    background = [i for i in group if not bool(objectness[i])]
    positive.sort(key=lambda i: float(baseline_score[i]), reverse=True)
    remaining = max(0, maximum - min(len(positive), maximum))
    hard_count = min(remaining, max(1, round(remaining * hard_fraction)))
    background.sort(key=lambda i: float(baseline_score[i]), reverse=True)
    selected_background = background[:hard_count]
    easy = background[hard_count:]
    if training and remaining > hard_count and easy:
        order = torch.randperm(len(easy), generator=generator).tolist()
        selected_background += [easy[i] for i in order[:remaining - hard_count]]
    else:
        selected_background += easy[:remaining - hard_count]
    return positive[:maximum] + selected_background


def relation_loss(output, batch, class_weights, args):
    valid = ~batch["padding_mask"]
    primary = batch["primary_targets"].bool() & valid
    background = (~batch["objectness_targets"].bool()) & valid
    supervised = primary | background
    obj_loss = F.binary_cross_entropy_with_logits(
        output["objectness_logits"][supervised], primary[supervised].float(),
    )
    positive = (batch["class_targets"] >= 0) & valid
    cls_loss = F.cross_entropy(
        output["class_logits"][positive], batch["class_targets"][positive], weight=class_weights,
    )
    detector_logit = batch["auxiliary"][..., 8] * 5.0
    final_score_logit = detector_logit + output["objectness_logits"]
    ranking = []
    for row in range(len(final_score_logit)):
        positives = final_score_logit[row][primary[row]]
        negatives = final_score_logit[row][background[row]]
        if len(positives) and len(negatives):
            hardest = negatives.topk(min(8, len(negatives))).values
            ranking.append(F.softplus(args.rank_margin - positives[:, None] + hardest[None, :]).mean())
        references = batch["reference_ids"][row]
        for reference in references[valid[row]].unique():
            if reference < 0:
                continue
            same = (references == reference) & valid[row]
            winners = same & primary[row]
            duplicates = same & (~primary[row])
            if winners.any() and duplicates.any():
                ranking.append(F.softplus(
                    args.rank_margin - final_score_logit[row][winners].max()
                    + final_score_logit[row][duplicates]
                ).mean())
    rank_loss = torch.stack(ranking).mean() if ranking else obj_loss * 0
    residual = (output["objectness_delta"][valid].square().mean()
                + output["class_delta"][valid].square().mean())
    total = (obj_loss + args.lambda_class * cls_loss + args.lambda_rank * rank_loss
             + args.lambda_residual * residual)
    return total, {"obj": obj_loss, "cls": cls_loss, "rank": rank_loss, "residual": residual}


@torch.inference_mode()
def validate(model, tensors, groups, class_weights, args, device):
    model.eval(); all_obj, all_cls = [], []
    target_obj, target_cls = [], []
    losses = []
    for start in range(0, len(groups), args.batch_images):
        batch = pad_group_batch(groups[start:start + args.batch_images], **tensors)
        output = model(batch["features"], batch["auxiliary"], batch["baseline_objectness"],
                       batch["baseline_classes"], batch["padding_mask"])
        loss, _ = relation_loss(output, batch, class_weights, args)
        valid = ~batch["padding_mask"]
        losses.append(float(loss)); all_obj.append(output["objectness_logits"][valid].cpu())
        all_cls.append(output["class_logits"][valid].cpu())
        target_obj.append(batch["primary_targets"][valid].cpu())
        target_cls.append(batch["class_targets"][valid].cpu())
    obj_logits, cls_logits = torch.cat(all_obj), torch.cat(all_cls)
    obj_target, cls_target = torch.cat(target_obj).bool(), torch.cat(target_cls)
    obj_pred = obj_logits >= 0
    tp = (obj_pred & obj_target).sum().item(); fp = (obj_pred & ~obj_target).sum().item()
    fn = (~obj_pred & obj_target).sum().item()
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    positive = cls_target >= 0
    class_f1 = macro_f1(cls_target[positive], cls_logits[positive].argmax(1))
    return {"loss": sum(losses) / max(1, len(losses)), "keep_precision": precision,
            "keep_recall": recall, "keep_f1": f1, "class_macro_f1": class_f1,
            "selection_score": 0.6 * f1 + 0.4 * class_f1}


def prepare(cache, metadata, device):
    outcomes = metadata["audit_outcomes"]
    return {
        "features": cache["features"].float().to(device),
        "auxiliary": metadata["auxiliary"].to(device),
        "baseline_objectness": cache["baseline_objectness_logits"].float().to(device),
        "baseline_classes": cache["baseline_class_logits"].float().to(device),
        "objectness_targets": cache["objectness_targets"].bool().to(device),
        "class_targets": cache["class_targets"].long().to(device),
        "primary_targets": torch.tensor([value == "tp" for value in outcomes]).to(device),
        "reference_ids": make_reference_ids(metadata["image_ids"], metadata["reference_gt_ids"]).to(device),
    }


def main():
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_cache = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    train_meta = load_curated_relation_metadata(train_cache, args.data, "train")
    val_meta = load_curated_relation_metadata(val_cache, args.data, "val")
    train = prepare(train_cache, train_meta, device); val = prepare(val_cache, val_meta, device)
    train_groups = grouped_indices(train_meta["image_ids"]); val_groups = grouped_indices(val_meta["image_ids"])
    baseline_score = train_cache["detector_scores"] * train_cache["baseline_objectness_logits"].float().sigmoid()
    generator = torch.Generator().manual_seed(args.seed)
    train_groups = [limit_group(group, args.max_proposals, train_cache["objectness_targets"],
                                baseline_score, args.hard_negative_fraction, True, generator)
                    for group in train_groups]
    val_score = val_cache["detector_scores"] * val_cache["baseline_objectness_logits"].float().sigmoid()
    val_groups = [limit_group(group, args.max_proposals, val_cache["objectness_targets"],
                              val_score, args.hard_negative_fraction, False, generator)
                  for group in val_groups]
    counts = Counter(train_cache["class_targets"][train_cache["class_targets"] >= 0].tolist())
    largest = max(counts.values())
    class_weights = torch.tensor([(largest / max(1, counts[i])) ** 0.5 for i in range(9)], device=device)
    class_weights /= class_weights.mean()
    model = ProposalRelationTransformer(
        feature_dim=train["features"].shape[1], auxiliary_dim=train["auxiliary"].shape[1],
        dimension=args.dimension, heads=args.heads, layers=args.layers, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            model.train(); order = list(range(len(train_groups))); random.Random(args.seed + epoch).shuffle(order)
            epoch_losses = []
            for start in range(0, len(order), args.batch_images):
                groups = [train_groups[i] for i in order[start:start + args.batch_images]]
                batch = pad_group_batch(groups, **train)
                output = model(batch["features"], batch["auxiliary"], batch["baseline_objectness"],
                               batch["baseline_classes"], batch["padding_mask"])
                loss, parts = relation_loss(output, batch, class_weights, args)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                epoch_losses.append(float(loss.detach()))
        metrics = validate(model, val, val_groups, class_weights, args, device)
        record = {"epoch": epoch, **metrics}
        history.append(record); print(json.dumps(record), flush=True)
        checkpoint = {
            "model": model.state_dict(), "classes": train_cache["classes"],
            "feature_dim": train["features"].shape[1], "auxiliary_dim": train["auxiliary"].shape[1],
            "config": vars(args), "epoch": epoch, "metrics": record,
        }
        torch.save(checkpoint, args.output / f"epoch_{epoch:03d}.pt")
    best = max(history, key=lambda item: item["selection_score"])
    shutil.copy2(args.output / f"epoch_{best['epoch']:03d}.pt", args.output / "best_curated.pt")
    report = {"baseline_epoch_zero": history[0], "best_curated": best, "history": history,
              "train_images": len(train_groups), "val_images": len(val_groups)}
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report | {"history": "omitted"}, indent=2))


if __name__ == "__main__":
    main()

