#!/usr/bin/env python3
"""Train one-to-one suppress/rescue objectness using cached proposal sets."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from torch.nn import functional as F

from proposal_relation import (grouped_indices, load_curated_relation_metadata,
                               pad_group_batch)
from proposal_set_objectness import ProposalSetObjectness, load_frozen_relation
from train_proposal_relation_head import limit_group


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--relation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-objectness-delta", type=float, default=1.0)
    parser.add_argument("--no-object-weight", type=float, default=0.2)
    parser.add_argument("--batch-images", type=int, default=16)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-rank", type=float, default=0.5)
    parser.add_argument("--lambda-specialize", type=float, default=0.2)
    parser.add_argument("--lambda-residual", type=float, default=2e-3)
    parser.add_argument("--rank-margin", type=float, default=0.7)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def one_to_one_targets(metadata, cache):
    """Choose exactly one highest-IoU cached proposal for every represented GT."""
    winners = torch.zeros(len(cache["features"]), dtype=torch.bool)
    duplicates = torch.zeros_like(winners)
    by_reference = {}
    # Reference ids are only unique inside an image.
    for index, (image_id, reference) in enumerate(zip(metadata["image_ids"], metadata["reference_gt_ids"])):
        if reference:
            by_reference.setdefault((image_id, reference), []).append(index)
    # IoU is added by the metadata loader below. Highest IoU avoids inheriting
    # the detector's old score ordering when deciding the one retained box.
    for indices in by_reference.values():
        best = max(indices, key=lambda index: metadata["ious"][index])
        winners[best] = True
        for index in indices:
            duplicates[index] = index != best
    return winners, duplicates


def add_ious(metadata, data: Path, split: str):
    import csv
    with (data / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    metadata["ious"] = [float(row["iou"] or 0) for row in rows]


def prepare(cache, metadata, device):
    winners, duplicates = one_to_one_targets(metadata, cache)
    references, mapping = [], {}
    for image_id, reference in zip(metadata["image_ids"], metadata["reference_gt_ids"]):
        if not reference:
            references.append(-1)
        else:
            key = (image_id, reference)
            if key not in mapping: mapping[key] = len(mapping)
            references.append(mapping[key])
    return {
        "features": cache["features"].float().to(device),
        "auxiliary": metadata["auxiliary"].to(device),
        "baseline_objectness": cache["baseline_objectness_logits"].float().to(device),
        "baseline_classes": cache["baseline_class_logits"].float().to(device),
        "objectness_targets": cache["objectness_targets"].bool().to(device),
        "class_targets": cache["class_targets"].long().to(device),
        "primary_targets": winners.to(device),
        "duplicate_targets": duplicates.to(device),
        "reference_ids": torch.tensor(references, dtype=torch.long, device=device),
    }


def set_loss(output, batch, args):
    valid = ~batch["padding_mask"]
    winner = batch["primary_targets"].bool() & valid
    unmatched = (~batch["primary_targets"].bool()) & valid
    duplicate = batch["duplicate_targets"].bool() & valid
    background = (~batch["objectness_targets"].bool()) & valid
    baseline_probability = batch["baseline_objectness"].sigmoid()
    # Current high-score false positives receive the strongest supervision;
    # weak winners receive extra rescue weight without defining a new threshold.
    weights = torch.ones_like(baseline_probability)
    # DETR-style low no-object weight prevents the much larger unmatched set
    # from drowning the one-per-GT positives. Only current high-score errors
    # approach full weight; easy background stays cheap.
    weights[unmatched] = args.no_object_weight + (
        1.0 - args.no_object_weight) * baseline_probability[unmatched]
    weights[duplicate] = 0.5 + 0.5 * baseline_probability[duplicate]
    weights[winner] += 2.0 * (1.0 - baseline_probability[winner])
    targets = winner.float()
    bce = F.binary_cross_entropy_with_logits(output["objectness_logits"][valid],
                                             targets[valid], reduction="none")
    focal = torch.where(targets[valid] > 0, 1.0 - output["objectness_logits"][valid].sigmoid(),
                        output["objectness_logits"][valid].sigmoid()).square()
    keep_loss = (bce * (1.0 + focal) * weights[valid]).sum() / weights[valid].sum()
    detector_logit = batch["auxiliary"][..., 8] * 5.0
    score = detector_logit + output["objectness_logits"]
    ranking = []
    for row in range(len(score)):
        positive_scores = score[row][winner[row]]
        negative_scores = score[row][unmatched[row]]
        if len(positive_scores) and len(negative_scores):
            hard = negative_scores.topk(min(12, len(negative_scores))).values
            ranking.append(F.softplus(args.rank_margin - positive_scores[:, None] + hard[None, :]).mean())
        refs = batch["reference_ids"][row]
        for reference in refs[valid[row]].unique():
            if reference < 0: continue
            same = (refs == reference) & valid[row]
            win = same & winner[row]; dup = same & duplicate[row]
            if win.any() and dup.any():
                ranking.append(F.softplus(args.rank_margin - score[row][win].max() + score[row][dup]).mean())
    rank_loss = torch.stack(ranking).mean() if ranking else keep_loss * 0
    # Encourage branch specialization without forcing every unmatched proposal
    # to receive a large fixed suppression.
    weak_winner = winner & (baseline_probability < 0.5)
    specialize_terms = []
    if unmatched.any():
        specialize_terms.append(F.relu(output["rescue"][unmatched]).mean())
    if weak_winner.any():
        specialize_terms.append(F.relu(output["suppression"][weak_winner]).mean())
    specialize = torch.stack(specialize_terms).mean() if specialize_terms else keep_loss * 0
    residual = output["objectness_delta"][valid].square().mean()
    loss = (keep_loss + args.lambda_rank * rank_loss + args.lambda_specialize * specialize
            + args.lambda_residual * residual)
    return loss, {"keep": keep_loss, "rank": rank_loss, "specialize": specialize,
                  "residual": residual, "background": background.sum(), "duplicate": duplicate.sum()}


@torch.inference_mode()
def validate(model, tensors, groups, args):
    model.eval(); logits, targets, losses = [], [], []
    suppressions, rescues = [], []
    for start in range(0, len(groups), args.batch_images):
        batch = pad_group_batch(groups[start:start + args.batch_images], **tensors)
        output = model(batch["features"], batch["auxiliary"], batch["baseline_objectness"],
                       batch["baseline_classes"], batch["padding_mask"])
        loss, _ = set_loss(output, batch, args); losses.append(float(loss))
        valid = ~batch["padding_mask"]
        logits.append(output["objectness_logits"][valid].cpu())
        targets.append(batch["primary_targets"][valid].cpu())
        suppressions.append(output["suppression"][valid].cpu())
        rescues.append((output["rescue_gate"] * output["rescue"])[valid].cpu())
    logits, targets = torch.cat(logits), torch.cat(targets).bool()
    prediction = logits >= 0
    tp = (prediction & targets).sum().item(); fp = (prediction & ~targets).sum().item()
    fn = (~prediction & targets).sum().item()
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"loss": sum(losses)/max(1, len(losses)), "keep_precision": precision,
            "keep_recall": recall, "keep_f1": f1,
            "mean_suppression": torch.cat(suppressions).mean().item(),
            "mean_gated_rescue": torch.cat(rescues).mean().item()}


def main():
    args = parse_args()
    if args.output.exists():
        if not args.overwrite: raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    relation_checkpoint = torch.load(args.relation_checkpoint, map_location="cpu", weights_only=False)
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_cache = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    train_meta = load_curated_relation_metadata(train_cache, args.data, "train"); add_ious(train_meta, args.data, "train")
    val_meta = load_curated_relation_metadata(val_cache, args.data, "val"); add_ious(val_meta, args.data, "val")
    train = prepare(train_cache, train_meta, device); val = prepare(val_cache, val_meta, device)
    train_groups = grouped_indices(train_meta["image_ids"]); val_groups = grouped_indices(val_meta["image_ids"])
    generator = torch.Generator().manual_seed(args.seed)
    train_score = train_cache["detector_scores"] * train_cache["baseline_objectness_logits"].float().sigmoid()
    val_score = val_cache["detector_scores"] * val_cache["baseline_objectness_logits"].float().sigmoid()
    train_groups = [limit_group(g, args.max_proposals, train_cache["objectness_targets"], train_score,
                                args.hard_negative_fraction, True, generator) for g in train_groups]
    val_groups = [limit_group(g, args.max_proposals, val_cache["objectness_targets"], val_score,
                              args.hard_negative_fraction, False, generator) for g in val_groups]
    model = ProposalSetObjectness(load_frozen_relation(relation_checkpoint),
                                  args.decision_hidden, args.dropout,
                                  args.max_objectness_delta).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train(); order = list(range(len(train_groups))); random.Random(args.seed + epoch).shuffle(order)
            for start in range(0, len(order), args.batch_images):
                batch = pad_group_batch([train_groups[i] for i in order[start:start+args.batch_images]], **train)
                output = model(batch["features"], batch["auxiliary"], batch["baseline_objectness"],
                               batch["baseline_classes"], batch["padding_mask"])
                loss, _ = set_loss(output, batch, args)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)
                optimizer.step()
        metrics = validate(model, val, val_groups, args)
        record = {"epoch": epoch, **metrics}; history.append(record); print(json.dumps(record), flush=True)
        torch.save({"model": model.state_dict(), "classes": relation_checkpoint["classes"],
                    "relation_checkpoint": str(args.relation_checkpoint), "relation_config": relation_checkpoint["config"],
                    "feature_dim": relation_checkpoint["feature_dim"],
                    "auxiliary_dim": relation_checkpoint["auxiliary_dim"], "epoch": epoch,
                    "config": vars(args), "metrics": record}, args.output/f"epoch_{epoch:03d}.pt")
    best = max(history, key=lambda item: item["keep_f1"])
    shutil.copy2(args.output/f"epoch_{best['epoch']:03d}.pt", args.output/"best_curated.pt")
    (args.output/"training_report.json").write_text(json.dumps({"history":history,"best":best},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
