#!/usr/bin/env python3
"""Train a frozen ROI Selection residual using final-fusion utility labels.

The detector, proposals, boxes, classes and official fusion protocol stay
unchanged.  Only a bounded ROI residual is learned from cached P2/P3/query
features.  Labels are defined by the counterfactual whole+tile audit, rather
than by a single-view IoU label.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from roi_refinement_common import ROISelectionResidualHead, geometry_features


ESSENTIAL, REPLACEABLE, DUPLICATE, WRONG_CLASS, NEAR, FINAL_BG, POTENTIAL_BG, HARMLESS = range(8)


class CacheDataset(Dataset):
    def __init__(self, cache): self.cache = cache
    def __len__(self): return len(self.cache["query"])
    def __getitem__(self, index):
        return {key: value[index] for key, value in self.cache.items() if isinstance(value, torch.Tensor)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=768)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--score-delta-max", type=float, default=.25)
    p.add_argument("--essential-tolerance", type=float, default=.015)
    p.add_argument("--replaceable-tolerance", type=float, default=.08)
    p.add_argument("--final-bg-margin", type=float, default=.15)
    p.add_argument("--potential-bg-margin", type=float, default=.08)
    p.add_argument("--duplicate-margin", type=float, default=.04)
    p.add_argument("--essential-preserve-weight", type=float, default=.50)
    p.add_argument("--replaceable-preserve-weight", type=float, default=.15)
    p.add_argument("--final-bg-weight", type=float, default=.45)
    p.add_argument("--potential-bg-weight", type=float, default=.20)
    p.add_argument("--duplicate-weight", type=float, default=.06)
    p.add_argument("--anchor-weight", type=float, default=.02)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def weighted_sampler(cache):
    """Balance original images/views, then emphasize final-fusion utility roles."""
    source = cache["source_id"].long()
    image = cache["candidate_image_index"].long()
    group = source * (int(image.max()) + 1) + image
    _, inverse, counts = torch.unique(group, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse].double()
    source_counts = torch.bincount(source, minlength=2).double().clamp_min(1)
    weights *= source_counts.sum() / (2 * source_counts[source])
    role = cache["utility_role_id"].long()
    multipliers = torch.tensor((5., 2.5, 1.2, .8, .8, 4.5, 2.5, .5), dtype=torch.double)
    return WeightedRandomSampler(weights * multipliers[role], len(weights), replacement=True)


def protocol_logit(raw_logit, source, border):
    """Exact official tile penalty: sigmoid first, then multiply score by .5."""
    factor = torch.where(source.bool() & border.bool(), .5, 1.).to(raw_logit.dtype)
    probability = (raw_logit.sigmoid() * factor).clamp(1e-6, 1 - 1e-6)
    return torch.logit(probability)


def pair_rank(positive, negative, margin):
    if not positive.numel() or not negative.numel():
        return (positive.sum() if positive.numel() else negative.sum()) * 0
    count = min(positive.numel(), negative.numel())
    positive = positive[torch.randint(positive.numel(), (count,), device=positive.device)]
    negative = negative[torch.randint(negative.numel(), (count,), device=negative.device)]
    return F.softplus(margin - positive + negative).mean()


def duplicate_rank(scores, roles, groups, margin):
    """Keep a final TP above same-GT duplicate candidates, at low weight."""
    values = []
    positives = (roles == ESSENTIAL) | (roles == REPLACEABLE)
    duplicates = roles == DUPLICATE
    for group in torch.unique(groups[positives]):
        if group < 0: continue
        winner = scores[positives & (groups == group)]
        rival = scores[duplicates & (groups == group)]
        if winner.numel() and rival.numel(): values.append(pair_rank(winner, rival, margin))
    return torch.stack(values).mean() if values else scores.sum() * 0


def loss_for_batch(model, batch, device, opt):
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    geometry = geometry_features(batch["boxes_cxcywh"].float(), batch["source_id"],
                                 batch["touches_internal_border"], batch["context_clipped"],
                                 batch["border_distance"].float())
    delta = model(batch["p2_roi"].float(), batch["p3_roi"].float(), batch["query"].float(), geometry)
    raw_base = batch["selection_logit"].float()
    base = protocol_logit(raw_base, batch["source_id"], batch["touches_internal_border"])
    refined = protocol_logit(raw_base + delta, batch["source_id"], batch["touches_internal_border"])
    roles = batch["utility_role_id"].long()
    essential, replaceable = roles == ESSENTIAL, roles == REPLACEABLE
    positives = essential | replaceable
    final_bg, potential_bg = roles == FINAL_BG, roles == POTENTIAL_BG
    preserve_essential = (F.relu(base[essential] - opt.essential_tolerance - refined[essential]).mean()
                          if essential.any() else delta.sum() * 0)
    preserve_replaceable = (F.relu(base[replaceable] - opt.replaceable_tolerance - refined[replaceable]).mean()
                            if replaceable.any() else delta.sum() * 0)
    final_rank = pair_rank(refined[positives], refined[final_bg], opt.final_bg_margin)
    potential_rank = pair_rank(refined[positives], refined[potential_bg], opt.potential_bg_margin)
    duplicate = duplicate_rank(refined, roles, batch["utility_nearest_gt_group"], opt.duplicate_margin)
    anchor = delta.square().mean()
    total = (opt.essential_preserve_weight * preserve_essential +
             opt.replaceable_preserve_weight * preserve_replaceable +
             opt.final_bg_weight * final_rank + opt.potential_bg_weight * potential_rank +
             opt.duplicate_weight * duplicate + opt.anchor_weight * anchor)
    return {"total": total, "preserve_essential": preserve_essential,
            "preserve_replaceable": preserve_replaceable, "final_rank": final_rank,
            "potential_rank": potential_rank, "duplicate_rank": duplicate, "anchor": anchor}


@torch.inference_mode()
def validate(model, loader, device, opt):
    model.eval(); totals, steps = {}, 0
    for batch in loader:
        losses = loss_for_batch(model, batch, device, opt); steps += 1
        for key, value in losses.items(): totals[key] = totals.get(key, 0.) + float(value)
    return {key: value / max(steps, 1) for key, value in totals.items()}


def main():
    opt = parse_args(); random.seed(opt.seed); torch.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)
    train = torch.load(opt.train_cache, map_location="cpu", weights_only=False)
    val = torch.load(opt.val_cache, map_location="cpu", weights_only=False)
    required = {"utility_role_id", "utility_nearest_gt_group"}
    for cache in (train, val):
        missing = required - set(cache)
        if missing: raise ValueError(f"Counterfactual utility cache is missing: {sorted(missing)}")
    if train["metadata"]["classes"] != val["metadata"]["classes"]: raise ValueError("Class mismatch")
    device = torch.device(opt.device)
    model = ROISelectionResidualHead(hidden=opt.hidden, score_delta_max=opt.score_delta_max).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr * .10)
    train_loader = DataLoader(CacheDataset(train), batch_size=opt.batch, sampler=weighted_sampler(train),
                              num_workers=opt.workers, pin_memory=True, persistent_workers=opt.workers > 0)
    val_loader = DataLoader(CacheDataset(val), batch_size=opt.batch * 2, shuffle=False,
                            num_workers=opt.workers, pin_memory=True, persistent_workers=opt.workers > 0)
    opt.output_dir.mkdir(parents=True, exist_ok=True); history = []
    for epoch in range(1, opt.epochs + 1):
        model.train(); totals, steps = {}, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = loss_for_batch(model, batch, device, opt); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); steps += 1
            for key, value in losses.items(): totals[key] = totals.get(key, 0.) + float(value.detach())
        scheduler.step()
        train_metrics = {key: value / steps for key, value in totals.items()}
        val_metrics = validate(model, val_loader, device, opt)
        checkpoint = {"architecture": "roi_selection_residual", "epoch": epoch, "model": model.state_dict(),
                      "classes": train["metadata"]["classes"],
                      "head": {"hidden": opt.hidden, "score_delta_max": opt.score_delta_max},
                      "hyperparameters": vars(opt), "train_cache": str(opt.train_cache.resolve()),
                      "val_cache": str(opt.val_cache.resolve())}
        torch.save(checkpoint, opt.output_dir / f"epoch_{epoch:02d}.pt")
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        history.append(row); print(json.dumps(row), flush=True)
    (opt.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
