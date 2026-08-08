#!/usr/bin/env python3
"""Train R2b: a frozen P2/P3/query Selection-residual head only."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from roi_refinement_common import (ROISelectionResidualHead, ROIShapeSelectionResidualHead,
                                   geometry_features)


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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--score-delta-max", type=float, default=.35)
    p.add_argument("--hidden", type=int, default=256,
                   help="Residual-head width; 512 is the medium-capacity R2m setting.")
    p.add_argument("--shape-aware", action="store_true",
                   help="Use directional/stripe-preserving P2/P3 ROI readout (R2d).")
    p.add_argument("--preserve-tolerance", type=float, default=.06)
    p.add_argument("--hard-margin", type=float, default=.15)
    p.add_argument("--duplicate-margin", type=float, default=.05)
    p.add_argument("--duplicate-weight", type=float, default=.10)
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sampler(cache):
    source = cache["source_id"].long()
    group = source * (int(cache["candidate_image_index"].max()) + 1) + cache["candidate_image_index"].long()
    _, inverse, counts = torch.unique(group, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse].double()
    source_counts = torch.bincount(source, minlength=2).double().clamp_min(1)
    weights *= source_counts.sum() / (2 * source_counts[source])
    fg = cache["is_base_tp"]
    labels = cache["nearest_gt_label"].long()
    label_counts = torch.bincount(labels[fg].clamp_min(0), minlength=9).double().clamp_min(1)
    weights[fg] *= (label_counts.max() / label_counts[labels[fg]]).sqrt().clamp(max=2.)
    weights[cache["is_final_hard_bg"]] *= 2.5
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def effective_logit(logit, source, border):
    penalty = (source.bool() & border.bool()).to(logit.dtype) * torch.log(torch.tensor(.5, device=logit.device))
    return logit + penalty


def ranking_loss(positive, negative, margin):
    if not len(positive) or not len(negative): return positive.sum() * 0 if len(positive) else negative.sum() * 0
    count = min(len(positive), len(negative))
    positive = positive[torch.randint(len(positive), (count,), device=positive.device)]
    negative = negative[torch.randint(len(negative), (count,), device=negative.device)]
    return F.softplus(margin - positive + negative).mean()


def duplicate_loss(scores, base_tp, duplicate, group_ids, margin):
    values = []
    # A duplicate is only compared with the final base winner of exactly the same GT.
    for group in torch.unique(group_ids[base_tp]):
        if group < 0: continue
        winner = scores[base_tp & (group_ids == group)]
        rivals = scores[duplicate & (group_ids == group)]
        if len(winner) and len(rivals): values.append(ranking_loss(winner, rivals, margin))
    return torch.stack(values).mean() if values else scores.sum() * 0


def loss_for_batch(model, batch, device, opt):
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    geometry = geometry_features(batch["boxes_cxcywh"].float(), batch["source_id"],
                                 batch["touches_internal_border"], batch["context_clipped"],
                                 batch["border_distance"].float())
    delta = model(batch["p2_roi"].float(), batch["p3_roi"].float(), batch["query"].float(), geometry)
    base = effective_logit(batch["selection_logit"].float(), batch["source_id"], batch["touches_internal_border"])
    refined = base + delta
    winners, hard_bg, duplicates = batch["is_base_tp"], batch["is_final_hard_bg"], batch["is_duplicate"]
    preserve = F.relu(base[winners] - opt.preserve_tolerance - refined[winners]).mean() if winners.any() else delta.sum() * 0
    hard_rank = ranking_loss(refined[winners], refined[hard_bg], opt.hard_margin)
    duplicate_rank = duplicate_loss(refined, winners, duplicates, batch["nearest_gt_group"], opt.duplicate_margin)
    # Stops an unidentifiable common logit shift while leaving relative ordering free.
    anchor = delta.square().mean()
    total = .30 * preserve + .35 * hard_rank + opt.duplicate_weight * duplicate_rank + .01 * anchor
    return {"total": total, "preserve": preserve, "hard_rank": hard_rank,
            "duplicate_rank": duplicate_rank, "anchor": anchor}


@torch.inference_mode()
def validate(model, loader, device, opt):
    model.eval(); totals, steps = {}, 0
    for batch in loader:
        losses = loss_for_batch(model, batch, device, opt); steps += 1
        for key, value in losses.items(): totals[key] = totals.get(key, 0.) + float(value)
    return {key: value / max(steps, 1) for key, value in totals.items()}


def main():
    opt = parse_args(); random.seed(opt.seed); torch.manual_seed(opt.seed)
    train = torch.load(opt.train_cache, map_location="cpu", weights_only=False)
    val = torch.load(opt.val_cache, map_location="cpu", weights_only=False)
    for cache in (train, val):
        if "nearest_gt_group" not in cache: raise ValueError("Relabel caches with label_roi_refinement_cache.py first")
    device = torch.device(opt.device)
    head_class = ROIShapeSelectionResidualHead if opt.shape_aware else ROISelectionResidualHead
    model = head_class(hidden=opt.hidden, score_delta_max=opt.score_delta_max).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
    train_loader = DataLoader(CacheDataset(train), batch_size=opt.batch, sampler=sampler(train), num_workers=opt.workers,
                              pin_memory=True, persistent_workers=opt.workers > 0)
    val_loader = DataLoader(CacheDataset(val), batch_size=opt.batch * 2, shuffle=False, num_workers=opt.workers,
                            pin_memory=True, persistent_workers=opt.workers > 0)
    opt.output_dir.mkdir(parents=True, exist_ok=True); history = []
    for epoch in range(1, opt.epochs + 1):
        model.train(); train_total, steps = {}, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True); losses = loss_for_batch(model, batch, device, opt)
            losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); steps += 1
            for key, value in losses.items(): train_total[key] = train_total.get(key, 0.) + float(value.detach())
        schedule.step(); train_metrics = {key: value / steps for key, value in train_total.items()}
        val_metrics = validate(model, val_loader, device, opt)
        checkpoint = {"architecture": "roi_selection_residual_shape" if opt.shape_aware else "roi_selection_residual",
                      "epoch": epoch, "model": model.state_dict(),
                      "classes": train["metadata"]["classes"],
                      "head": {"hidden": opt.hidden, "score_delta_max": opt.score_delta_max},
                      "hyperparameters": vars(opt), "train_cache": str(opt.train_cache.resolve()),
                      "val_cache": str(opt.val_cache.resolve())}
        torch.save(checkpoint, opt.output_dir / f"epoch_{epoch:02d}.pt")
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        history.append(row); print(json.dumps(row), flush=True)
    (opt.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
