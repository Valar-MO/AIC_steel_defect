#!/usr/bin/env python3
"""Train the cached P2/P3/query ROI head while the D-FINE detector is frozen."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.ops import generalized_box_iou

from roi_refinement_common import ROIRefinementHead, apply_box_delta, geometry_features


class CacheDataset(Dataset):
    def __init__(self, cache: dict): self.cache = cache
    def __len__(self): return len(self.cache["query"])
    def __getitem__(self, index): return {k: v[index] for k, v in self.cache.items() if isinstance(v, torch.Tensor)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch", type=int, default=384)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--score-delta-max", type=float, default=.35)
    p.add_argument("--class-delta-max", type=float, default=.75)
    p.add_argument("--box-delta-max", type=float, default=.20)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def make_sampler(cache: dict) -> WeightedRandomSampler:
    """50:50 source, then image/GT proxy and mild class balancing."""
    source = cache["source_id"].long(); fg = cache["max_iou"] >= .5
    labels = cache["nearest_gt_label"].long()
    weights = torch.ones(len(source), dtype=torch.double)
    # Candidate-level sampling would let one textured image with many duplicate
    # boxes dominate an epoch.  Equalize source-image groups before class weights.
    group = source * (int(cache["candidate_image_index"].max()) + 1) + cache["candidate_image_index"].long()
    _, inverse, group_count = torch.unique(group, return_inverse=True, return_counts=True)
    weights /= group_count[inverse].double()
    source_counts = torch.bincount(source, minlength=2).double().clamp_min(1)
    weights *= source_counts.sum() / (2 * source_counts[source])
    counts = torch.bincount(labels[fg].clamp_min(0), minlength=9).double().clamp_min(1)
    class_weight = (counts.max() / counts).sqrt().clamp(max=2.0)
    weights[fg] *= class_weight[labels[fg]]
    # Final NMS survivor background is exactly the failure mode R2 must reduce.
    weights[cache["is_final_hard_bg"]] *= 2.5
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def effective_logit(logit, source, border):
    return logit + ((source.bool() & border.bool()).to(logit.dtype) * torch.log(torch.tensor(.5, device=logit.device)))


def forward_loss(model, batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    geom = geometry_features(batch["boxes_cxcywh"].float(), batch["source_id"],
                             batch["touches_internal_border"], batch["context_clipped"],
                             batch["border_distance"].float())
    out = model(batch["p2_roi"].float(), batch["p3_roi"].float(), batch["query"].float(), geom)
    iou = batch["max_iou"].float(); foreground = iou >= .5
    final_bg = batch["is_final_hard_bg"]
    high_bg = (iou < .1) & (batch["selection_logit"] > -1.0)
    veto_mask = foreground | final_bg | high_bg
    veto_target = foreground.float()
    veto_weight = torch.where(final_bg, 3.0, torch.where(high_bg, 1.5, 1.0))
    veto = F.binary_cross_entropy_with_logits(out["veto_logit"][veto_mask], veto_target[veto_mask],
                                              weight=veto_weight[veto_mask]) if veto_mask.any() else out["veto_logit"].sum() * 0
    refined = batch["selection_logit"].float() + out["score_delta"]
    base_eff = effective_logit(batch["selection_logit"].float(), batch["source_id"], batch["touches_internal_border"])
    refined_eff = effective_logit(refined, batch["source_id"], batch["touches_internal_border"])
    winners = batch["is_base_tp"]
    preserve = F.relu(base_eff[winners] - .08 - refined_eff[winners]).mean() if winners.any() else refined.sum() * 0
    # Every final hard background must remain below a randomly paired base TP.
    hard = final_bg
    if winners.any() and hard.any():
        positive = refined_eff[winners]; negative = refined_eff[hard]
        pairs = min(len(positive), len(negative)); random_positive = positive[torch.randint(len(positive), (pairs,), device=device)]
        random_negative = negative[torch.randint(len(negative), (pairs,), device=device)]
        hard_rank = F.softplus(.15 - random_positive + random_negative).mean()
    else: hard_rank = refined.sum() * 0
    cls = batch["class_logits"].float() + out["class_delta"]
    class_loss = F.cross_entropy(cls[foreground], batch["nearest_gt_label"][foreground].long()) if foreground.any() else cls.sum() * 0
    refined_boxes = apply_box_delta(batch["boxes_cxcywh"].float(), out["box_delta"])
    if foreground.any():
        target = batch["target_box_cxcywh"].float()
        box_l1 = F.smooth_l1_loss(refined_boxes[foreground], target[foreground], beta=.05)
        pred_xyxy = torch.cat((refined_boxes[:, :2] - refined_boxes[:, 2:] / 2,
                               refined_boxes[:, :2] + refined_boxes[:, 2:] / 2), -1)
        target_xyxy = torch.cat((target[:, :2] - target[:, 2:] / 2,
                                 target[:, :2] + target[:, 2:] / 2), -1)
        giou = 1 - generalized_box_iou(pred_xyxy[foreground], target_xyxy[foreground]).diag().mean()
    else: box_l1 = giou = refined_boxes.sum() * 0
    # Role split: veto learns strong background evidence; residual only reorders.
    total = veto + .25 * preserve + .30 * hard_rank + .20 * class_loss + .35 * box_l1 + .20 * giou
    losses = {"total": total, "veto": veto, "preserve": preserve, "hard_rank": hard_rank,
              "class": class_loss, "box_l1": box_l1, "giou": giou}
    return losses


@torch.inference_mode()
def evaluate(model, loader, device):
    sums, count = {}, 0
    model.eval()
    for batch in loader:
        losses = forward_loss(model, batch, device)
        count += 1
        for key, value in losses.items(): sums[key] = sums.get(key, 0.) + float(value)
    return {key: value / max(count, 1) for key, value in sums.items()}


def main() -> None:
    opt = parse_args(); torch.manual_seed(opt.seed); random.seed(opt.seed)
    train = torch.load(opt.train_cache, map_location="cpu", weights_only=False)
    val = torch.load(opt.val_cache, map_location="cpu", weights_only=False)
    if train["metadata"]["classes"] != val["metadata"]["classes"]: raise ValueError("Class mismatch")
    device = torch.device(opt.device)
    model = ROIRefinementHead(score_delta_max=opt.score_delta_max, class_delta_max=opt.class_delta_max,
                              box_delta_max=opt.box_delta_max).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
    train_loader = DataLoader(CacheDataset(train), batch_size=opt.batch, sampler=make_sampler(train),
                              num_workers=opt.workers, pin_memory=True, persistent_workers=opt.workers > 0)
    val_loader = DataLoader(CacheDataset(val), batch_size=opt.batch * 2, shuffle=False,
                            num_workers=opt.workers, pin_memory=True, persistent_workers=opt.workers > 0)
    opt.output_dir.mkdir(parents=True, exist_ok=True); best = float("inf")
    history = []
    for epoch in range(1, opt.epochs + 1):
        model.train(); sums, count = {}, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = forward_loss(model, batch, device); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); count += 1
            for key, value in losses.items(): sums[key] = sums.get(key, 0.) + float(value.detach())
        scheduler.step(); train_metrics = {key: value / count for key, value in sums.items()}
        val_metrics = evaluate(model, val_loader, device); row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                                                                     "train": train_metrics, "val": val_metrics}
        history.append(row); print(json.dumps(row), flush=True)
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "classes": train["metadata"]["classes"],
                      "head": {"score_delta_max": opt.score_delta_max, "class_delta_max": opt.class_delta_max,
                               "box_delta_max": opt.box_delta_max}, "train_cache": str(opt.train_cache.resolve()),
                      "val_cache": str(opt.val_cache.resolve()), "hyperparameters": vars(opt)}
        torch.save(checkpoint, opt.output_dir / "last.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]; torch.save(checkpoint, opt.output_dir / "best.pt")
    (opt.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
