#!/usr/bin/env python3
"""Train a residual ranking head on cached query+DINO features and unique-GT targets."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--samples-per-epoch", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--residual-weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class CachedResidualHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden // 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, base: torch.Tensor):
        delta = self.output(self.body(features)).squeeze(1)
        return base + delta, delta


def load_rows(cache_path: Path, feature_path: Path):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    if "features" not in payload:
        raise KeyError(f"{feature_path} has no cached features")
    if len(payload["features"]) != len(cache["selection_logits"]):
        raise ValueError("Feature/cache length mismatch")
    features = torch.cat(
        [payload["features"].float(), cache["query_features"].float()], dim=1
    )
    positive = cache["verifier_positive"].bool()
    negative = cache["verifier_negative"].bool()
    valid = positive | negative
    target = positive.long()
    return cache, features, valid, target


def binary_ap(scores: torch.Tensor, target: torch.Tensor) -> float:
    order = scores.argsort(descending=True)
    ranked = target[order].double()
    total = int(ranked.sum())
    precision = ranked.cumsum(0) / torch.arange(1, len(ranked) + 1, dtype=torch.double)
    return float((precision * ranked).sum() / max(1, total))


def recall80_point(scores: torch.Tensor, target: torch.Tensor) -> dict:
    order = scores.argsort(descending=True)
    ranked = target[order].long()
    required = math.ceil(int(target.sum()) * 0.8 - 1e-12)
    locations = torch.where(ranked.cumsum(0) >= required)[0]
    if not len(locations):
        return {"reachable": False}
    count = int(locations[0]) + 1
    tp = int(ranked[:count].sum())
    return {"reachable": True, "tp": tp, "fp": count - tp, "prediction_count": count}


@torch.inference_mode()
def score_all(model, features, base, device, batch):
    model.eval()
    rows = []
    for start in range(0, len(features), batch):
        end = min(len(features), start + batch)
        score, _ = model(
            features[start:end].to(device),
            base[start:end].to(device),
        )
        rows.append(score.float().cpu())
    return torch.cat(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_cache, train_features, train_valid, train_target = load_rows(
        args.train_cache, args.train_features
    )
    val_cache, val_features, val_valid, val_target = load_rows(
        args.val_cache, args.val_features
    )
    train_indices = torch.where(train_valid)[0]
    positive = train_target[train_indices].bool()
    negative = ~positive
    weights = torch.zeros(len(train_indices), dtype=torch.double)
    weights[positive] = 0.5 / max(1, int(positive.sum()))
    hard = train_cache["selection_logits"][train_indices[negative]].float().sigmoid().pow(2)
    weights[negative] = (0.5 * hard / hard.sum().clamp_min(1e-12)).double()
    sampler = WeightedRandomSampler(
        weights,
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dataset = TensorDataset(
        train_features[train_indices],
        train_cache["selection_logits"][train_indices].float(),
        train_target[train_indices].float(),
    )
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, drop_last=True)
    device = torch.device(args.device)
    model = CachedResidualHead(
        train_features.shape[1], args.hidden, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    val_indices = torch.where(val_valid)[0]
    best_ap, best_fp, history = -1.0, math.inf, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for feature, base, target in loader:
            feature, base, target = (
                feature.to(device), base.to(device), target.to(device)
            )
            score, delta = model(feature, base)
            bce = F.binary_cross_entropy_with_logits(score, target)
            pos = score[target.bool()]
            neg = score[~target.bool()]
            rank = F.softplus(
                0.5
                - pos.topk(min(16, len(pos)), largest=False).values.unsqueeze(1)
                + neg.topk(min(16, len(neg))).values.unsqueeze(0)
            ).mean()
            loss = bce + args.rank_weight * rank + args.residual_weight * delta.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        scheduler.step()
        all_scores = score_all(
            model,
            val_features,
            val_cache["selection_logits"].float(),
            device,
            args.batch,
        )
        valid_scores = all_scores[val_indices]
        valid_targets = val_target[val_indices]
        ap = binary_ap(valid_scores, valid_targets)
        recall80 = recall80_point(valid_scores, valid_targets)
        record = {
            "epoch": epoch,
            "train_loss": total / seen,
            "val_unique_binary_ap": ap,
            "val_unique_recall80": recall80,
        }
        history.append(record)
        improved_ap = ap > best_ap + 1e-8
        improved_fp = recall80.get("reachable") and recall80["fp"] < best_fp
        if improved_ap:
            best_ap = ap
            torch.save(
                {"model": model.state_dict(), "scores": all_scores, "record": record},
                args.output / "best.pt",
            )
        if improved_fp:
            best_fp = recall80["fp"]
            torch.save(
                {"model": model.state_dict(), "scores": all_scores, "record": record},
                args.output / "best_recall80.pt",
            )
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(record), flush=True)
    print(json.dumps({
        "best_ap": best_ap,
        "best_recall80_fp": best_fp,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
