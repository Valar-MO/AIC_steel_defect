"""Shared utilities for cached Hierarchical D-FINE query classification."""

from __future__ import annotations

from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F


class HierarchicalQueryClassifier(nn.Module):
    """The detector's existing linear conditional nine-class head."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

    @torch.no_grad()
    def initialize_from_detector(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        if self.linear.weight.shape != weight.shape or self.linear.bias.shape != bias.shape:
            raise ValueError(
                f"Detector head shape mismatch: expected {tuple(self.linear.weight.shape)} / "
                f"{tuple(self.linear.bias.shape)}, got {tuple(weight.shape)} / {tuple(bias.shape)}"
            )
        self.linear.weight.copy_(weight)
        self.linear.bias.copy_(bias)


class SelectionResidualRanker(nn.Module):
    """Small residual ranker that preserves detector Selection as the prior."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        residual_scale: float = 0.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(features).squeeze(-1)) * self.residual_scale

    def forward(self, features: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or base_logits.ndim != 1 or len(features) != len(base_logits):
            raise ValueError("Expected features [N,D] and base_logits [N]")
        return base_logits + self.residual(features)


def selection_ranking_loss(
    final_logits: torch.Tensor,
    base_logits: torch.Tensor,
    targets: torch.Tensor,
    tiers: torch.Tensor,
    weights: torch.Tensor,
    *,
    margin: float = 0.2,
    pairwise_weight: float = 0.5,
    preserve_weight: float = 0.0,
    preserve_margin: float = 0.05,
    max_pairs: int = 4096,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Soft Selection BCE plus winner/duplicate/background pairwise ordering."""
    if not (final_logits.ndim == base_logits.ndim == targets.ndim == tiers.ndim == weights.ndim == 1):
        raise ValueError("Expected one-dimensional logits, targets, tiers and weights")
    if not (len(final_logits) == len(base_logits) == len(targets) == len(tiers) == len(weights)):
        raise ValueError("Selection loss tensors must have the same length")
    bce = F.binary_cross_entropy_with_logits(final_logits, targets, weight=weights)
    positive = torch.where(tiers >= 3)[0]
    negative = torch.where(tiers <= 1)[0]
    pairwise = final_logits.detach().new_zeros(())
    if len(positive) and len(negative) and pairwise_weight > 0:
        pair_count = min(int(max_pairs), int(len(positive) * len(negative)))
        pos_index = positive[torch.randint(len(positive), (pair_count,), device=final_logits.device)]
        neg_index = negative[torch.randint(len(negative), (pair_count,), device=final_logits.device)]
        pair_weights = (weights[pos_index] * weights[neg_index]).sqrt()
        pairwise = (
            F.relu(float(margin) - final_logits[pos_index] + final_logits[neg_index])
            * pair_weights
        ).mean()
    preserve = final_logits.detach().new_zeros(())
    preserve_mask = tiers >= 3
    if preserve_weight > 0 and preserve_mask.any():
        preserve = (
            F.relu(base_logits[preserve_mask] - final_logits[preserve_mask] - float(preserve_margin))
            * weights[preserve_mask]
        ).mean()
    loss = bce + float(pairwise_weight) * pairwise + float(preserve_weight) * preserve
    return loss, {"bce": bce, "pairwise": pairwise, "preserve": preserve}


def hardest_class_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Require the true logit to exceed the strongest wrong class by ``margin``."""
    if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("Expected logits [N,C] and targets [N]")
    if not 0 <= margin:
        raise ValueError("margin must be non-negative")
    true_logits = logits.gather(1, targets[:, None]).squeeze(1)
    wrong_logits = logits.masked_fill(
        F.one_hot(targets, num_classes=logits.shape[1]).bool(), float("-inf")
    )
    return F.relu(float(margin) - true_logits + wrong_logits.max(dim=1).values).mean()


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    label_smoothing: float = 0.05,
    margin: float = 0.2,
    margin_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ce = F.cross_entropy(logits, targets, label_smoothing=float(label_smoothing))
    ranking = hardest_class_margin_loss(logits, targets, margin) \
        if margin_weight > 0 else ce.detach() * 0
    return ce + float(margin_weight) * ranking, {"ce": ce, "margin": ranking}


def balanced_sample_weights(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Give every class equal total sampling mass without changing the loss."""
    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    counts = torch.bincount(targets.cpu(), minlength=num_classes).float()
    if torch.any(counts <= 0):
        missing = torch.where(counts <= 0)[0].tolist()
        raise ValueError(f"Cannot balance classes without samples: {missing}")
    weights = counts.reciprocal()[targets.cpu()]
    return weights / weights.sum()


def confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    flat = targets.to(torch.long).cpu() * num_classes + predictions.to(torch.long).cpu()
    return torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    classes: list[str],
) -> dict:
    predictions = logits.argmax(dim=1)
    matrix = confusion_matrix(targets, predictions, len(classes))
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    true_positive = matrix.diag()
    precision = true_positive.float() / predicted.clamp_min(1)
    recall = true_positive.float() / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    per_class = {
        name: {
            "support": int(support[index]),
            "tp": int(true_positive[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, name in enumerate(classes)
    }
    confusions = []
    for true_index, true_name in enumerate(classes):
        for pred_index, pred_name in enumerate(classes):
            if true_index == pred_index or matrix[true_index, pred_index] == 0:
                continue
            confusions.append({
                "true": true_name,
                "predicted": pred_name,
                "count": int(matrix[true_index, pred_index]),
            })
    confusions.sort(key=lambda row: (-row["count"], row["true"], row["predicted"]))
    return {
        "accuracy": float(true_positive.sum() / matrix.sum().clamp_min(1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "top_confusions": confusions[:30],
    }


def class_count_dict(targets: torch.Tensor, classes: list[str]) -> dict[str, int]:
    counts = Counter(targets.cpu().tolist())
    return {name: int(counts.get(index, 0)) for index, name in enumerate(classes)}
