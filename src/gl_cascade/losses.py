"""Long-tail classification and quality losses for the cascade heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EQLv2Loss(nn.Module):
    """EQLv2 equalization for nine sigmoid class logits.

    The running positive/negative gradient statistics are intentionally shared
    across cascade stages; rare categories therefore receive the same signal
    regardless of which stage produced it.
    """

    def __init__(self, num_classes: int, gamma: float = 12.0, mu: float = .8, alpha: float = 4.0):
        super().__init__()
        self.gamma, self.mu, self.alpha = gamma, mu, alpha
        self.register_buffer("positive_grad", torch.ones(num_classes))
        self.register_buffer("negative_grad", torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        # -1 is an ignored severe-boundary ROI; -2 is a valid background ROI.
        valid = labels >= -2
        if not valid.any():
            return logits.sum() * 0
        logits, labels = logits[valid], labels[valid]
        target = torch.zeros_like(logits)
        positive = labels >= 0
        target[torch.arange(len(labels), device=labels.device), labels.clamp(min=0)] = positive.to(logits.dtype)
        probability = logits.detach().sigmoid()
        ratio = self.positive_grad / self.negative_grad.clamp(min=1e-6)
        negative_weight = 1 / (1 + torch.exp(-self.gamma * (ratio - self.mu)))
        # EQLv2 keeps rare-class positives strong while suppressing their
        # overwhelming negative gradients.  ``alpha`` is the positive boost,
        # not a focal exponent.
        positive_weight = 1 + self.alpha * (1 - negative_weight)
        class_weight = target * positive_weight[None] + (1 - target) * negative_weight[None]
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * class_weight
        if self.training:
            with torch.no_grad():
                gradient = (probability - target).abs()
                self.positive_grad.add_((gradient * target).sum(dim=0))
                self.negative_grad.add_((gradient * (1 - target)).sum(dim=0))
        per_roi = loss.sum(dim=1)
        if weights is not None:
            per_roi = per_roi * weights[valid]
            return per_roi.sum() / weights[valid].sum().clamp(min=1)
        return per_roi.mean()


def quality_loss(quality_logits: torch.Tensor, matched_iou: torch.Tensor, positive: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    if not positive.any():
        return quality_logits.sum() * 0
    return F.binary_cross_entropy_with_logits(quality_logits[positive], matched_iou[positive].detach(), reduction="none").mul(
        weights[positive]).sum() / weights[positive].sum().clamp(min=1)
