"""Long-tail classification and quality losses for the cascade heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ClassBalancedFocalSoftmaxLoss(nn.Module):
    """Mutually exclusive ROI classification with an explicit background class.

    The defect weights are deliberately capped.  They compensate for the
    long tail without removing the gradient that distinguishes a defect from
    the other defects or from background.
    """

    def __init__(self, num_defect_classes: int, defect_counts: list[int] | None = None,
                 gamma: float = 1.0, max_defect_weight: float = 2.5):
        super().__init__()
        counts = torch.tensor(defect_counts or [1] * num_defect_classes, dtype=torch.float32).clamp_min(1)
        reference = counts.median()
        defect_weights = (reference / counts).sqrt().clamp(min=1 / max_defect_weight, max=max_defect_weight)
        self.gamma = gamma
        self.register_buffer("class_weights", torch.cat((defect_weights, torch.ones(1))))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        valid = sample_weights > 0
        if not valid.any():
            return logits.sum() * 0
        logits, targets, sample_weights = logits[valid], targets[valid], sample_weights[valid]
        log_probability = F.log_softmax(logits, dim=1)
        log_pt = log_probability.gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp()
        weights = sample_weights * self.class_weights[targets]
        loss = -(1 - pt).pow(self.gamma) * log_pt * weights
        return loss.sum() / weights.sum().clamp_min(1)


def roi_background_margin_losses(logits: torch.Tensor, labels: torch.Tensor, matched_iou: torch.Tensor,
                                 sample_weights: torch.Tensor, image_indices: torch.Tensor, background_index: int,
                                 class_weights: torch.Tensor, positive_iou: float = .60,
                                 min_visible_fraction: float = .70, positive_margin: float = .40,
                                 class_margin: float = .20, background_margin: float = .20,
                                 hard_background_topk: int = 3) -> dict[str, torch.Tensor]:
    """Margins for final-stage hard positives and genuinely confusing backgrounds.

    The caller supplies only sampled RoIs.  Positives are deliberately limited
    to reliable, sufficiently visible final-stage matches, while backgrounds
    are mined independently inside each tile to avoid easy negatives
    overwhelming the classification boundary.
    """
    zero = logits.sum() * 0
    result = {"keep_margin": zero, "class_margin": zero, "hard_background": zero}
    if not len(logits):
        return result

    positive = ((labels >= 0) & (labels < background_index) & (matched_iou >= positive_iou)
                & (sample_weights >= min_visible_fraction))
    if positive.any():
        positive_indices = torch.where(positive)[0]
        true_logits = logits[positive_indices, labels[positive_indices]]
        background_logits = logits[positive_indices, background_index]
        positive_weights = sample_weights[positive_indices] * class_weights[labels[positive_indices]]

        background_gap = true_logits - background_logits
        hard_keep = background_gap < positive_margin
        if hard_keep.any():
            keep_loss = F.softplus(positive_margin - background_gap[hard_keep]) * positive_weights[hard_keep]
            result["keep_margin"] = keep_loss.sum() / positive_weights[hard_keep].sum().clamp_min(1)

        defect_logits = logits[positive_indices, :background_index].clone()
        defect_logits[torch.arange(len(positive_indices), device=logits.device), labels[positive_indices]] = float("-inf")
        competing_gap = true_logits - defect_logits.max(dim=1).values
        hard_class = competing_gap < class_margin
        if hard_class.any():
            pair_loss = F.softplus(class_margin - competing_gap[hard_class]) * positive_weights[hard_class]
            result["class_margin"] = pair_loss.sum() / positive_weights[hard_class].sum().clamp_min(1)

    # Use true background only.  Stage-3's ordinary negative pool includes
    # near-matches; those should not be trained as texture-background vetoes.
    background = (labels == background_index) & (matched_iou < .10) & (sample_weights > 0)
    if not background.any():
        return result
    hard_indices = []
    for image_index in image_indices[background].unique(sorted=True):
        candidates = torch.where(background & (image_indices == image_index))[0]
        hardness = logits[candidates, :background_index].max(dim=1).values - logits[candidates, background_index]
        hard_indices.append(candidates[hardness.topk(min(hard_background_topk, len(candidates))).indices])
    hard_indices = torch.cat(hard_indices)
    hardness = logits[hard_indices, :background_index].max(dim=1).values - logits[hard_indices, background_index]
    hard_loss = F.softplus(background_margin + hardness) * sample_weights[hard_indices]
    result["hard_background"] = hard_loss.sum() / sample_weights[hard_indices].sum().clamp_min(1)
    return result


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
