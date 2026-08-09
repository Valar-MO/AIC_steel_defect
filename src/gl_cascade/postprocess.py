"""Class-wise Soft-NMS that ranks by a separate ranking logit."""
from __future__ import annotations

import torch

from .boxes import box_iou


def classwise_soft_nms(boxes: torch.Tensor, labels: torch.Tensor, detection_scores: torch.Tensor, ranking_logits: torch.Tensor,
                        iou_threshold: float = .80, sigma: float = .5, min_score: float = .001,
                        class_scores: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Keep calibrated class scores separate while decaying selection scores.

    ``ranking_logits`` decides which overlapping candidate wins.  The Gaussian
    Soft-NMS decay is then applied to ``selection_scores`` so final thresholding
    actually suppresses duplicate candidates.  ``class_scores`` are retained
    for calibration audits and are never used as the merge survival score.
    """
    class_scores = detection_scores if class_scores is None else class_scores
    keep_boxes, keep_labels, keep_selection, keep_class, keep_ranking = [], [], [], [], []
    for label in labels.unique(sorted=True):
        indices = torch.where(labels == label)[0]
        local_boxes = boxes[indices]
        local_class = class_scores[indices]
        local_selection = detection_scores[indices].clone()
        local_ranking = ranking_logits[indices].clone()
        while len(local_boxes):
            winner = local_ranking.argmax()
            keep_boxes.append(local_boxes[winner]); keep_labels.append(label)
            keep_selection.append(local_selection[winner]); keep_class.append(local_class[winner]); keep_ranking.append(local_ranking[winner])
            if len(local_boxes) == 1:
                break
            rest = torch.arange(len(local_boxes), device=boxes.device) != winner
            ious = box_iou(local_boxes[winner:winner + 1], local_boxes[rest]).squeeze(0)
            decay = torch.where(ious > iou_threshold, torch.exp(-(ious * ious) / sigma), torch.ones_like(ious))
            local_boxes = local_boxes[rest]
            local_class = local_class[rest]
            local_selection = local_selection[rest] * decay
            local_ranking = local_ranking[rest] + decay.log()
            valid = local_selection >= min_score
            local_boxes, local_class = local_boxes[valid], local_class[valid]
            local_selection, local_ranking = local_selection[valid], local_ranking[valid]
    if not keep_boxes:
        return {"boxes": boxes.new_zeros((0, 4)), "labels": labels.new_zeros((0,), dtype=torch.long),
                "scores": detection_scores.new_zeros((0,)), "class_scores": detection_scores.new_zeros((0,)),
                "ranking_logits": ranking_logits.new_zeros((0,))}
    order = torch.stack(keep_ranking).argsort(descending=True)
    return {"boxes": torch.stack(keep_boxes)[order], "labels": torch.stack(keep_labels)[order],
            "scores": torch.stack(keep_selection)[order], "class_scores": torch.stack(keep_class)[order],
            "ranking_logits": torch.stack(keep_ranking)[order]}
