"""Class-agnostic Hungarian assignment for Hierarchical D-FINE."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


@register()
class HierarchicalHungarianMatcher(nn.Module):
    """Match with defectness and boxes only; category logits never move a box match."""

    __share__ = ["use_focal_loss"]

    def __init__(self, weight_dict, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_objectness = weight_dict.get("cost_objectness", weight_dict.get("cost_class", 2))
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets, return_topk=False):
        logits = outputs.get("pred_objectness_logits", outputs["pred_logits"])
        batch_size, queries = logits.shape[:2]
        probability = logits.flatten(0, 1).sigmoid().squeeze(-1)
        boxes = outputs["pred_boxes"].flatten(0, 1)
        target_boxes = torch.cat([target["boxes"] for target in targets])
        target_count = len(target_boxes)
        if self.use_focal_loss:
            negative = (1 - self.alpha) * probability.pow(self.gamma) * (
                -(1 - probability + 1e-8).log()
            )
            positive = self.alpha * (1 - probability).pow(self.gamma) * (
                -(probability + 1e-8).log()
            )
            objectness_cost = (positive - negative).unsqueeze(1).expand(-1, target_count)
        else:
            objectness_cost = -probability.unsqueeze(1).expand(-1, target_count)
        bbox_cost = torch.cdist(boxes, target_boxes, p=1)
        giou_cost = -generalized_box_iou(
            box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target_boxes)
        )
        total = (self.cost_objectness * objectness_cost + self.cost_bbox * bbox_cost
                 + self.cost_giou * giou_cost)
        total = torch.nan_to_num(total.view(batch_size, queries, -1), nan=1.0).cpu()
        sizes = [len(target["boxes"]) for target in targets]
        raw = [linear_sum_assignment(cost[index])
               for index, cost in enumerate(total.split(sizes, dim=-1))]
        indices = [(torch.as_tensor(source, dtype=torch.int64),
                    torch.as_tensor(target, dtype=torch.int64)) for source, target in raw]
        if return_topk:
            if return_topk != 1:
                raise NotImplementedError("Hierarchical matcher currently supports top-k=1 only")
            return {"indices_o2m": indices}
        return {"indices": indices}

