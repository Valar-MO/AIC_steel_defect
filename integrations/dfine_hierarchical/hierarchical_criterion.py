"""Factorized defectness/localization/category losses for Hierarchical D-FINE."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_iou
from .dfine_criterion import DFINECriterion


@register()
class HierarchicalDFINECriterion(DFINECriterion):
    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(self, matcher, weight_dict, losses, alpha=0.75, gamma=2.0,
                 num_classes=9, reg_max=32, boxes_weight_format=None,
                 share_matched_indices=False, class_weights=None,
                 class_label_smoothing=0.05, class_focal_gamma=1.0):
        super().__init__(matcher, weight_dict, losses, alpha, gamma, num_classes,
                         reg_max, boxes_weight_format, share_matched_indices)
        if class_weights is None:
            class_weights = [1.0] * num_classes
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.numel() != num_classes:
            raise ValueError(f"Expected {num_classes} class weights, got {weights.numel()}")
        self.register_buffer("conditional_class_weights", weights / weights.mean())
        self.class_label_smoothing = class_label_smoothing
        self.class_focal_gamma = class_focal_gamma

    def loss_hierarchical(self, outputs, targets, indices, num_boxes, **kwargs):
        objectness = outputs.get("pred_objectness_logits", outputs["pred_logits"]).float()
        matched = self._get_src_permutation_idx(indices)
        objectness_target = torch.zeros_like(objectness)
        if matched[0].numel():
            predicted_boxes = outputs["pred_boxes"][matched]
            target_boxes = torch.cat([
                target["boxes"][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ])
            iou = torch.diag(box_iou(
                box_cxcywh_to_xyxy(predicted_boxes), box_cxcywh_to_xyxy(target_boxes)
            )[0]).detach().to(objectness.dtype)
            objectness_target[matched] = iou.unsqueeze(-1)
        probability = objectness.sigmoid().detach()
        foreground = objectness_target > 0
        weight = self.alpha * probability.pow(self.gamma) * (~foreground) + objectness_target
        defectness_loss = F.binary_cross_entropy_with_logits(
            objectness, objectness_target, weight=weight, reduction="sum"
        ) / num_boxes

        class_logits = outputs.get("pred_class_logits")
        if class_logits is None or not matched[0].numel():
            class_loss = objectness.sum() * 0
        else:
            labels = torch.cat([
                target["labels"][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ])
            positive_logits = class_logits[matched].float()
            per_item = F.cross_entropy(
                positive_logits, labels, weight=self.conditional_class_weights,
                label_smoothing=self.class_label_smoothing, reduction="none",
            )
            if self.class_focal_gamma > 0:
                true_probability = positive_logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
                per_item = per_item * (1.0 - true_probability).pow(self.class_focal_gamma)
            class_loss = per_item.sum() / num_boxes
        return {"loss_defectness": defectness_loss, "loss_class": class_loss}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == "hierarchical":
            return self.loss_hierarchical(outputs, targets, indices, num_boxes, **kwargs)
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
