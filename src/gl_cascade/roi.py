"""Three-stage Cascade ROI heads with general and shape-preserving evidence."""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from .backbone import group_norm
from .boxes import box_iou, clip_boxes, decode_boxes, encode_boxes
from .losses import ClassBalancedFocalSoftmaxLoss, EQLv2Loss, quality_loss, roi_background_margin_losses


FPN_LEVELS = ("P2", "P3", "P4", "P5", "P6")


def _roi_levels(boxes: torch.Tensor) -> torch.Tensor:
    sizes = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=1).sqrt()
    return (torch.floor(torch.log2(sizes / 56 + 1e-6)).to(torch.long) + 2).clamp(2, 6)


def _rois(boxes: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([torch.cat((box.new_full((len(box), 1), image_index), box), dim=1)
                      for image_index, box in enumerate(boxes) if len(box)], dim=0) if any(len(box) for box in boxes) else boxes[0].new_zeros((0, 5))


class MultiScaleROIExtractor(nn.Module):
    def __init__(self, output_size: tuple[int, int] = (7, 7), sampling_ratio: int = 2):
        super().__init__()
        self.output_size, self.sampling_ratio = output_size, sampling_ratio

    def forward(self, features: OrderedDict[str, torch.Tensor], boxes: list[torch.Tensor]) -> torch.Tensor:
        rois = _rois(boxes)
        if not len(rois):
            channel = next(iter(features.values())).shape[1]
            return next(iter(features.values())).new_zeros((0, channel, *self.output_size))
        levels = _roi_levels(rois[:, 1:])
        result = next(iter(features.values())).new_zeros((len(rois), next(iter(features.values())).shape[1], *self.output_size))
        for level, name in zip(range(2, 7), FPN_LEVELS):
            select = torch.where(levels == level)[0]
            if len(select):
                result[select] = roi_align(features[name], rois[select], self.output_size,
                                           spatial_scale=1 / (2 ** level), sampling_ratio=self.sampling_ratio,
                                           aligned=True)
        return result


class ShapeROIExtractor(nn.Module):
    """Aspect-preserving ROI routes retaining horizontal/vertical layout."""

    def __init__(self, channels: int):
        super().__init__()
        self.horizontal = nn.Sequential(nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels, bias=False),
                                        group_norm(channels), nn.GELU(), nn.Conv2d(channels, channels, 1, bias=False), group_norm(channels), nn.GELU())
        self.vertical = nn.Sequential(nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels, bias=False),
                                      group_norm(channels), nn.GELU(), nn.Conv2d(channels, channels, 1, bias=False), group_norm(channels), nn.GELU())
        self.compact = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                                     group_norm(channels), nn.GELU())

    def forward(self, features: OrderedDict[str, torch.Tensor], boxes: list[torch.Tensor]) -> torch.Tensor:
        rois = _rois(boxes)
        channel = next(iter(features.values())).shape[1]
        if not len(rois):
            return next(iter(features.values())).new_zeros((0, channel))
        ratio = (rois[:, 3] - rois[:, 1]).clamp(min=1) / (rois[:, 4] - rois[:, 2]).clamp(min=1)
        groups = ((ratio >= 2).to(torch.long) - (ratio <= .5).to(torch.long))  # 1=horiz, -1=vertical
        embedding = next(iter(features.values())).new_zeros((len(rois), channel))
        for condition, output_size, branch in ((groups == 1, (7, 28), self.horizontal),
                                               (groups == -1, (28, 7), self.vertical),
                                               (groups == 0, (14, 14), self.compact)):
            selected = torch.where(condition)[0]
            if not len(selected):
                continue
            selected_rois, levels = rois[selected], _roi_levels(rois[selected, 1:])
            pooled = next(iter(features.values())).new_zeros((len(selected), channel, *output_size))
            for level, name in zip(range(2, 7), FPN_LEVELS):
                local = torch.where(levels == level)[0]
                if len(local):
                    pooled[local] = roi_align(features[name], selected_rois[local], output_size,
                                              spatial_scale=1 / (2 ** level), sampling_ratio=2, aligned=True)
            embedding[selected] = branch(pooled).mean(dim=(-2, -1))
        return embedding


class CascadeStage(nn.Module):
    def __init__(self, channels: int, class_output_classes: int):
        super().__init__()
        self.general = MultiScaleROIExtractor((7, 7))
        self.shape = ShapeROIExtractor(channels)
        self.general_projection = nn.Sequential(nn.Flatten(), nn.Linear(channels * 49, 1024), nn.GELU(), nn.Dropout(.1))
        self.fusion = nn.Sequential(nn.Linear(1024 + channels + 4, 1024), nn.GELU(), nn.Dropout(.1))
        self.class_logits = nn.Linear(1024, class_output_classes)
        self.foreground = nn.Linear(1024, 1)
        self.box_delta = nn.Linear(1024, 4)
        self.quality = nn.Linear(1024, 1)
        for layer in (self.class_logits, self.foreground, self.box_delta, self.quality):
            nn.init.normal_(layer.weight, std=.01); nn.init.zeros_(layer.bias)

    def forward(self, features: OrderedDict[str, torch.Tensor], boxes: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        general = self.general(features, boxes)
        shape = self.shape(features, boxes)
        flat_boxes = torch.cat(boxes) if any(len(item) for item in boxes) else general.new_zeros((0, 4))
        geometry = torch.stack((flat_boxes[:, 2] - flat_boxes[:, 0], flat_boxes[:, 3] - flat_boxes[:, 1],
                                (flat_boxes[:, :2] + flat_boxes[:, 2:])[:, 0] / 2,
                                (flat_boxes[:, :2] + flat_boxes[:, 2:])[:, 1] / 2), dim=1) if len(flat_boxes) else general.new_zeros((0, 4))
        geometry = torch.log1p(geometry.clamp(min=0))
        embedding = self.fusion(torch.cat((self.general_projection(general), shape, geometry), dim=1))
        return {"embedding": embedding, "class_logits": self.class_logits(embedding), "foreground": self.foreground(embedding).squeeze(1),
                "box_delta": self.box_delta(embedding), "quality": self.quality(embedding).squeeze(1)}


class QilieJiebaResidual(nn.Module):
    def __init__(self, feature_dim: int, qilie_index: int, jieba_index: int):
        super().__init__()
        self.qilie_index, self.jieba_index = qilie_index, jieba_index
        self.residual = nn.Sequential(nn.Linear(feature_dim, 128), nn.GELU(), nn.Linear(128, 2))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)

    def forward(self, logits: torch.Tensor, embedding: torch.Tensor, force_pair: torch.Tensor | None = None) -> torch.Tensor:
        if not len(logits):
            return logits
        top3 = logits.topk(min(3, logits.shape[1]), dim=1).indices
        triggered = (top3 == self.qilie_index).any(dim=1) | (top3 == self.jieba_index).any(dim=1)
        if force_pair is not None:
            triggered |= force_pair
        result = logits.clone()
        result[triggered, self.qilie_index] += self.residual(embedding[triggered])[:, 0]
        result[triggered, self.jieba_index] += self.residual(embedding[triggered])[:, 1]
        return result


class CascadeROIHeads(nn.Module):
    thresholds = (.50, .55, .60)

    def __init__(self, num_classes: int, channels: int = 256, samples_per_image: int = 512, positive_fraction: float = .25,
                 classifier_mode: str = "softmax_background", defect_class_counts: list[int] | None = None,
                 r3a_enabled: bool = False, r3a_keep_weight: float = .35, r3a_class_weight: float = .15,
                 r3a_hard_background_weight: float = .10, r3a_positive_margin: float = .40,
                 r3a_class_margin: float = .20, r3a_background_margin: float = .20, r3a_hard_background_topk: int = 3):
        super().__init__()
        if classifier_mode not in {"legacy_eql", "softmax_background"}:
            raise ValueError(f"Unsupported classifier mode: {classifier_mode}")
        self.num_classes, self.samples_per_image, self.positive_fraction = num_classes, samples_per_image, positive_fraction
        self.classifier_mode, self.background_index = classifier_mode, num_classes
        self.r3a_enabled = r3a_enabled
        self.r3a_keep_weight, self.r3a_class_weight = r3a_keep_weight, r3a_class_weight
        self.r3a_hard_background_weight = r3a_hard_background_weight
        self.r3a_positive_margin, self.r3a_class_margin = r3a_positive_margin, r3a_class_margin
        self.r3a_background_margin, self.r3a_hard_background_topk = r3a_background_margin, r3a_hard_background_topk
        output_classes = num_classes if classifier_mode == "legacy_eql" else num_classes + 1
        self.stages = nn.ModuleList([CascadeStage(channels, output_classes) for _ in self.thresholds])
        if classifier_mode == "legacy_eql":
            self.eql = EQLv2Loss(num_classes)
            self.pair = QilieJiebaResidual(1024, qilie_index=2, jieba_index=0)
        else:
            self.class_loss = ClassBalancedFocalSoftmaxLoss(num_classes, defect_class_counts)

    def _assign(self, boxes: torch.Tensor, target: dict[str, torch.Tensor], threshold: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not len(target["boxes"]):
            return (torch.full((len(boxes),), -1, dtype=torch.long, device=boxes.device), torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device),
                    torch.zeros(len(boxes), device=boxes.device), torch.ones(len(boxes), device=boxes.device))
        ious = box_iou(boxes, target["boxes"])
        best_iou, matched = ious.max(dim=1)
        positive = best_iou >= threshold
        valid = positive | (best_iou < threshold - .1)
        ignored_boxes = target.get("ignore_boxes")
        if ignored_boxes is not None and len(ignored_boxes):
            ignored = box_iou(boxes, ignored_boxes).max(dim=1).values >= .5
            valid[ignored & ~positive] = False
        visible = torch.ones(len(boxes), device=boxes.device)
        boundary_weight = target.get("boundary_weight", target["visible_fraction"])
        visible[positive] = boundary_weight[matched[positive]]
        return matched, positive, best_iou, visible * valid

    def _sample(self, positive: torch.Tensor, valid_weight: torch.Tensor) -> torch.Tensor:
        positive_indices, negative_indices = torch.where(positive & (valid_weight > 0))[0], torch.where((~positive) & (valid_weight > 0))[0]
        count_pos = min(int(self.samples_per_image * self.positive_fraction), len(positive_indices))
        count_neg = min(self.samples_per_image - count_pos, len(negative_indices))
        return torch.cat((positive_indices[torch.randperm(len(positive_indices), device=positive.device)[:count_pos]],
                          negative_indices[torch.randperm(len(negative_indices), device=positive.device)[:count_neg]]))

    def forward(self, features: OrderedDict[str, torch.Tensor], proposals: list[torch.Tensor], image_size: tuple[int, int],
                targets: list[dict[str, torch.Tensor]] | None = None) -> dict:
        # Training-only GT injection gives every ROI stage a positive example
        # while the RPN is still learning to propose tiny defects.  Inference
        # remains strictly proposal-driven.
        boxes = ([torch.cat((proposal, target["boxes"].detach()), dim=0) for proposal, target in zip(proposals, targets)]
                 if self.training and targets is not None else proposals)
        losses: dict[str, torch.Tensor] = {}
        final = None
        for stage_index, (threshold, stage) in enumerate(zip(self.thresholds, self.stages), start=1):
            train_boxes, assignments = boxes, None
            if self.training and targets is not None:
                sampled_boxes, assignments = [], []
                for item, target in zip(boxes, targets):
                    matched, positive, iou, weight = self._assign(item, target, threshold)
                    keep = self._sample(positive, weight)
                    sampled_boxes.append(item[keep])
                    assignments.append((matched[keep], positive[keep], iou[keep], weight[keep]))
                train_boxes = sampled_boxes
            output = stage(features, train_boxes)
            force_pair = None
            if self.classifier_mode == "legacy_eql" and assignments is not None:
                label_parts = []
                for target, (matched, positive, _, _) in zip(targets, assignments):
                    part = torch.full_like(matched, -2)
                    if len(target["labels"]):
                        part[positive] = target["labels"][matched[positive]]
                    label_parts.append(part)
                labels = torch.cat(label_parts)
                force_pair = (labels == 2) | (labels == 0)
            if self.classifier_mode == "legacy_eql":
                output["class_logits"] = self.pair(output["class_logits"], output["embedding"], force_pair)
            refined = self._split_and_refine(train_boxes, output["box_delta"], image_size)
            if assignments is not None:
                self._losses(losses, stage_index, output, train_boxes, targets, assignments)
            boxes, final = refined, output
        result = {"boxes": boxes, "final": final}
        if losses:
            result["losses"] = losses
        return result

    def _split_and_refine(self, boxes: list[torch.Tensor], delta: torch.Tensor, image_size: tuple[int, int]) -> list[torch.Tensor]:
        result, offset = [], 0
        for item in boxes:
            item_delta = delta[offset:offset + len(item)]
            # Cascade targets treat the previous stage's proposal geometry as
            # fixed; later losses must not backpropagate through it.
            result.append(clip_boxes(decode_boxes(item, item_delta), *image_size).detach())
            offset += len(item)
        return result

    def _losses(self, losses: dict[str, torch.Tensor], stage: int, output: dict[str, torch.Tensor], sampled_boxes: list[torch.Tensor],
                targets: list[dict[str, torch.Tensor]], assignments: list[tuple[torch.Tensor, ...]]) -> None:
        labels, foreground, target_boxes, ious, weights, image_indices = [], [], [], [], [], []
        for image_index, (target, (matched, positive, matched_iou, weight)) in enumerate(zip(targets, assignments)):
            class_labels = torch.full_like(matched, -2 if self.classifier_mode == "legacy_eql" else self.background_index)
            if len(target["labels"]):
                class_labels[positive] = target["labels"][matched[positive]]
            labels.append(class_labels)
            foreground.append(positive.to(torch.float32))
            target_boxes.append(target["boxes"][matched.clamp(min=0)] if len(target["boxes"]) else target["boxes"].new_zeros((len(matched), 4)))
            ious.append(matched_iou); weights.append(weight)
            image_indices.append(torch.full((len(matched),), image_index, dtype=torch.long, device=matched.device))
        labels, foreground, target_boxes, ious, weights, image_indices = map(torch.cat, (labels, foreground, target_boxes, ious, weights, image_indices))
        if self.classifier_mode == "legacy_eql":
            losses[f"roi{stage}_class"] = self.eql(output["class_logits"], labels, weights)
        else:
            losses[f"roi{stage}_class"] = self.class_loss(output["class_logits"], labels, weights)
            if self.r3a_enabled and stage == len(self.thresholds):
                margins = roi_background_margin_losses(
                    output["class_logits"], labels, ious, weights, image_indices, self.background_index,
                    self.class_loss.class_weights, positive_iou=self.thresholds[-1],
                    positive_margin=self.r3a_positive_margin, class_margin=self.r3a_class_margin,
                    background_margin=self.r3a_background_margin, hard_background_topk=self.r3a_hard_background_topk)
                losses[f"roi{stage}_keep_margin"] = self.r3a_keep_weight * margins["keep_margin"]
                losses[f"roi{stage}_class_margin"] = self.r3a_class_weight * margins["class_margin"]
                losses[f"roi{stage}_hard_background"] = self.r3a_hard_background_weight * margins["hard_background"]
        foreground_loss = F.binary_cross_entropy_with_logits(output["foreground"], foreground, reduction="none").mul(weights).sum() / weights.sum().clamp(min=1)
        losses[f"roi{stage}_foreground"] = foreground_loss if self.classifier_mode == "legacy_eql" else .25 * foreground_loss
        positive = foreground.bool()
        references = torch.cat(sampled_boxes)
        if positive.any():
            encoded = encode_boxes(references[positive], target_boxes[positive])
            losses[f"roi{stage}_box"] = F.smooth_l1_loss(output["box_delta"][positive], encoded, reduction="none").sum(dim=1).mul(
                weights[positive]).sum() / weights[positive].sum().clamp(min=1)
            # Quality must describe the *refined* cascade box, not the IoU of
            # the proposal before this stage has corrected it.
            quality_target = ious.clone()
            with torch.no_grad():
                quality_target[positive] = box_iou(decode_boxes(references[positive], output["box_delta"][positive]),
                                                    target_boxes[positive]).diag()
        else:
            losses[f"roi{stage}_box"] = output["box_delta"].sum() * 0
            quality_target = ious
        losses[f"roi{stage}_quality"] = quality_loss(output["quality"], quality_target, positive, weights)
