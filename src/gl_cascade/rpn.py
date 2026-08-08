"""Class-agnostic ATSS proposal network with data-derived anchor shapes."""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from .backbone import group_norm
from .boxes import box_iou, clip_boxes, decode_boxes, encode_boxes, generalized_box_iou, remove_small_boxes


class AnchorGenerator(nn.Module):
    def __init__(self, strides: tuple[int, ...] = (4, 8, 16, 32, 64),
                 sizes: tuple[int, ...] = (16, 32, 64, 128, 256),
                 ratios: tuple[float, ...] = (.064, .188, .488, 1.224, 3.564)):
        super().__init__()
        if not (len(strides) == len(sizes) and all(value > 0 for value in ratios)):
            raise ValueError("strides/sizes must align and ratios must be positive")
        self.strides, self.sizes, self.ratios = strides, sizes, ratios

    @property
    def num_anchors(self) -> int:
        return len(self.ratios)

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> list[torch.Tensor]:
        anchors = []
        for feature, stride, size in zip(features.values(), self.strides, self.sizes):
            height, width, device, dtype = feature.shape[-2], feature.shape[-1], feature.device, feature.dtype
            y = (torch.arange(height, device=device, dtype=dtype) + .5) * stride
            x = (torch.arange(width, device=device, dtype=dtype) + .5) * stride
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            centres = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
            shapes = []
            for ratio in self.ratios:  # ratio == width / height
                w, h = size * ratio ** .5, size / ratio ** .5
                shapes.append(torch.tensor((w, h), device=device, dtype=dtype))
            shapes = torch.stack(shapes)
            expanded_centres = centres[:, None, :].expand(-1, len(shapes), -1)
            expanded_shapes = shapes[None].expand(len(centres), -1, -1)
            anchors.append(torch.cat((expanded_centres - .5 * expanded_shapes,
                                      expanded_centres + .5 * expanded_shapes), dim=-1).reshape(-1, 4))
        return anchors


class ATSSAssigner:
    def __init__(self, topk: int = 9):
        self.topk = topk

    def __call__(self, anchors_by_level: list[torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = torch.cat(anchors_by_level)
        gt_boxes = targets["boxes"]
        visible = targets.get("boundary_weight", targets["visible_fraction"])
        count = len(anchors)
        matched = torch.full((count,), -1, dtype=torch.long, device=anchors.device)
        positive = torch.zeros(count, dtype=torch.bool, device=anchors.device)
        if not len(gt_boxes):
            return matched, positive, torch.zeros(count, device=anchors.device)
        centres = (anchors[:, :2] + anchors[:, 2:]) / 2
        gt_centres = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        candidate_indices = []
        offset = 0
        for level_anchors in anchors_by_level:
            level_centres = centres[offset:offset + len(level_anchors)]
            distances = ((level_centres[:, None] - gt_centres[None]) ** 2).sum(dim=-1)
            candidate_indices.append(distances.topk(min(self.topk, len(level_anchors)), dim=0, largest=False).indices + offset)
            offset += len(level_anchors)
        candidates = torch.cat(candidate_indices, dim=0)  # [L*topk, num_gt]
        # ``candidates[row, gt]`` is paired with that same GT.  A full IoU
        # matrix here would otherwise mix all GT columns whenever an image
        # contains more than one defect.
        pair_iou_matrix = box_iou(anchors[candidates.reshape(-1)], gt_boxes)
        paired_gt = torch.arange(len(gt_boxes), device=anchors.device).repeat(candidates.shape[0])
        candidate_ious = pair_iou_matrix[torch.arange(candidates.numel(), device=anchors.device), paired_gt].reshape(candidates.shape)
        threshold = candidate_ious.mean(dim=0) + candidate_ious.std(dim=0, unbiased=False)
        candidate_centres = centres[candidates]
        inside = ((candidate_centres[..., 0] >= gt_boxes[None, :, 0]) & (candidate_centres[..., 0] <= gt_boxes[None, :, 2]) &
                  (candidate_centres[..., 1] >= gt_boxes[None, :, 1]) & (candidate_centres[..., 1] <= gt_boxes[None, :, 3]))
        candidate_positive = (candidate_ious >= threshold[None]) & inside
        best_iou = torch.full((count,), -1.0, device=anchors.device)
        for gt_index in range(len(gt_boxes)):
            indices = candidates[candidate_positive[:, gt_index], gt_index]
            if len(indices):
                ious = box_iou(anchors[indices], gt_boxes[gt_index:gt_index + 1]).squeeze(1)
                replace = ious > best_iou[indices]
                selected = indices[replace]
                best_iou[selected], matched[selected], positive[selected] = ious[replace], gt_index, True
        weights = torch.ones(count, device=anchors.device)
        weights[positive] = visible[matched[positive]]
        ignored_boxes = targets.get("ignore_boxes")
        if ignored_boxes is not None and len(ignored_boxes):
            ignored = box_iou(anchors, ignored_boxes).max(dim=1).values >= .5
            weights[ignored & ~positive] = 0
        return matched, positive, weights


class RPNHead(nn.Module):
    def __init__(self, channels: int, anchors_per_location: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), group_norm(channels), nn.GELU())
        self.objectness = nn.Conv2d(channels, anchors_per_location, 1)
        self.box_delta = nn.Conv2d(channels, anchors_per_location * 4, 1)
        for layer in (self.objectness, self.box_delta):
            nn.init.normal_(layer.weight, std=.01); nn.init.zeros_(layer.bias)

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        logits, deltas = [], []
        for feature in features.values():
            hidden = self.shared(feature)
            logits.append(self.objectness(hidden))
            deltas.append(self.box_delta(hidden))
        return logits, deltas


def _flatten_predictions(logits: list[torch.Tensor], deltas: list[torch.Tensor], anchors_per_location: int) -> tuple[torch.Tensor, torch.Tensor]:
    objectness, box_deltas = [], []
    for level_logits, level_deltas in zip(logits, deltas):
        batch, _, height, width = level_logits.shape
        objectness.append(level_logits.permute(0, 2, 3, 1).reshape(batch, height * width * anchors_per_location))
        box_deltas.append(level_deltas.reshape(batch, anchors_per_location, 4, height, width).permute(0, 3, 4, 1, 2).reshape(batch, -1, 4))
    return torch.cat(objectness, dim=1), torch.cat(box_deltas, dim=1)


class ATSSRPN(nn.Module):
    def __init__(self, channels: int = 256, pre_nms_topk: tuple[int, ...] = (1200, 1000, 800, 600, 400),
                 post_nms_topk: int = 1000):
        super().__init__()
        self.anchor_generator = AnchorGenerator()
        self.assigner = ATSSAssigner()
        self.head = RPNHead(channels, self.anchor_generator.num_anchors)
        if len(pre_nms_topk) != 5:
            raise ValueError("pre_nms_topk must provide one quota for P2-P6")
        self.pre_nms_topk, self.post_nms_topk = pre_nms_topk, post_nms_topk

    def forward(self, features: OrderedDict[str, torch.Tensor], image_size: tuple[int, int],
                targets: list[dict[str, torch.Tensor]] | None = None) -> dict:
        logits_by_level, deltas_by_level = self.head(features)
        anchors_by_level = self.anchor_generator(features)
        logits, deltas = _flatten_predictions(logits_by_level, deltas_by_level, self.anchor_generator.num_anchors)
        anchors = torch.cat(anchors_by_level)
        proposals = self._proposals(logits_by_level, deltas_by_level, anchors_by_level, image_size)
        output = {"proposals": proposals, "rpn_logits": logits, "rpn_deltas": deltas, "anchors": anchors}
        if targets is not None:
            output["losses"] = self.losses(logits, deltas, anchors_by_level, targets)
        return output

    def _proposals(self, logits_by_level: list[torch.Tensor], deltas_by_level: list[torch.Tensor], anchors_by_level: list[torch.Tensor],
                   image_size: tuple[int, int]) -> list[torch.Tensor]:
        proposals = []
        batch_size = logits_by_level[0].shape[0]
        for image_index in range(batch_size):
            all_boxes, all_scores = [], []
            for level_logits, level_deltas, level_anchors, quota in zip(logits_by_level, deltas_by_level, anchors_by_level, self.pre_nms_topk):
                scores = level_logits[image_index].permute(1, 2, 0).reshape(-1).detach().sigmoid()
                anchors_per_location = self.anchor_generator.num_anchors
                image_deltas = level_deltas[image_index].reshape(anchors_per_location, 4, *level_logits.shape[-2:]).permute(2, 3, 0, 1).reshape(-1, 4).detach()
                count = min(quota, len(scores))
                top_scores, indices = scores.topk(count)
                all_boxes.append(clip_boxes(decode_boxes(level_anchors[indices], image_deltas[indices]), *image_size))
                all_scores.append(top_scores)
            boxes, top_scores = torch.cat(all_boxes), torch.cat(all_scores)
            keep = remove_small_boxes(boxes, 2)
            boxes, top_scores = boxes[keep], top_scores[keep]
            keep = nms(boxes, top_scores, .7)[:self.post_nms_topk]
            proposals.append(boxes[keep])
        return proposals

    def losses(self, logits: torch.Tensor, deltas: torch.Tensor, anchors_by_level: list[torch.Tensor],
               targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        objectness_losses, box_losses = [], []
        anchors = torch.cat(anchors_by_level)
        for pred_logits, pred_deltas, target in zip(logits, deltas, targets):
            matched, positive, weights = self.assigner(anchors_by_level, target)
            labels = positive.to(pred_logits.dtype)
            bce = F.binary_cross_entropy_with_logits(pred_logits, labels, reduction="none")
            probability = pred_logits.sigmoid()
            focal = ((1 - probability) * labels + probability * (1 - labels)).pow(2) * bce
            objectness_losses.append((focal * weights).sum() / weights.sum().clamp(min=1))
            if positive.any():
                predicted = decode_boxes(anchors[positive], pred_deltas[positive])
                truth = target["boxes"][matched[positive]]
                giou = 1 - generalized_box_iou(predicted, truth).diag()
                box_losses.append((giou * weights[positive]).sum() / weights[positive].sum().clamp(min=1))
            else:
                box_losses.append(pred_deltas.sum() * 0)
        return {"rpn_objectness": torch.stack(objectness_losses).mean(), "rpn_box": torch.stack(box_losses).mean()}
