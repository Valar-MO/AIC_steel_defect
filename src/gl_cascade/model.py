"""End-to-end Global-Local Cascade detector."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .backbone import ResNet50DCNBackbone
from .fpn import GlobalLocalFPN
from .postprocess import classwise_soft_nms
from .roi import CascadeROIHeads
from .rpn import ATSSRPN


@dataclass
class GLCascadeOutput:
    losses: dict[str, torch.Tensor] | None = None
    detections: list[dict[str, torch.Tensor]] | None = None
    proposals: list[torch.Tensor] | None = None


class GLCascadeDetector(nn.Module):
    """Shared global/local detector; global features provide context only at P3/P4."""

    def __init__(self, num_classes: int = 9, pretrained_backbone: bool = True, ranking_quality_weight: float = .25):
        super().__init__()
        self.backbone = ResNet50DCNBackbone(pretrained=pretrained_backbone)
        self.fpn = GlobalLocalFPN(self.backbone.out_channels)
        self.rpn = ATSSRPN()
        self.roi_heads = CascadeROIHeads(num_classes=num_classes)
        self.ranking_quality_weight = ranking_quality_weight

    def forward(self, local: torch.Tensor, global_image: torch.Tensor, view_xyxy: torch.Tensor,
                image_sizes: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None) -> GLCascadeOutput:
        local_features = self.backbone.forward_features(local)
        global_features = self.backbone.forward_features(global_image)
        features = self.fpn(local_features, global_features, view_xyxy, image_sizes)
        local_size = tuple(local.shape[-2:])
        rpn = self.rpn(features, local_size, targets if self.training else None)
        roi = self.roi_heads(features, rpn["proposals"], local_size, targets if self.training else None)
        if self.training:
            losses = {**rpn["losses"], **roi["losses"]}
            return GLCascadeOutput(losses=losses, proposals=rpn["proposals"])
        return GLCascadeOutput(detections=self._detections(roi), proposals=rpn["proposals"])

    def _detections(self, roi: dict) -> list[dict[str, torch.Tensor]]:
        final, result, offset = roi["final"], [], 0
        for boxes in roi["boxes"]:
            count = len(boxes)
            logits = final["class_logits"][offset:offset + count]
            foreground = final["foreground"][offset:offset + count]
            quality = final["quality"][offset:offset + count]
            offset += count
            if not count:
                result.append({"boxes": boxes, "labels": boxes.new_zeros((0,), dtype=torch.long),
                               "scores": boxes.new_zeros((0,)), "ranking_logits": boxes.new_zeros((0,))})
                continue
            detection_scores, labels = logits.sigmoid().max(dim=1)
            # Foreground is an explicit veto, never multiplied into the score.
            keep = foreground.sigmoid() >= .05
            detection_scores, labels, boxes, logits, quality = detection_scores[keep], labels[keep], boxes[keep], logits[keep], quality[keep]
            ranking = logits[torch.arange(len(labels), device=labels.device), labels] + self.ranking_quality_weight * torch.tanh(quality)
            result.append(classwise_soft_nms(boxes, labels, detection_scores, ranking))
        return result
