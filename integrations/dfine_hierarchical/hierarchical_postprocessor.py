"""Single-query hierarchical postprocessing without class-wise query duplication."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from ...core import register


@register()
class HierarchicalDFINEPostProcessor(nn.Module):
    __share__ = ["num_classes", "num_top_queries", "remap_mscoco_category"]

    def __init__(self, num_classes=9, num_top_queries=300, remap_mscoco_category=False,
                 score_mode="objectness"):
        super().__init__()
        if score_mode not in {"objectness", "decoupled", "selection_gated", "joint"}:
            raise ValueError(score_mode)
        self.num_classes = int(num_classes)
        self.num_top_queries = num_top_queries
        self.remap_mscoco_category = remap_mscoco_category
        self.score_mode = score_mode
        self.deploy_mode = False

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        selection = outputs["pred_objectness_logits"].sigmoid().squeeze(-1)
        quality_logits = outputs.get("pred_quality_logits")
        defectness_logits = outputs.get("pred_defectness_logits")
        if self.score_mode == "selection_gated":
            if quality_logits is None or defectness_logits is None:
                raise KeyError(
                    "selection_gated score_mode requires defectness and quality logits"
                )
            objectness = (
                selection
                * defectness_logits.sigmoid().squeeze(-1)
                * quality_logits.sigmoid().squeeze(-1)
            )
        elif self.score_mode == "decoupled":
            if quality_logits is None:
                raise KeyError("decoupled score_mode requires pred_quality_logits")
            objectness = selection * quality_logits.sigmoid().squeeze(-1)
        else:
            objectness = selection
        class_probability = F.softmax(outputs["pred_class_logits"], dim=-1)
        class_confidence, labels = class_probability.max(dim=-1)
        scores = (
            objectness * class_confidence
            if self.score_mode == "joint" else objectness
        )
        boxes = torchvision.ops.box_convert(outputs["pred_boxes"], in_fmt="cxcywh", out_fmt="xyxy")
        boxes = boxes * orig_target_sizes.repeat(1, 2).unsqueeze(1)
        if scores.shape[1] > self.num_top_queries:
            scores, indices = scores.topk(self.num_top_queries, dim=1)
            labels = labels.gather(1, indices)
            boxes = boxes.gather(1, indices.unsqueeze(-1).expand(-1, -1, 4))
        if self.deploy_mode:
            return labels, boxes, scores
        return [dict(labels=label, boxes=box, scores=score)
                for label, box, score in zip(labels, boxes, scores)]

    def deploy(self):
        self.eval(); self.deploy_mode = True
        return self
