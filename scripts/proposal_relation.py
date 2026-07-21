#!/usr/bin/env python3
"""Shared proposal-relation model and cache metadata helpers."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


def proposal_auxiliary(box, width, height, detector_score, objectness_logit, class_logits):
    """Build explicit geometry and baseline-decision features for one proposal."""
    x1, y1, x2, y2 = map(float, box)
    width, height = max(1.0, float(width)), max(1.0, float(height))
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    w, h = max(1e-6, (x2 - x1) / width), max(1e-6, (y2 - y1) / height)
    cx, cy = ((x1 + x2) * 0.5 / width), ((y1 + y2) * 0.5 / height)
    logits = torch.as_tensor(class_logits, dtype=torch.float32)
    probabilities = logits.softmax(dim=0)
    return torch.cat((
        torch.tensor([
            x1 / width, y1 / height, x2 / width, y2 / height,
            cx, cy, math.sqrt(w * h), math.log(w / h) / 5.0,
            _logit(detector_score) / 5.0, float(objectness_logit) / 5.0,
        ], dtype=torch.float32),
        probabilities,
    ))


def load_curated_relation_metadata(cache, data_root: Path, split: str):
    with (data_root / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if len(rows) != len(cache["features"]):
        raise ValueError(f"metadata/cache length mismatch: {len(rows)} != {len(cache['features'])}")
    auxiliary, image_ids, outcomes, references = [], [], [], []
    for index, row in enumerate(rows):
        box = [float(row[key]) for key in ("proposal_x1", "proposal_y1", "proposal_x2", "proposal_y2")]
        auxiliary.append(proposal_auxiliary(
            box, row["image_width"], row["image_height"], cache["detector_scores"][index],
            cache["baseline_objectness_logits"][index], cache["baseline_class_logits"][index],
        ))
        image_ids.append(Path(row["image_id"]).name)
        outcomes.append(row["audit_outcome"])
        references.append(row.get("reference_gt_id", ""))
    return {
        "auxiliary": torch.stack(auxiliary), "image_ids": image_ids,
        "audit_outcomes": outcomes, "reference_gt_ids": references,
    }


def load_raw_relation_metadata(cache, predictions: Path):
    payload = json.loads(predictions.read_text(encoding="utf-8"))
    auxiliary, image_ids = [], []
    for index, (record_index, prediction_index) in enumerate(cache["proposal_locations"].tolist()):
        record = payload["images"][record_index]
        pred = record["predictions"][prediction_index]
        box = pred.get("bbox_xyxy", pred.get("bbox"))
        auxiliary.append(proposal_auxiliary(
            box, record["width"], record["height"], cache["detector_scores"][index],
            cache["baseline_objectness_logits"][index], cache["baseline_class_logits"][index],
        ))
        image_ids.append(Path(str(record["image_id"])).name)
    return {"auxiliary": torch.stack(auxiliary), "image_ids": image_ids, "payload": payload}


def grouped_indices(image_ids):
    groups = defaultdict(list)
    for index, image_id in enumerate(image_ids):
        groups[image_id].append(index)
    return list(groups.values())


def pairwise_geometry(boxes: torch.Tensor) -> torch.Tensor:
    """Return directed pair features [IoU, dx, dy, |dx|, |dy|, log wr, log hr]."""
    left_top = torch.maximum(boxes[:, :, None, :2], boxes[:, None, :, :2])
    right_bottom = torch.minimum(boxes[:, :, None, 2:4], boxes[:, None, :, 2:4])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    wh = (boxes[..., 2:4] - boxes[..., :2]).clamp_min(1e-6)
    area = wh.prod(dim=-1)
    union = area[:, :, None] + area[:, None, :] - intersection
    iou = intersection / union.clamp_min(1e-6)
    centers = (boxes[..., :2] + boxes[..., 2:4]) * 0.5
    delta = centers[:, None, :, :] - centers[:, :, None, :]
    log_ratio = (wh[:, None, :, :] / wh[:, :, None, :]).clamp(1e-4, 1e4).log() / 5.0
    return torch.cat((iou.unsqueeze(-1), delta, delta.abs(), log_ratio), dim=-1)


class GeometryRelationBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True,
        )
        self.geometry_bias = nn.Sequential(
            nn.Linear(7, 32), nn.GELU(), nn.Linear(32, heads),
        )
        self.norm2 = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension), nn.Dropout(dropout),
        )

    def forward(self, tokens, boxes, padding_mask):
        batch, length, _ = tokens.shape
        bias = self.geometry_bias(pairwise_geometry(boxes))
        bias = bias.permute(0, 3, 1, 2).reshape(batch * self.heads, length, length)
        if padding_mask is not None:
            key_mask = padding_mask[:, None, None, :].expand(batch, self.heads, length, length)
            bias = bias.masked_fill(key_mask.reshape_as(bias), -1e4)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=bias,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.feed_forward(self.norm2(tokens))


class ProposalRelationTransformer(nn.Module):
    """Two-layer graph transformer that only predicts residual logit corrections."""

    def __init__(self, feature_dim=1664, auxiliary_dim=19, num_classes=9,
                 dimension=256, heads=8, layers=2, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.auxiliary_dim = auxiliary_dim
        self.num_classes = num_classes
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, dimension),
        )
        self.auxiliary_projection = nn.Sequential(
            nn.LayerNorm(auxiliary_dim), nn.Linear(auxiliary_dim, dimension), nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(dimension)
        self.blocks = nn.ModuleList([
            GeometryRelationBlock(dimension, heads, dropout) for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(dimension)
        self.objectness_residual = nn.Linear(dimension, 1)
        self.class_residual = nn.Linear(dimension, num_classes)
        nn.init.zeros_(self.objectness_residual.weight)
        nn.init.zeros_(self.objectness_residual.bias)
        nn.init.zeros_(self.class_residual.weight)
        nn.init.zeros_(self.class_residual.bias)

    def encode(self, features, auxiliary, padding_mask=None):
        tokens = self.input_norm(
            self.feature_projection(features) + self.auxiliary_projection(auxiliary)
        )
        boxes = auxiliary[..., :4]
        for block in self.blocks:
            tokens = block(tokens, boxes, padding_mask)
        return self.output_norm(tokens)

    def forward(self, features, auxiliary, baseline_objectness, baseline_classes, padding_mask=None):
        tokens = self.encode(features, auxiliary, padding_mask)
        objectness_delta = self.objectness_residual(tokens).squeeze(-1)
        class_delta = self.class_residual(tokens)
        return {
            "objectness_logits": baseline_objectness + objectness_delta,
            "class_logits": baseline_classes + class_delta,
            "objectness_delta": objectness_delta,
            "class_delta": class_delta,
        }


def pad_group_batch(groups, features, auxiliary, baseline_objectness, baseline_classes,
                    objectness_targets=None, class_targets=None, primary_targets=None,
                    duplicate_targets=None, reference_ids=None):
    batch, length = len(groups), max(len(group) for group in groups)
    device, feature_dim = features.device, features.shape[1]
    indices = torch.zeros((batch, length), dtype=torch.long, device=device)
    padding = torch.ones((batch, length), dtype=torch.bool, device=device)
    for row, group in enumerate(groups):
        values = torch.as_tensor(group, dtype=torch.long, device=device)
        indices[row, :len(group)] = values
        padding[row, :len(group)] = False
    output = {
        "indices": indices, "padding_mask": padding,
        "features": features[indices], "auxiliary": auxiliary[indices],
        "baseline_objectness": baseline_objectness[indices],
        "baseline_classes": baseline_classes[indices],
    }
    for name, values in (("objectness_targets", objectness_targets),
                         ("class_targets", class_targets),
                         ("primary_targets", primary_targets),
                         ("duplicate_targets", duplicate_targets),
                         ("reference_ids", reference_ids)):
        if values is not None:
            output[name] = values[indices]
    return output
