#!/usr/bin/env python3
"""Shared components for frozen-backbone hierarchical D-FINE ROI refinement.

The first detector remains the source of proposals and global priors.  This
module only reads its true HGNetv2 P2/P3 maps at selected candidate boxes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def expand_and_clip(boxes: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand normalized xyxy boxes inside a view and report clipping."""
    center_size = xyxy_to_cxcywh(boxes)
    center_size[..., 2:] *= ratio
    expanded = cxcywh_to_xyxy(center_size)
    clipped = expanded.clamp(0.0, 1.0)
    was_clipped = (clipped - expanded).abs().amax(dim=-1) > 1e-6
    return clipped, was_clipped


def make_rois(boxes_xyxy: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert BxQ normalized boxes to torchvision ROIAlign coordinates."""
    batch, queries = boxes_xyxy.shape[:2]
    scale = boxes_xyxy.new_tensor([image_size, image_size, image_size, image_size])
    boxes = boxes_xyxy * scale
    indices = torch.arange(batch, device=boxes.device, dtype=boxes.dtype)
    indices = indices[:, None, None].expand(batch, queries, 1)
    return torch.cat((indices, boxes), dim=-1).reshape(-1, 5)


def extract_candidate_rois(
    p2: torch.Tensor,
    p3: torch.Tensor,
    boxes_cxcywh: torch.Tensor,
    active: torch.Tensor,
    image_size: int,
    context_ratio: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return P2-tight, P3-context ROIs and context-clipped flags for BxQ boxes."""
    if boxes_cxcywh.ndim != 3 or active.shape != boxes_cxcywh.shape[:2]:
        raise ValueError("Expected boxes BxQx4 and active BxQ")
    boxes = cxcywh_to_xyxy(boxes_cxcywh).clamp(0.0, 1.0)
    context, context_clipped = expand_and_clip(boxes, context_ratio)
    # ROIAlign accepts a single (N,5) tensor.  Restrict it before pooling; P2 is
    # intentionally never broadcast to all 300 decoder queries.
    active_flat = active.reshape(-1)
    tight_rois = make_rois(boxes, image_size)[active_flat]
    context_rois = make_rois(context, image_size)[active_flat]
    p2_roi = roi_align(p2, tight_rois, output_size=14, spatial_scale=1 / 4,
                       sampling_ratio=2, aligned=True)
    p3_roi = roi_align(p3, context_rois, output_size=7, spatial_scale=1 / 8,
                       sampling_ratio=2, aligned=True)
    return p2_roi, p3_roi, context_clipped[active]


def geometry_features(
    boxes_cxcywh: torch.Tensor,
    source_id: torch.Tensor,
    touches_border: torch.Tensor,
    context_clipped: torch.Tensor,
    border_distance: torch.Tensor,
) -> torch.Tensor:
    """Small explicit geometry/source input; all values are view-relative."""
    eps = torch.finfo(boxes_cxcywh.dtype).eps
    cx, cy, width, height = boxes_cxcywh.unbind(-1)
    area = (width * height).clamp_min(eps)
    aspect = (width / height.clamp_min(eps)).clamp(1 / 100, 100)
    return torch.stack((
        cx, cy, width, height, area.sqrt(), aspect.log() / math.log(100),
        source_id.to(boxes_cxcywh.dtype), touches_border.to(boxes_cxcywh.dtype),
        context_clipped.to(boxes_cxcywh.dtype), border_distance,
    ), dim=-1)


class ConvPool(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        groups = 8 if hidden % 8 == 0 else 1
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(groups, hidden), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value).flatten(1)


class ROIRefinementHead(nn.Module):
    """Local evidence head with deliberately bounded changes to base detection."""

    def __init__(self, p2_channels: int = 128, p3_channels: int = 512,
                 query_dim: int = 256, hidden: int = 256, classes: int = 9,
                 score_delta_max: float = 0.35, class_delta_max: float = 0.75,
                 box_delta_max: float = 0.20) -> None:
        super().__init__()
        self.score_delta_max = float(score_delta_max)
        self.class_delta_max = float(class_delta_max)
        self.box_delta_max = float(box_delta_max)
        self.p2 = ConvPool(p2_channels, hidden // 2)
        self.p3 = ConvPool(p3_channels, hidden // 2)
        self.query = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, hidden // 2), nn.SiLU())
        self.geometry = nn.Sequential(nn.Linear(10, hidden // 4), nn.SiLU(), nn.Linear(hidden // 4, hidden // 4), nn.SiLU())
        fused_dim = hidden + hidden // 2 + hidden // 4
        self.fuse = nn.Sequential(nn.Linear(fused_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.10),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.veto = nn.Linear(hidden, 1)
        self.score = nn.Linear(hidden, 1)
        self.class_residual = nn.Linear(hidden, classes)
        self.box_delta = nn.Linear(hidden, 4)
        # Start as the identity scoring function; training must earn every change.
        nn.init.zeros_(self.score.weight); nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.class_residual.weight); nn.init.zeros_(self.class_residual.bias)
        nn.init.zeros_(self.box_delta.weight); nn.init.zeros_(self.box_delta.bias)

    def forward(self, p2_roi: torch.Tensor, p3_roi: torch.Tensor, query: torch.Tensor,
                geometry: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = self.fuse(torch.cat((self.p2(p2_roi), self.p3(p3_roi), self.query(query),
                                     self.geometry(geometry)), dim=-1))
        return {
            "veto_logit": self.veto(fused).squeeze(-1),
            "score_delta": self.score_delta_max * torch.tanh(self.score(fused).squeeze(-1)),
            "class_delta": self.class_delta_max * torch.tanh(self.class_residual(fused)),
            "box_delta": self.box_delta_max * torch.tanh(self.box_delta(fused)),
        }


class ROISelectionResidualHead(nn.Module):
    """R2b: one bounded Selection residual, with no competing refinement task."""

    def __init__(self, p2_channels: int = 128, p3_channels: int = 512,
                 query_dim: int = 256, hidden: int = 256,
                 score_delta_max: float = 0.35) -> None:
        super().__init__()
        self.score_delta_max = float(score_delta_max)
        self.p2 = ConvPool(p2_channels, hidden // 2)
        self.p3 = ConvPool(p3_channels, hidden // 2)
        self.query = nn.Sequential(
            nn.LayerNorm(query_dim), nn.Linear(query_dim, hidden // 2), nn.SiLU()
        )
        self.geometry = nn.Sequential(
            nn.Linear(10, hidden // 4), nn.SiLU(),
            nn.Linear(hidden // 4, hidden // 4), nn.SiLU(),
        )
        fused_dim = hidden + hidden // 2 + hidden // 4
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.score = nn.Linear(hidden, 1)
        # The initial model is exactly the base Selection scorer.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, p2_roi: torch.Tensor, p3_roi: torch.Tensor, query: torch.Tensor,
                geometry: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(torch.cat((self.p2(p2_roi), self.p3(p3_roi), self.query(query),
                                     self.geometry(geometry)), dim=-1))
        return self.score_delta_max * torch.tanh(self.score(fused).squeeze(-1))


class DirectionalSpatialEncoder(nn.Module):
    """Directional local evidence that keeps the ROI's row/column layout."""

    def __init__(self, in_channels: int, channels: int, kernels: tuple[tuple[int, int, int], ...]) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False), nn.GroupNorm(groups, channels), nn.SiLU()
        )
        self.branches = nn.ModuleList()
        for kernel_h, kernel_w, dilation in kernels:
            padding = (dilation * (kernel_h - 1) // 2, dilation * (kernel_w - 1) // 2)
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, channels, (kernel_h, kernel_w), padding=padding,
                          dilation=dilation, groups=channels, bias=False),
                nn.GroupNorm(groups, channels), nn.SiLU(),
            ))
        self.branch_count = len(self.branches)
        # mean, max, four horizontal stripes, four vertical stripes
        self.output_dim = channels * 10

    def forward(self, feature: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        value = self.project(feature)
        maps = torch.stack([branch(value) for branch in self.branches], dim=1)
        fused = (maps * gates[:, :, None, None, None]).sum(dim=1)
        avg = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(fused, 1).flatten(1)
        horizontal = F.adaptive_avg_pool2d(fused, (1, 4)).flatten(1)
        vertical = F.adaptive_avg_pool2d(fused, (4, 1)).flatten(1)
        return torch.cat((avg, maximum, horizontal, vertical), dim=1)


class ROIShapeSelectionResidualHead(nn.Module):
    """R2d residual head with gated directional and stripe-preserving ROI readout."""

    def __init__(self, p2_channels: int = 128, p3_channels: int = 512,
                 query_dim: int = 256, hidden: int = 256,
                 score_delta_max: float = 0.35) -> None:
        super().__init__()
        self.score_delta_max = float(score_delta_max)
        # Horizontal, vertical, long/diagonal-like continuity, and local texture.
        self.p2 = DirectionalSpatialEncoder(p2_channels, 48, ((1, 7, 1), (7, 1, 1), (3, 3, 2), (5, 5, 1)))
        self.p3 = DirectionalSpatialEncoder(p3_channels, 32, ((3, 3, 1), (3, 3, 2)))
        self.query = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, 128), nn.SiLU())
        self.geometry = nn.Sequential(nn.Linear(10, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.p2_gate = nn.Sequential(nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, self.p2.branch_count))
        self.p3_gate = nn.Sequential(nn.Linear(192, 64), nn.SiLU(), nn.Linear(64, self.p3.branch_count))
        fused_dim = self.p2.output_dim + self.p3.output_dim + 128 + 64
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(.10),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.score = nn.Linear(hidden, 1)
        nn.init.zeros_(self.score.weight); nn.init.zeros_(self.score.bias)

    def forward(self, p2_roi: torch.Tensor, p3_roi: torch.Tensor, query: torch.Tensor,
                geometry: torch.Tensor) -> torch.Tensor:
        query_value, geometry_value = self.query(query), self.geometry(geometry)
        gate_input = torch.cat((query_value, geometry_value), dim=-1)
        p2_gate = self.p2_gate(gate_input).softmax(dim=-1)
        p3_gate = self.p3_gate(gate_input).softmax(dim=-1)
        fused = self.fuse(torch.cat((self.p2(p2_roi, p2_gate), self.p3(p3_roi, p3_gate),
                                     query_value, geometry_value), dim=-1))
        return self.score_delta_max * torch.tanh(self.score(fused).squeeze(-1))


def apply_box_delta(boxes_cxcywh: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Bounded relative cx/cy shift plus log-scale width/height adjustment."""
    cxcy = boxes_cxcywh[..., :2] + delta[..., :2] * boxes_cxcywh[..., 2:]
    wh = boxes_cxcywh[..., 2:] * delta[..., 2:].exp()
    return torch.cat((cxcy, wh.clamp(1e-4, 1.0)), dim=-1).clamp(0.0, 1.0)


@dataclass(frozen=True)
class RefinementConfig:
    prefilter: float = 0.26
    veto_threshold: float = 0.98
    context_ratio: float = 1.25
    border_factor: float = 0.5
