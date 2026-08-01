"""Shared model and dataset for D-FINE proposal crop verification."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
META_FEATURE_NAMES = (
    "det_logit", "box_width", "box_height", "sqrt_area", "log_area",
    "log_aspect", "center_x", "center_y", "source_whole", "source_tile",
    "touches_internal_border",
)
META_DIM = len(META_FEATURE_NAMES)


def expanded_crop_box(box, width: int, height: int, expand: float, min_size: int):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid proposal: {box}")
    side = max((x2 - x1) * expand, (y2 - y1) * expand, float(min_size))
    crop_w, crop_h = min(side, float(width)), min(side, float(height))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - crop_w / 2, 0.0), width - crop_w)
    top = min(max(cy - crop_h / 2, 0.0), height - crop_h)
    return (int(math.floor(left)), int(math.floor(top)),
            int(math.ceil(left + crop_w)), int(math.ceil(top + crop_h)))


def metadata_features(*, score: float, box, width: int, height: int,
                      source: str, touches_internal_border: bool) -> torch.Tensor:
    x1, y1, x2, y2 = map(float, box)
    w = max(1e-6, (x2 - x1) / width)
    h = max(1e-6, (y2 - y1) / height)
    area = max(1e-8, w * h)
    score = min(max(float(score), 1e-5), 1 - 1e-5)
    source = source.lower()
    values = [
        math.log(score / (1 - score)) / 5.0, w, h, math.sqrt(area),
        math.log(area) / 10.0, math.log(w / h) / 5.0,
        ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
        float(source == "whole"), float(source == "tile"), float(touches_internal_border),
    ]
    return torch.tensor(values, dtype=torch.float32)


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(image.height, image.width, 3).permute(2, 0, 1).float().div_(255.0)
    return (data - IMAGENET_MEAN) / IMAGENET_STD


class DFineCropDataset(Dataset):
    def __init__(self, root: Path, split: str, *, image_size: int = 384,
                 tight_expand: float = 1.0, context_expand: float = 1.8,
                 min_crop_size: int = 128, train: bool = False,
                 max_samples: int | None = None, seed: int = 42):
        with (root / "metadata.csv").open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if max_samples is not None:
            random.Random(seed).shuffle(rows); rows = rows[:max_samples]
        if not rows:
            raise ValueError(f"No metadata rows for split={split}")
        self.rows, self.train = rows, train
        self.image_size, self.tight_expand = image_size, tight_expand
        self.context_expand, self.min_crop_size = context_expand, min_crop_size
        self.obj_targets = [int(r["objectness"]) for r in rows]
        self.class_targets = [int(r["class_id"]) for r in rows]

    def __len__(self):
        return len(self.rows)

    def _augment_pair(self, tight: Image.Image, context: Image.Image):
        if not self.train:
            return tight, context
        if random.random() < 0.5:
            tight, context = ImageOps.mirror(tight), ImageOps.mirror(context)
        if random.random() < 0.10:
            tight, context = ImageOps.flip(tight), ImageOps.flip(context)
        brightness = random.uniform(0.85, 1.15) if random.random() < 0.30 else 1.0
        contrast = random.uniform(0.85, 1.20) if random.random() < 0.30 else 1.0
        return tuple(ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(brightness)).enhance(contrast)
                     for im in (tight, context))

    def __getitem__(self, index):
        r = self.rows[index]
        box = [float(r[k]) for k in ("proposal_x1", "proposal_y1", "proposal_x2", "proposal_y2")]
        with Image.open(r["source_image"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            tight = image.crop(expanded_crop_box(box, image.width, image.height,
                                                 self.tight_expand, self.min_crop_size))
            context = image.crop(expanded_crop_box(box, image.width, image.height,
                                                   self.context_expand, self.min_crop_size))
        tight, context = self._augment_pair(tight, context)
        meta = metadata_features(score=float(r["det_score"]), box=box,
                                 width=int(r["image_width"]), height=int(r["image_height"]),
                                 source=r.get("source", "unknown"),
                                 touches_internal_border=bool(int(r.get("touches_internal_border", 0))))
        return (image_to_tensor(tight, self.image_size), image_to_tensor(context, self.image_size), meta,
                torch.tensor(self.obj_targets[index], dtype=torch.float32),
                torch.tensor(self.class_targets[index], dtype=torch.long))


class TokenAwareCropEncoder(nn.Module):
    """Keep local patch evidence instead of reducing a crop to its CLS token."""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        attention_hidden = max(64, hidden_size // 4)
        self.patch_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(attention_hidden, 1),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        cls = tokens[:, 0]
        patches = self.patch_norm(tokens[:, 1:])
        weights = self.attention(patches).squeeze(-1).softmax(dim=1)
        attended = torch.sum(patches * weights.unsqueeze(-1), dim=1)
        # RMS pooling preserves weak high-energy texture responses that mean pooling can erase.
        texture = patches.float().square().mean(dim=1).clamp_min(1e-8).sqrt().to(patches.dtype)
        return self.projection(torch.cat([cls, attended, texture], dim=1))


class TokenAwarePairFusion(nn.Module):
    """Fuse tight detail and wider context with a learned per-channel gate."""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size * 2), nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid(),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 5),
            nn.Linear(hidden_size * 5, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
        )

    def forward(self, tight: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([tight, context], dim=1))
        mixed = gate * tight + (1.0 - gate) * context
        pair = torch.cat([tight, context, torch.abs(tight - context), tight * context, mixed], dim=1)
        return self.projection(pair)


class SpatialResidualAdapter(nn.Module):
    """A small spatial/texture branch whose zero-init output preserves the legacy model."""

    def __init__(self, hidden_size: int, adapter_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.down = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, adapter_dim), nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(adapter_dim, adapter_dim, 3, padding=1, groups=adapter_dim),
            nn.GELU(), nn.Dropout2d(dropout),
        )
        per_crop_dim = adapter_dim * 5  # 2x2 spatial average plus global RMS texture.
        self.projection = nn.Sequential(
            nn.LayerNorm(per_crop_dim * 3),
            nn.Linear(per_crop_dim * 3, hidden_size * 2),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        patches = self.down(tokens[:, 1:])
        patch_count = patches.shape[1]
        side = int(math.isqrt(patch_count))
        if side * side != patch_count:
            raise ValueError(f"Expected a square DINO patch grid, got {patch_count} tokens")
        grid = patches.transpose(1, 2).reshape(-1, self.adapter_dim, side, side)
        grid = grid + self.local(grid)
        spatial = torch.nn.functional.adaptive_avg_pool2d(grid, (2, 2)).flatten(1)
        texture = grid.float().square().mean(dim=(2, 3)).clamp_min(1e-8).sqrt().to(grid.dtype)
        return torch.cat([spatial, texture], dim=1)

    def forward(self, tight_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        tight = self._pool(tight_tokens)
        context = self._pool(context_tokens)
        return self.projection(torch.cat([tight, context, torch.abs(tight - context)], dim=1))


class DFineDualHeadClassifier(nn.Module):
    def __init__(self, weights: Path, num_classes: int = 9, metadata_hidden: int = 128,
                 dropout: float = 0.1, architecture: str = "legacy", adapter_dim: int = 128):
        super().__init__()
        from transformers import Dinov2Model
        if architecture not in {"legacy", "token_fusion_v2", "residual_token_v21"}:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.architecture = architecture
        self.backbone = Dinov2Model.from_pretrained(str(weights), local_files_only=True)
        hidden_size = int(self.backbone.config.hidden_size)
        if architecture == "token_fusion_v2":
            self.crop_encoder = TokenAwareCropEncoder(hidden_size, dropout)
            self.pair_fusion = TokenAwarePairFusion(hidden_size, dropout)
            visual_dim = hidden_size
        else:
            visual_dim = hidden_size * 2
            if architecture == "residual_token_v21":
                self.spatial_adapter = SpatialResidualAdapter(hidden_size, adapter_dim, dropout)
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(META_DIM), nn.Linear(META_DIM, metadata_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(metadata_hidden, metadata_hidden), nn.GELU(),
        )
        fused_dim = visual_dim + metadata_hidden
        self.dropout = nn.Dropout(dropout)
        if architecture == "token_fusion_v2":
            head_hidden = max(256, hidden_size // 2)
            self.objectness = nn.Sequential(
                nn.LayerNorm(fused_dim), nn.Linear(fused_dim, head_hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(head_hidden, 1),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(fused_dim), nn.Linear(fused_dim, head_hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(head_hidden, num_classes),
            )
        else:
            self.objectness = nn.Linear(fused_dim, 1)
            self.classifier = nn.Linear(fused_dim, num_classes)

    def set_trainable_blocks(self, count: int):
        layers = self.backbone.encoder.layer
        if not 0 <= count <= len(layers):
            raise ValueError(f"unfreeze count must be in [0,{len(layers)}]")
        self.backbone.requires_grad_(False)
        for layer in layers[-count:] if count else []:
            layer.requires_grad_(True)
        if count and hasattr(self.backbone, "layernorm"):
            self.backbone.layernorm.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, tight, context, metadata):
        both = torch.cat([tight, context], 0)
        tokens = self.backbone(pixel_values=both, return_dict=True).last_hidden_state
        if self.architecture == "token_fusion_v2":
            tight_tokens, context_tokens = tokens.chunk(2, 0)
            tight_feature = self.crop_encoder(tight_tokens)
            context_feature = self.crop_encoder(context_tokens)
            visual = self.pair_fusion(tight_feature, context_feature)
        else:
            tight_tokens, context_tokens = tokens.chunk(2, 0)
            visual = torch.cat([tight_tokens[:, 0], context_tokens[:, 0]], dim=1)
        metadata_feature = self.metadata_encoder(metadata)
        base_fused = self.dropout(torch.cat([visual, metadata_feature], 1))
        if self.architecture == "residual_token_v21":
            residual = self.spatial_adapter(tight_tokens, context_tokens)
            enhanced_fused = torch.cat(
                [base_fused[:, :visual.shape[1]] + residual, base_fused[:, visual.shape[1]:]], dim=1)
            return {
                "objectness_logits": self.objectness(enhanced_fused).squeeze(1),
                "class_logits": self.classifier(enhanced_fused),
                "base_objectness_logits": self.objectness(base_fused).squeeze(1),
                "base_class_logits": self.classifier(base_fused),
                "residual": residual,
                "fused_features": enhanced_fused,
            }
        return {"objectness_logits": self.objectness(base_fused).squeeze(1),
                "class_logits": self.classifier(base_fused),
                "fused_features": base_fused}
