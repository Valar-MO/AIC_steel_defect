"""Datasets and models for the fixed-output DINOv2 verifier upper bound."""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import Dataset

from dfine_crop_classifier import (
    TokenAwareCropEncoder,
    TokenAwarePairFusion,
    expanded_crop_box,
)


GEOMETRY_DIM = 9
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_to_uint8_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """Resize identically to the old path but defer normalization to the GPU."""
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.view(image.height, image.width, 3).permute(2, 0, 1)


def geometry_features(cache: dict, index: int) -> torch.Tensor:
    image = cache["images"][int(cache["candidate_image_index"][index])]
    width, height = float(image["width"]), float(image["height"])
    x1, y1, x2, y2 = [float(value) for value in cache["boxes_xyxy"][index]]
    box_w = max(1e-6, (x2 - x1) / width)
    box_h = max(1e-6, (y2 - y1) / height)
    area = max(1e-8, box_w * box_h)
    logit = float(cache["selection_logits"][index])
    class_confidence = float(
        cache["class_logits"][index].float().softmax(dim=0).max()
    )
    return torch.tensor([
        box_w,
        box_h,
        math.sqrt(area),
        math.log(area) / 10.0,
        math.log(box_w / box_h) / 5.0,
        ((x1 + x2) * 0.5) / width,
        ((y1 + y2) * 0.5) / height,
        logit / 5.0,
        class_confidence,
    ], dtype=torch.float32)


class VerifierCandidateDataset(Dataset):
    def __init__(
        self,
        cache_path: Path,
        indices: torch.Tensor | None = None,
        *,
        need_crops: bool,
        image_size: int = 384,
        tight_expand: float = 1.0,
        context_expand: float = 1.8,
        min_crop_size: int = 128,
        train: bool = False,
    ):
        self.cache_path = Path(cache_path)
        self.cache = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        count = len(self.cache["selection_logits"])
        self.indices = (
            indices.long().clone()
            if indices is not None else torch.arange(count, dtype=torch.long)
        )
        if len(self.indices) and (
            int(self.indices.min()) < 0 or int(self.indices.max()) >= count
        ):
            raise IndexError("Candidate index outside cache")
        self.need_crops = bool(need_crops)
        self.image_size = int(image_size)
        self.tight_expand = float(tight_expand)
        self.context_expand = float(context_expand)
        self.min_crop_size = int(min_crop_size)
        self.train_mode = bool(train)
        # Validation candidates are stored image-by-image.  Keep two decoded 4K
        # sources per worker so dozens of boxes from the same steel image do not
        # repeatedly decode the same JPEG.  Training uses random weighted sampling,
        # where a tiny LRU is less useful and can waste worker memory.
        self._decoded_images: OrderedDict[str, Image.Image] = OrderedDict()
        self._decoded_image_limit = 0 if self.train_mode else 2

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, tight: Image.Image, context: Image.Image):
        if not self.train_mode:
            return tight, context
        if random.random() < 0.5:
            tight, context = ImageOps.mirror(tight), ImageOps.mirror(context)
        if random.random() < 0.1:
            tight, context = ImageOps.flip(tight), ImageOps.flip(context)
        brightness = random.uniform(0.85, 1.15) if random.random() < 0.3 else 1.0
        contrast = random.uniform(0.85, 1.2) if random.random() < 0.3 else 1.0
        return tuple(
            ImageEnhance.Contrast(
                ImageEnhance.Brightness(image).enhance(brightness)
            ).enhance(contrast)
            for image in (tight, context)
        )

    def _load_image(self, path: str) -> Image.Image:
        if path in self._decoded_images:
            image = self._decoded_images.pop(path)
            self._decoded_images[path] = image
            return image
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
        if self._decoded_image_limit:
            self._decoded_images[path] = image
            while len(self._decoded_images) > self._decoded_image_limit:
                self._decoded_images.popitem(last=False)
        return image

    def __getitem__(self, row: int) -> dict:
        index = int(self.indices[row])
        result = {
            "candidate_index": torch.tensor(index, dtype=torch.long),
            "query": self.cache["query_features"][index].float(),
            "geometry": geometry_features(self.cache, index),
            "base_logit": self.cache["selection_logits"][index].float(),
            "max_iou": self.cache["max_iou"][index].float(),
            "nearest_label": self.cache["nearest_gt_label"][index].long(),
        }
        if self.need_crops:
            image_row = self.cache["images"][
                int(self.cache["candidate_image_index"][index])
            ]
            box = [float(value) for value in self.cache["boxes_xyxy"][index]]
            # A tiny number of low-score decoder boxes can have x2<x1 or y2<y1
            # after conversion.  They are background candidates, but the verifier
            # still needs a deterministic crop instead of aborting the full cache.
            x1, y1, x2, y2 = box
            if x2 <= x1:
                center = (x1 + x2) * 0.5
                x1, x2 = center - 0.5, center + 0.5
            if y2 <= y1:
                center = (y1 + y2) * 0.5
                y1, y2 = center - 0.5, center + 0.5
            box = [x1, y1, x2, y2]
            image = self._load_image(image_row["source_path"])
            tight = image.crop(expanded_crop_box(
                box, image.width, image.height,
                self.tight_expand, self.min_crop_size,
            ))
            context = image.crop(expanded_crop_box(
                box, image.width, image.height,
                self.context_expand, self.min_crop_size,
            ))
            tight, context = self._augment(tight, context)
            result["tight"] = image_to_uint8_tensor(tight, self.image_size)
            result["context"] = image_to_uint8_tensor(context, self.image_size)
        return result


class QueryVerifier(nn.Module):
    def __init__(self, query_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(query_dim + GEOMETRY_DIM),
            nn.Linear(query_dim + GEOMETRY_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, query, geometry, base_logit, **_kwargs):
        feature = self.encoder(torch.cat([query, geometry], dim=1))
        delta = self.delta(feature).squeeze(1)
        return {
            "logit": base_logit + delta,
            "delta": delta,
            "feature": feature,
        }


class DinoCropEncoder(nn.Module):
    def __init__(self, weights: Path, dropout: float = 0.1):
        super().__init__()
        from transformers import Dinov2Model

        self.backbone = Dinov2Model.from_pretrained(
            str(weights), local_files_only=True
        )
        hidden = int(self.backbone.config.hidden_size)
        self.crop_encoder = TokenAwareCropEncoder(hidden, dropout)
        self.pair_fusion = TokenAwarePairFusion(hidden, dropout)
        self.output_dim = hidden
        self.register_buffer(
            "image_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def set_trainable_blocks(self, count: int) -> None:
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
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, tight: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        pixels = torch.cat([tight, context], dim=0)
        if pixels.dtype == torch.uint8:
            pixels = pixels.float().div_(255.0)
            pixels = (pixels - self.image_mean) / self.image_std
        tokens = self.backbone(
            pixel_values=pixels,
            return_dict=True,
        ).last_hidden_state
        tight_tokens, context_tokens = tokens.chunk(2, dim=0)
        return self.pair_fusion(
            self.crop_encoder(tight_tokens),
            self.crop_encoder(context_tokens),
        )


class DinoVerifier(nn.Module):
    def __init__(
        self,
        weights: Path,
        *,
        query_dim: int = 256,
        hidden_dim: int = 384,
        dropout: float = 0.1,
        use_query: bool,
    ):
        super().__init__()
        self.use_query = bool(use_query)
        self.visual = DinoCropEncoder(weights, dropout)
        input_dim = self.visual.output_dim
        if self.use_query:
            input_dim += query_dim + GEOMETRY_DIM
        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, 1)
        if self.use_query:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def set_trainable_blocks(self, count: int) -> None:
        self.visual.set_trainable_blocks(count)

    def forward(self, tight, context, query, geometry, base_logit, **_kwargs):
        visual = self.visual(tight, context)
        inputs = [visual]
        if self.use_query:
            inputs.extend([query, geometry])
        feature = self.fusion(torch.cat(inputs, dim=1))
        raw = self.output(feature).squeeze(1)
        logit = base_logit + raw if self.use_query else raw
        return {"logit": logit, "delta": raw, "feature": feature}


def build_model(
    mode: str,
    *,
    query_dim: int,
    hidden_dim: int,
    dropout: float,
    dinov2_weights: Path,
) -> nn.Module:
    if mode == "query":
        return QueryVerifier(query_dim, hidden_dim, dropout)
    if mode in {"dino", "query_dino"}:
        return DinoVerifier(
            dinov2_weights,
            query_dim=query_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_query=mode == "query_dino",
        )
    raise ValueError(mode)
