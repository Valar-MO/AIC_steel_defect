"""Read an Ultralytics-style dataset for Transformers object detectors."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, pil_to_tensor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def _indexed_names(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        normalized = {int(key): str(item) for key, item in value.items()}
        if sorted(normalized) != list(range(len(normalized))):
            raise ValueError("Class IDs must be contiguous from zero")
        return [normalized[index] for index in range(len(normalized))]
    raise ValueError("Dataset YAML names must be a list or mapping")


class YoloDetectionDataset(Dataset):
    """YOLO ``class cx cy width height`` labels with square letterboxing."""

    def __init__(
        self,
        dataset_yaml: str | Path,
        split: str,
        image_size: int,
        *,
        horizontal_flip: float = 0.0,
        vertical_flip: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_yaml = Path(dataset_yaml).resolve()
        config = yaml.safe_load(self.dataset_yaml.read_text(encoding="utf-8"))
        root = Path(config.get("path", self.dataset_yaml.parent))
        if not root.is_absolute():
            root = (self.dataset_yaml.parent / root).resolve()
        split_value = config.get(split)
        if not isinstance(split_value, str):
            raise ValueError(f"Dataset YAML has no string split named {split!r}")
        image_source = Path(split_value)
        self.image_source = image_source if image_source.is_absolute() else root / image_source
        self.names = _indexed_names(config.get("names"))
        self.image_size = int(image_size)
        self.horizontal_flip = float(horizontal_flip)
        self.vertical_flip = float(vertical_flip)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= self.horizontal_flip <= 1.0:
            raise ValueError("horizontal_flip must be in [0, 1]")
        if not 0.0 <= self.vertical_flip <= 1.0:
            raise ValueError("vertical_flip must be in [0, 1]")
        if self.brightness < 0 or self.contrast < 0:
            raise ValueError("brightness and contrast must be non-negative")

        if self.image_source.is_dir():
            images = sorted(
                path for path in self.image_source.iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
            )
        elif self.image_source.is_file():
            images = []
            for line in self.image_source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    path = Path(line.strip())
                    images.append(path if path.is_absolute() else root / path)
        else:
            raise FileNotFoundError(f"Image split not found: {self.image_source}")
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            images = images[:limit]
        if not images:
            raise ValueError(f"No images found for split {split!r}: {self.image_source}")
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _label_path(image_path: Path) -> Path:
        parts = list(image_path.parts)
        try:
            index = len(parts) - 1 - parts[::-1].index("images")
        except ValueError as exc:
            raise ValueError(f"Image path has no 'images' component: {image_path}") from exc
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")

    def _read_labels(self, image_path: Path) -> tuple[Tensor, Tensor]:
        label_path = self._label_path(image_path)
        if not label_path.is_file():
            return torch.empty(0, dtype=torch.long), torch.empty((0, 4), dtype=torch.float32)
        classes, boxes = [], []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO row at {label_path}:{line_number}")
            class_id = int(fields[0])
            box = [float(value) for value in fields[1:]]
            if not 0 <= class_id < len(self.names):
                raise ValueError(f"Invalid class {class_id} at {label_path}:{line_number}")
            if not all(0.0 <= value <= 1.0 for value in box):
                raise ValueError(f"Non-normalized box at {label_path}:{line_number}: {box}")
            classes.append(class_id)
            boxes.append(box)
        return torch.tensor(classes, dtype=torch.long), torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        original_width, original_height = image.size
        classes, boxes = self._read_labels(image_path)

        scale = min(self.image_size / original_width, self.image_size / original_height)
        resized_width = max(1, round(original_width * scale))
        resized_height = max(1, round(original_height * scale))
        left = (self.image_size - resized_width) // 2
        top = (self.image_size - resized_height) // 2
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.image_size, self.image_size), color=(114, 114, 114))
        canvas.paste(resized, (left, top))

        if boxes.numel():
            boxes[:, 0] = (boxes[:, 0] * original_width * scale + left) / self.image_size
            boxes[:, 1] = (boxes[:, 1] * original_height * scale + top) / self.image_size
            boxes[:, 2] = boxes[:, 2] * original_width * scale / self.image_size
            boxes[:, 3] = boxes[:, 3] * original_height * scale / self.image_size
            boxes.clamp_(0.0, 1.0)

        if self.horizontal_flip and random.random() < self.horizontal_flip:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if boxes.numel():
                boxes[:, 0] = 1.0 - boxes[:, 0]
        if self.vertical_flip and random.random() < self.vertical_flip:
            canvas = canvas.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if boxes.numel():
                boxes[:, 1] = 1.0 - boxes[:, 1]
        if self.brightness:
            canvas = adjust_brightness(canvas, random.uniform(1 - self.brightness, 1 + self.brightness))
        if self.contrast:
            canvas = adjust_contrast(canvas, random.uniform(1 - self.contrast, 1 + self.contrast))

        pixel_values = pil_to_tensor(canvas).float().div_(255.0)
        pixel_values = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD
        pixel_mask = torch.zeros((self.image_size, self.image_size), dtype=torch.bool)
        pixel_mask[top : top + resized_height, left : left + resized_width] = True
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": {"class_labels": classes, "boxes": boxes},
            "image_path": str(image_path),
        }


def detection_collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in batch]),
        "pixel_mask": torch.stack([sample["pixel_mask"] for sample in batch]),
        "labels": [sample["labels"] for sample in batch],
        "image_paths": [sample["image_path"] for sample in batch],
    }
