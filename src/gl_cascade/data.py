"""Official-grid global/local data loader with visibility-aware labels."""
from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode, normalize, pil_to_tensor, resize
import yaml


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    values, last = list(range(0, length - size + 1, stride)), length - size
    if values[-1] != last:
        values.append(last)
    return values


def official_windows(width: int, height: int, tile_size: int = 2048, overlap: int = 512) -> list[tuple[int, int, int, int]]:
    stride = tile_size - overlap
    return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in _starts(height, tile_size, stride) for x in _starts(width, tile_size, stride)]


def _read_voc(path: Path, class_to_id: dict[str, int]) -> tuple[int, int, list[tuple[int, tuple[float, float, float, float]]]]:
    root = ET.parse(path).getroot()
    width, height = int(float(root.findtext("size/width", "0"))), int(float(root.findtext("size/height", "0")))
    boxes = []
    for obj in root.findall("object"):
        name, node = (obj.findtext("name") or "").strip(), obj.find("bndbox")
        if name not in class_to_id or node is None:
            continue
        try:
            x1, y1, x2, y2 = (float(node.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        box = (max(0., min(x1, width)), max(0., min(y1, height)), max(0., min(x2, width)), max(0., min(y2, height)))
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append((class_to_id[name], box))
    return width, height, boxes


def _crop_targets(boxes: list[tuple[int, tuple[float, ...]]], view: tuple[int, int, int, int], output_size: int) -> dict[str, torch.Tensor]:
    targets, labels, visible, boundary_weight, ignored = [], [], [], [], []
    x1, y1, x2, y2 = view
    scale_x, scale_y = output_size / (x2 - x1), output_size / (y2 - y1)
    for label, box in boxes:
        ix1, iy1, ix2, iy2 = max(box[0], x1), max(box[1], y1), min(box[2], x2), min(box[3], y2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        fraction = (ix2 - ix1) * (iy2 - iy1) / ((box[2] - box[0]) * (box[3] - box[1]))
        local = torch.tensor(((ix1 - x1) * scale_x, (iy1 - y1) * scale_y, (ix2 - x1) * scale_x, (iy2 - y1) * scale_y), dtype=torch.float32)
        if fraction >= .5:
            targets.append(local); labels.append(label); visible.append(fraction)
            boundary_weight.append(1.0 if fraction >= .70 else fraction)
        else:
            ignored.append(local)
    return {"boxes": torch.stack(targets) if targets else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long), "visible_fraction": torch.tensor(visible, dtype=torch.float32),
            "boundary_weight": torch.tensor(boundary_weight, dtype=torch.float32),
            "ignore_boxes": torch.stack(ignored) if ignored else torch.zeros((0, 4), dtype=torch.float32)}


@dataclass(frozen=True)
class GridRecord:
    image_path: Path
    width: int
    height: int
    boxes: list[tuple[int, tuple[float, float, float, float]]]
    view: tuple[int, int, int, int]


class OfficialGridGlobalLocalDataset(Dataset):
    """Each row is one fixed official local tile plus its source global view."""

    def __init__(self, manifest: Path, classes: Path, split: str = "train", tile_size: int = 2048,
                 overlap: int = 512, global_input_size: int = 1024, local_input_size: int = 1536):
        names = (yaml.safe_load(classes.read_text(encoding="utf-8")) or {}).get("names")
        if not isinstance(names, list):
            raise ValueError("classes YAML must define names")
        class_to_id = {str(name): index for index, name in enumerate(names)}
        rows = [row for row in csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")) if row["split"] == split]
        if not rows:
            raise ValueError(f"No {split!r} rows in {manifest}")
        self.global_input_size, self.local_input_size = global_input_size, local_input_size
        self.records = []
        self.tile_labels: list[tuple[int, ...]] = []
        for row in rows:
            width, height, boxes = _read_voc(Path(row["xml_path"]), class_to_id)
            for view in official_windows(width, height, tile_size, overlap):
                self.records.append(GridRecord(Path(row["image_path"]), width, height, boxes, view))
                self.tile_labels.append(tuple(_crop_targets(boxes, view, local_input_size)["labels"].tolist()))

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _image_tensor(image: Image.Image, size: int) -> torch.Tensor:
        image = resize(image, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        return normalize(pil_to_tensor(image).float() / 255., MEAN, STD)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record.image_path) as opened:
            image = opened.convert("RGB")
        local_image = image.crop(record.view)
        target = _crop_targets(record.boxes, record.view, self.local_input_size)
        target.update({"view_xyxy": torch.tensor(record.view, dtype=torch.float32),
                       "image_size": torch.tensor((record.width, record.height), dtype=torch.float32),
                       "image_id": record.image_path.name})
        return {"local": self._image_tensor(local_image, self.local_input_size),
                "global": self._image_tensor(image, self.global_input_size), "target": target}


def gl_cascade_collate(batch: list[dict]) -> dict:
    return {"local": torch.stack([item["local"] for item in batch]), "global": torch.stack([item["global"] for item in batch]),
            "targets": [item["target"] for item in batch]}


class OfficialGridBalancedSampler(WeightedRandomSampler):
    """Keep real empty-tile pressure while giving rare positive tiles a chance.

    Empty windows receive a fixed total probability mass.  The remaining mass
    is divided between positive tiles using a capped inverse-square-root count
    for every class present in a tile.  Thus EQLv2 is not asked to repair a
    proposal distribution that almost never sees qilie or huashang.
    """

    def __init__(self, dataset: OfficialGridGlobalLocalDataset, empty_fraction: float = .55,
                 rare_power: float = .5, max_rare_boost: float = 4.0, seed: int = 20260724):
        if not 0 < empty_fraction < 1:
            raise ValueError("empty_fraction must be in (0, 1)")
        labels = dataset.tile_labels
        empty = [index for index, item in enumerate(labels) if not item]
        positive = [index for index, item in enumerate(labels) if item]
        if not empty or not positive:
            raise ValueError("Official grid sampler requires both empty and positive tiles")
        counts = torch.zeros(9, dtype=torch.float64)
        for item in labels:
            for label in set(item):
                counts[label] += 1
        reference = counts[counts > 0].median()
        boost = (reference / counts.clamp(min=1)).pow(rare_power).clamp(max=max_rare_boost)
        weights = torch.zeros(len(labels), dtype=torch.double)
        weights[empty] = empty_fraction / len(empty)
        positive_raw = torch.tensor([max(float(boost[label]) for label in set(labels[index])) for index in positive], dtype=torch.double)
        weights[positive] = positive_raw * ((1 - empty_fraction) / positive_raw.sum())
        generator = torch.Generator(); generator.manual_seed(seed)
        super().__init__(weights, len(weights), replacement=True, generator=generator)
        self.class_tile_counts = counts.to(torch.long).tolist()
        self.empty_fraction = empty_fraction
