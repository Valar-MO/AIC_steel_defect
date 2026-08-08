"""Deterministic sampling pools for official whole/tile joint training."""
from __future__ import annotations

import random
from collections import Counter

from torch.utils.data import Dataset

from ...core import register


@register()
class OfficialViewBatchMixDataset(Dataset):
    """Draw a fixed source/content pattern while shuffling each pool per epoch."""
    __inject__ = ["dataset"]

    GROUP_FIELDS = {
        "whole": "is_whole", "natural_tile": "is_tile", "positive_tile": "is_positive",
        "weak_positive_tile": "is_weak_positive", "hard_background_tile": "is_hard_background",
    }

    def __init__(self, dataset, pattern, epoch_size=None, seed=42, transforms=None):
        # D-FINE's inherited dataloader config injects an outer ``transforms``
        # key. The wrapped view dataset owns geometry-aware transforms, so the
        # outer value must not be applied a second time.
        del transforms
        if not pattern: raise ValueError("pattern must not be empty")
        self.dataset, self.pattern, self.seed = dataset, tuple(pattern), int(seed)
        if not hasattr(dataset, "ids") or not hasattr(dataset, "coco"): raise TypeError("Requires COCO-style dataset")
        self.groups = {name: [] for name in set(self.pattern)}
        for index, image_id in enumerate(dataset.ids):
            image = dataset.coco.imgs[int(image_id)]
            for name in self.groups:
                if name not in self.GROUP_FIELDS: raise ValueError(f"Unknown group: {name}")
                if bool(image.get(self.GROUP_FIELDS[name], False)): self.groups[name].append(index)
        missing = [name for name, items in self.groups.items() if not items]
        if missing: raise ValueError(f"Empty sampling groups: {missing}")
        requested = len(dataset) if epoch_size is None else int(epoch_size)
        self.epoch_size = ((requested + len(self.pattern) - 1) // len(self.pattern)) * len(self.pattern)
        self.group_counts = Counter(self.pattern); self.set_epoch(0)

    def __len__(self): return self.epoch_size

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"): self.dataset.set_epoch(epoch)
        self.orders = {}
        for offset, (name, items) in enumerate(sorted(self.groups.items())):
            values = list(items); random.Random(self.seed + 1009 * int(epoch) + offset).shuffle(values); self.orders[name] = values

    def __getitem__(self, index):
        name = self.pattern[index % len(self.pattern)]; cycles, offset = divmod(index, len(self.pattern))
        before = cycles * self.group_counts[name] + sum(item == name for item in self.pattern[:offset])
        return self.dataset[self.orders[name][before % len(self.orders[name])]]
