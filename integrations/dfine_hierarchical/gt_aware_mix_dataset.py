"""Deterministic grouped sampling for the GT-aware D-FINE E1 experiment."""

from __future__ import annotations

import random
from collections import Counter

import torch.multiprocessing as multiprocessing
from torch.utils.data import Dataset

from ...core import register


@register()
class GTAwareBatchMixDataset(Dataset):
    """Wrap a COCO dataset and enforce a repeatable per-microbatch composition.

    ``pattern`` is repeated over the epoch.  With the E1 pattern and a physical
    batch size of four, every batch contains two baseline samples and two
    targeted crops; across eight batches the exact 16/6/6/4 ratio is obtained.
    The underlying sample order is reshuffled at every epoch without changing
    the group composition or duplicating image files.
    """

    __inject__ = ["dataset"]

    def __init__(self, dataset, pattern, epoch_size=None, seed=42):
        # AutoDL containers can exhaust ancillary file descriptors when large
        # tensor batches use PyTorch's default file_descriptor IPC strategy.
        # file_system keeps multi-worker loading stable for a full epoch.
        if multiprocessing.get_sharing_strategy() != "file_system":
            multiprocessing.set_sharing_strategy("file_system")
        if not pattern:
            raise ValueError("pattern must not be empty")
        self.dataset = dataset
        self.pattern = tuple(str(value) for value in pattern)
        self.seed = int(seed)
        requested_size = len(dataset) if epoch_size is None else int(epoch_size)
        if requested_size <= 0:
            raise ValueError("epoch_size must be positive")
        # Avoid a partial pattern at the end of an epoch.  The training config
        # also makes this divisible by the physical batch size.
        self.epoch_size = (
            (requested_size + len(self.pattern) - 1) // len(self.pattern)
        ) * len(self.pattern)

        groups = {name: [] for name in set(self.pattern)}
        if not hasattr(dataset, "ids") or not hasattr(dataset, "coco"):
            raise TypeError("GTAwareBatchMixDataset requires a COCO-style dataset")
        for dataset_index, image_id in enumerate(dataset.ids):
            image = dataset.coco.imgs[int(image_id)]
            group = str(image.get("gtaware_group", "base"))
            if group in groups:
                groups[group].append(dataset_index)
        missing = [name for name, indices in groups.items() if not indices]
        if missing:
            raise ValueError(f"No samples found for GT-aware groups: {missing}")
        self.groups = groups
        self.group_counts_per_pattern = Counter(self.pattern)
        self._epoch = -1
        self._orders = {}
        self.set_epoch(0)

    def __len__(self):
        return self.epoch_size

    def set_epoch(self, epoch):
        self._epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        orders = {}
        for offset, name in enumerate(sorted(self.groups)):
            indices = list(self.groups[name])
            random.Random(self.seed + 1009 * self._epoch + 7919 * offset).shuffle(indices)
            orders[name] = indices
        self._orders = orders

    @property
    def epoch(self):
        return self._epoch

    def _occurrence_before(self, position, group):
        full_patterns, remainder = divmod(position, len(self.pattern))
        return (
            full_patterns * self.group_counts_per_pattern[group]
            + sum(value == group for value in self.pattern[:remainder])
        )

    def source_index(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = self.pattern[index % len(self.pattern)]
        occurrence = self._occurrence_before(index, group)
        order = self._orders[group]
        # Small targeted groups intentionally cycle.  The order changes every
        # epoch, so repeated crops are not presented in a fixed sequence.
        return group, order[occurrence % len(order)]

    def __getitem__(self, index):
        _group, source_index = self.source_index(index)
        return self.dataset[source_index]

    def sampling_report(self):
        counts = Counter(self.pattern)
        repeats = self.epoch_size // len(self.pattern)
        return {
            "epoch_size": self.epoch_size,
            "pattern_length": len(self.pattern),
            "pattern_counts": dict(counts),
            "epoch_counts": {name: count * repeats for name, count in counts.items()},
            "pool_sizes": {name: len(indices) for name, indices in self.groups.items()},
        }
