#!/usr/bin/env python3
"""Create a reproducible, group-safe multilabel train/validation split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


PATTERNS = (
    ("numeric", re.compile(r"^(?P<group>\d+)-Raw\d+-f_\d+$", re.I)),
    ("numeric_underscore", re.compile(r"^(?P<group>\d+)_Raw\d+_f_\d+$", re.I)),
    ("c_series", re.compile(r"^(?P<group>C\d+)_V\d+_F\d+(?:_.+)?$", re.I)),
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class ImageRecord:
    image_path: Path
    xml_path: Path
    group_id: str
    class_bbox_counts: Counter = field(default_factory=Counter)

    @property
    def labels(self) -> set[str]:
        return set(self.class_bbox_counts)

    @property
    def is_empty(self) -> bool:
        return not self.class_bbox_counts


@dataclass
class GroupRecord:
    group_id: str
    images: list[ImageRecord] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return len(self.images)


@dataclass
class Stats:
    image_count: int
    empty_count: int
    class_image_counts: Counter
    class_bbox_counts: Counter
    class_group_counts: Counter


RARE_BOUNDS = {
    "qilie": {"images": (5, 7), "min_groups": 2},
    "huashang": {"images": (10, 14), "min_groups": 3},
    "jiaza": {"images": (20, 25), "min_groups": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_defect/train/train"),
    )
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits/v1"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--attempts", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-size-tolerance", type=int, default=20,
        help="Allowed difference from the rounded validation image target.",
    )
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{path} must contain a non-empty 'names' list")
    return [str(name) for name in names]


def extract_group_id(stem: str) -> str:
    for family, pattern in PATTERNS:
        match = pattern.fullmatch(stem)
        if match:
            return f"{family}:{match.group('group')}"
    raise ValueError(f"Unrecognized filename pattern: {stem}")


def find_image(data_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = data_dir / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def read_index(data_dir: Path, classes: list[str]) -> list[ImageRecord]:
    known = set(classes)
    records = []
    xml_paths = sorted(data_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No XML files found in {data_dir}")
    for xml_path in xml_paths:
        image_path = find_image(data_dir, xml_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"Image missing for {xml_path}")
        root = ET.parse(xml_path).getroot()
        counts = Counter()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in known:
                raise ValueError(f"Unknown class {name!r} in {xml_path}")
            counts[name] += 1
        records.append(
            ImageRecord(
                image_path=image_path.resolve(),
                xml_path=xml_path.resolve(),
                group_id=extract_group_id(xml_path.stem),
                class_bbox_counts=counts,
            )
        )
    return records


def make_groups(records: Iterable[ImageRecord]) -> list[GroupRecord]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    return [GroupRecord(group_id, images) for group_id, images in sorted(grouped.items())]


def calculate_stats(groups: Iterable[GroupRecord], classes: list[str]) -> Stats:
    image_count = empty_count = 0
    class_images = Counter({name: 0 for name in classes})
    class_boxes = Counter({name: 0 for name in classes})
    class_groups = Counter({name: 0 for name in classes})
    for group in groups:
        group_labels = set()
        for image in group.images:
            image_count += 1
            empty_count += int(image.is_empty)
            group_labels.update(image.labels)
            for name in image.labels:
                class_images[name] += 1
            class_boxes.update(image.class_bbox_counts)
        for name in group_labels:
            class_groups[name] += 1
    return Stats(image_count, empty_count, class_images, class_boxes, class_groups)


def relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / max(target, 1.0)


def hard_constraints(
    val: Stats, val_group_ids: set[str], all_group_ids: set[str],
    target_images: int, tolerance: int,
) -> list[str]:
    failures = []
    if abs(val.image_count - target_images) > tolerance:
        failures.append("validation image count outside tolerance")
    if not val_group_ids or val_group_ids == all_group_ids:
        failures.append("one split has no groups")
    for name, rule in RARE_BOUNDS.items():
        low, high = rule["images"]
        if not low <= val.class_image_counts[name] <= high:
            failures.append(f"{name} image count outside [{low}, {high}]")
        if val.class_group_counts[name] < rule["min_groups"]:
            failures.append(f"{name} group count below {rule['min_groups']}")
    return failures


def score_candidate(val: Stats, total: Stats, classes: list[str], ratio: float) -> float:
    image_total_error = relative_error(val.image_count, total.image_count * ratio)
    empty_error = relative_error(val.empty_count, total.empty_count * ratio)
    class_image_error = sum(
        relative_error(val.class_image_counts[name], total.class_image_counts[name] * ratio)
        for name in classes
    )
    class_bbox_error = sum(
        relative_error(val.class_bbox_counts[name], total.class_bbox_counts[name] * ratio)
        for name in classes
    )

    rare_boundary_penalty = 0.0
    rare_group_penalty = 0.0
    for name, rule in RARE_BOUNDS.items():
        low, high = rule["images"]
        actual_images = val.class_image_counts[name]
        if actual_images < low:
            rare_boundary_penalty += (low - actual_images) / low
        elif actual_images > high:
            rare_boundary_penalty += (actual_images - high) / high

        preferred_groups = max(rule["min_groups"], round(total.class_group_counts[name] * ratio))
        preferred_groups = min(preferred_groups, total.class_group_counts[name])
        if val.class_group_counts[name] < preferred_groups:
            rare_group_penalty += (
                preferred_groups - val.class_group_counts[name]
            ) / max(preferred_groups, 1)

    return (
        3.0 * image_total_error
        + 2.0 * empty_error
        + 2.0 * class_image_error
        + 0.8 * class_bbox_error
        + 8.0 * rare_boundary_penalty
        + 4.0 * rare_group_penalty
    )


def random_candidate(groups: list[GroupRecord], target_images: int, rng: random.Random) -> set[str]:
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    selected = set()
    count = 0
    # Packing whole groups without exceeding the exact target. Singleton groups
    # make an exact 640-image validation set possible for this dataset.
    for group in shuffled:
        if count + group.image_count <= target_images:
            selected.add(group.group_id)
            count += group.image_count
        if count == target_images:
            break
    return selected


def search_split(
    groups: list[GroupRecord], classes: list[str], ratio: float,
    attempts: int, seed: int, tolerance: int,
):
    total = calculate_stats(groups, classes)
    target_images = round(total.image_count * ratio)
    all_ids = {group.group_id for group in groups}
    rng = random.Random(seed)
    best = None
    valid_count = 0
    failure_counts = Counter()

    for _ in range(attempts):
        val_ids = random_candidate(groups, target_images, rng)
        val_groups = [group for group in groups if group.group_id in val_ids]
        val_stats = calculate_stats(val_groups, classes)
        failures = hard_constraints(val_stats, val_ids, all_ids, target_images, tolerance)
        if failures:
            failure_counts.update(failures)
            continue
        valid_count += 1
        value = score_candidate(val_stats, total, classes, ratio)
        if best is None or value < best[0]:
            best = (value, val_ids, val_stats)

    if best is None:
        common = "; ".join(f"{k}: {v}" for k, v in failure_counts.most_common())
        raise RuntimeError(f"No valid split found after {attempts} attempts. Failures: {common}")
    return best, total, valid_count, failure_counts


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def write_outputs(
    output_dir: Path, groups: list[GroupRecord], val_ids: set[str], classes: list[str],
    total: Stats, val: Stats, score: float, args: argparse.Namespace,
    valid_candidates: int, failure_counts: Counter,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [image for group in groups for image in group.images]
    records.sort(key=lambda item: item.image_path.name)
    train = [r for r in records if r.group_id not in val_ids]
    valid = [r for r in records if r.group_id in val_ids]
    write_lines(output_dir / "train.txt", (str(r.image_path) for r in train))
    write_lines(output_dir / "val.txt", (str(r.image_path) for r in valid))

    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["image_path", "xml_path", "group_id", "split", "is_empty", "labels", "bbox_counts"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "image_path": record.image_path,
                "xml_path": record.xml_path,
                "group_id": record.group_id,
                "split": "val" if record.group_id in val_ids else "train",
                "is_empty": record.is_empty,
                "labels": "|".join(sorted(record.labels)),
                "bbox_counts": json.dumps(dict(record.class_bbox_counts), ensure_ascii=False, sort_keys=True),
            })

    train_groups = [group for group in groups if group.group_id not in val_ids]
    train_stats = calculate_stats(train_groups, classes)
    summary_rows = []
    for name in ["__all__", "__empty__", *classes]:
        if name == "__all__":
            total_value, train_value, val_value = total.image_count, train_stats.image_count, val.image_count
            total_groups = len(groups)
            train_group_count, val_group_count = len(train_groups), len(val_ids)
            total_boxes = train_boxes = val_boxes = ""
        elif name == "__empty__":
            total_value, train_value, val_value = total.empty_count, train_stats.empty_count, val.empty_count
            total_groups = train_group_count = val_group_count = ""
            total_boxes = train_boxes = val_boxes = ""
        else:
            total_value = total.class_image_counts[name]
            train_value = train_stats.class_image_counts[name]
            val_value = val.class_image_counts[name]
            total_groups = total.class_group_counts[name]
            train_group_count = train_stats.class_group_counts[name]
            val_group_count = val.class_group_counts[name]
            total_boxes = total.class_bbox_counts[name]
            train_boxes = train_stats.class_bbox_counts[name]
            val_boxes = val.class_bbox_counts[name]
        summary_rows.append({
            "category": name,
            "total_images": total_value,
            "train_images": train_value,
            "val_images": val_value,
            "val_image_ratio": val_value / max(total_value, 1),
            "total_groups": total_groups,
            "train_groups": train_group_count,
            "val_groups": val_group_count,
            "total_bboxes": total_boxes,
            "train_bboxes": train_boxes,
            "val_bboxes": val_boxes,
            "val_bbox_ratio": (val_boxes / max(total_boxes, 1)) if isinstance(val_boxes, int) else "",
        })
    with (output_dir / "split_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    config = {
        "data_dir": str(args.data_dir.resolve()),
        "classes_file": str(args.classes.resolve()),
        "classes": classes,
        "val_ratio": args.val_ratio,
        "attempts": args.attempts,
        "seed": args.seed,
        "val_size_tolerance": args.val_size_tolerance,
        "selected_score": score,
        "valid_candidates": valid_candidates,
        "candidate_failure_counts": dict(failure_counts),
        "rare_bounds": RARE_BOUNDS,
        "group_patterns": [family for family, _ in PATTERNS],
    }
    with (output_dir / "split_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def audit(groups: list[GroupRecord], val_ids: set[str], expected_images: int) -> None:
    all_images = [str(image.image_path) for group in groups for image in group.images]
    if len(all_images) != expected_images or len(set(all_images)) != expected_images:
        raise AssertionError("Images are missing or duplicated in the index")
    train_ids = {group.group_id for group in groups if group.group_id not in val_ids}
    if train_ids & val_ids:
        raise AssertionError("Group leakage detected")
    if train_ids | val_ids != {group.group_id for group in groups}:
        raise AssertionError("Some groups were not assigned")


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    classes = load_classes(args.classes)
    records = read_index(args.data_dir, classes)
    groups = make_groups(records)
    (best, total, valid_count, failures) = search_split(
        groups, classes, args.val_ratio, args.attempts, args.seed, args.val_size_tolerance
    )
    score, val_ids, val_stats = best
    audit(groups, val_ids, len(records))
    write_outputs(
        args.output_dir, groups, val_ids, classes, total, val_stats, score, args,
        valid_count, failures,
    )
    print(f"Images: {total.image_count} (train={total.image_count-val_stats.image_count}, val={val_stats.image_count})")
    print(f"Groups: {len(groups)} (train={len(groups)-len(val_ids)}, val={len(val_ids)})")
    print(f"Empty val: {val_stats.empty_count}/{total.empty_count} ({val_stats.empty_count/total.empty_count:.1%})")
    for name in classes:
        print(
            f"{name:16s} val images={val_stats.class_image_counts[name]:3d}/{total.class_image_counts[name]:3d} "
            f"groups={val_stats.class_group_counts[name]:3d}/{total.class_group_counts[name]:3d} "
            f"bboxes={val_stats.class_bbox_counts[name]:4d}/{total.class_bbox_counts[name]:4d}"
        )
    print(f"Valid candidates: {valid_count}/{args.attempts}; score={score:.6f}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
