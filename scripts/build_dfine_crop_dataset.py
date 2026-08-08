#!/usr/bin/env python3
"""Build metadata for the D-FINE -> DINOv2 dual-head crop classifier.

Input CSV files are produced by ``analyze_dfine_proposal_precision.py``.
Only unambiguous proposals are retained:

* tp/duplicate -> defect, with the referenced nine-class GT label;
* background   -> background;
* near_miss/weak_overlap -> ignored.

Images are not copied.  ``metadata.csv`` stores source paths and boxes so the
training loader can create tight/context crops dynamically.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import yaml


POSITIVE_OUTCOMES = {"tp", "duplicate"}
IGNORED_OUTCOMES = {"near_miss", "weak_overlap"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-audit", type=Path, required=True)
    p.add_argument("--val-audit", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-score", type=float, default=0.001)
    p.add_argument("--max-positive-per-gt", type=int, default=3,
                   help="TP plus highest-scoring duplicates retained per GT.")
    p.add_argument("--train-background-ratio", type=float, default=3.0,
                   help="Maximum train backgrounds per retained positive; negative to disable.")
    p.add_argument("--max-train-background-per-image", type=int, default=50)
    p.add_argument("--max-val-background-per-image", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_classes(path: Path) -> list[str]:
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(names, list) or len(names) != 9 or len(set(names)) != 9:
        raise ValueError(f"Expected exactly nine unique classes in {path}")
    return [str(x) for x in names]


def load_manifest(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    result = {"train": {}, "val": {}}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split")
            if split in result:
                result[split][Path(row["image_path"]).name] = row
    return result


def read_audit(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"image_id", "pred_index", "score", "outcome", "best_iou",
                "reference_gt_id", "reference_class", "x1", "y1", "x2", "y2"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def cap_per_image(rows: list[dict], limit: int) -> list[dict]:
    if limit < 0:
        return rows
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["image_id"]].append(row)
    output = []
    for image_rows in grouped.values():
        # Hard negatives first. Stable pred_index tie-break makes runs reproducible.
        image_rows.sort(key=lambda r: (-float(r["score"]), int(r["pred_index"])))
        output.extend(image_rows[:limit])
    return output


def select_rows(rows: list[dict[str, str]], split: str, classes: list[str], args) -> tuple[list[dict], Counter]:
    class_set = set(classes)
    stats = Counter(total_audit=len(rows))
    positives_by_gt: dict[tuple[str, str], list[dict]] = defaultdict(list)
    backgrounds = []
    for row in rows:
        outcome = row["outcome"]
        score = float(row["score"])
        if score < args.min_score:
            stats["ignored_low_score"] += 1
        elif outcome in POSITIVE_OUTCOMES:
            if row["reference_class"] not in class_set or not row["reference_gt_id"]:
                raise ValueError(f"Positive row lacks a valid GT label: {row}")
            positives_by_gt[(row["image_id"], row["reference_gt_id"])].append(row)
        elif outcome == "background":
            backgrounds.append(row)
        elif outcome in IGNORED_OUTCOMES:
            stats[f"ignored_{outcome}"] += 1
        else:
            raise ValueError(f"Unknown audit outcome: {outcome!r}")

    positives = []
    for group in positives_by_gt.values():
        group.sort(key=lambda r: (r["outcome"] != "tp", -float(r["score"])))
        positives.extend(group[: args.max_positive_per_gt])
        stats["ignored_excess_duplicate"] += max(0, len(group) - args.max_positive_per_gt)

    image_cap = (args.max_train_background_per_image if split == "train"
                 else args.max_val_background_per_image)
    backgrounds = cap_per_image(backgrounds, image_cap)
    if split == "train" and args.train_background_ratio >= 0:
        max_backgrounds = int(round(len(positives) * args.train_background_ratio))
        backgrounds.sort(key=lambda r: (-float(r["score"]), r["image_id"], int(r["pred_index"])))
        stats["ignored_background_ratio"] += max(0, len(backgrounds) - max_backgrounds)
        backgrounds = backgrounds[:max_backgrounds]

    stats["positive"] = len(positives)
    stats["background"] = len(backgrounds)
    stats["retained"] = len(positives) + len(backgrounds)
    selected = positives + backgrounds
    random.Random(args.seed + (0 if split == "train" else 1)).shuffle(selected)
    return selected, stats


def enrich(rows: list[dict], split: str, manifest_rows: dict, classes: list[str]) -> list[dict]:
    class_to_id = {name: i for i, name in enumerate(classes)}
    size_cache = {}
    output = []
    for row in rows:
        image_id = Path(row["image_id"]).name
        manifest = manifest_rows.get(image_id)
        if manifest is None:
            raise ValueError(f"{split} audit image absent from manifest: {image_id}")
        source_image = Path(manifest["image_path"])
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        if source_image not in size_cache:
            with Image.open(source_image) as im:
                size_cache[source_image] = im.size
        width, height = size_cache[source_image]
        positive = row["outcome"] in POSITIVE_OUTCOMES
        class_name = row["reference_class"] if positive else "background"
        output.append({
            "split": split, "image_id": image_id, "source_image": str(source_image.resolve()),
            "image_width": width, "image_height": height,
            "pred_index": row["pred_index"], "sample_type": "positive" if positive else "background",
            "audit_outcome": row["outcome"], "objectness": 1 if positive else 0,
            "class_id": class_to_id[class_name] if positive else -1, "class_name": class_name,
            "det_score": row["score"], "iou": row["best_iou"],
            "reference_gt_id": row["reference_gt_id"],
            "proposal_x1": row["x1"], "proposal_y1": row["y1"],
            "proposal_x2": row["x2"], "proposal_y2": row["y2"],
            "source": row.get("source", "unknown"),
            "touches_internal_border": int(str(row.get("touches_internal_border", "0")).lower() in {"1", "true", "yes"}),
        })
    return output


def main() -> None:
    args = parse_args()
    if not 0 <= args.min_score <= 1 or args.max_positive_per_gt < 1:
        raise ValueError("Invalid score or positive cap")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output}; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    classes = load_classes(args.classes)
    manifest = load_manifest(args.manifest)
    audit_paths = {"train": args.train_audit, "val": args.val_audit}
    all_rows, report = [], {"classes": classes, "splits": {}, "policy": vars(args).copy()}
    report["policy"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["policy"].items()}
    for split in ("train", "val"):
        selected, stats = select_rows(read_audit(audit_paths[split]), split, classes, args)
        enriched = enrich(selected, split, manifest[split], classes)
        all_rows.extend(enriched)
        report["splits"][split] = dict(stats)
    if not all_rows or not any(r["split"] == "train" and r["objectness"] == 1 for r in all_rows):
        raise ValueError("Dataset has no train positives")
    metadata = args.output / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    report["metadata"] = str(metadata.resolve())
    (args.output / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
