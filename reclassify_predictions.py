#!/usr/bin/env python3
"""Re-classify detector predictions with a trained crop classifier.

The script keeps detector boxes unchanged and only adjusts class labels/scores.
It is intended as a conservative second-stage experiment:

    detector predictions -> crop proposal boxes -> DINOv2 crop classifier
    -> class/score correction -> evaluate_predictions.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_crop_classifier import (  # noqa: E402
    Dinov2CropClassifier,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_classes,
)


DEFAULT_TARGET_CLASSES = (
    "zonglie",
    "jiaza",
    "yiwuyaru",
    "huashang",
    "mamianmakeng",
    "yanghuatiepi",
    "gunyin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--dinov2-weights", type=Path, default=Path("/root/autodl-tmp/weights/dinov2-small"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument(
        "--classifier-classes",
        type=Path,
        help="Optional classifier classes. Use a 10-class config here for verifier checkpoints with background.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--mode",
        choices=("conservative", "replace_all", "rescore_only"),
        default="conservative",
        help=(
            "conservative: only high-confidence low-risk class changes; "
            "replace_all: classifier argmax always replaces class; "
            "rescore_only: keep detector class and only fuse score"
        ),
    )
    parser.add_argument("--min-cls-conf", type=float, default=0.65)
    parser.add_argument(
        "--min-det-score-for-reclass",
        type=float,
        default=0.03,
        help="Only crop/reclassify predictions with detector score >= this value; lower-score boxes stay unchanged.",
    )
    parser.add_argument(
        "--qilie-promote-conf",
        type=float,
        default=0.80,
        help="Higher confidence needed to change a non-qilie prediction into qilie.",
    )
    parser.add_argument(
        "--target-classes",
        nargs="*",
        default=list(DEFAULT_TARGET_CLASSES),
        help="Classes allowed to be changed in conservative mode.",
    )
    parser.add_argument(
        "--frozen-source-classes",
        nargs="*",
        default=["qilie", "jieba"],
        help="Detector source classes that cannot be changed in conservative mode.",
    )
    parser.add_argument(
        "--blocked-target-classes",
        nargs="*",
        default=["qilie", "jieba"],
        help="Classifier target classes that cannot be introduced in conservative mode.",
    )
    parser.add_argument(
        "--protect-qilie-jieba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not change detector qilie into classifier jieba in conservative mode.",
    )
    parser.add_argument(
        "--allowed-reclass-pairs",
        nargs="*",
        default=[],
        metavar="FROM:TO",
        help=(
            "Optional whitelist of class-change pairs. If provided, a conservative "
            "class change is allowed only when original:candidate is in this list."
        ),
    )
    parser.add_argument(
        "--blocked-reclass-pairs",
        nargs="*",
        default=[],
        metavar="FROM:TO",
        help="Optional blacklist of class-change pairs, formatted as FROM:TO.",
    )
    parser.add_argument("--background-class", default="background")
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=0.70,
        help="If classifier predicts background with confidence >= this, down-score the detector box.",
    )
    parser.add_argument(
        "--background-score-scale",
        type=float,
        default=0.10,
        help="Score multiplier applied to boxes classified as background.",
    )
    parser.add_argument(
        "--score-fusion",
        choices=(
            "soft",
            "soft_light",
            "soft_lighter",
            "disagreement_soft_light",
            "disagreement_soft_lighter",
            "multiply",
            "keep",
        ),
        default="soft",
        help=(
            "Score fusion strategy. "
            "soft = det_score * (0.5 + 0.5 * cls_prob); "
            "soft_light = det_score * (0.8 + 0.2 * cls_prob); "
            "soft_lighter = det_score * (0.9 + 0.1 * cls_prob); "
            "disagreement_* only fuses score when the classifier changes the detector class, "
            "otherwise keeps detector score unchanged."
        ),
    )
    parser.add_argument(
        "--crop-expand",
        type=float,
        default=1.8,
        help="Proposal crop expansion ratio. Mirrors GT crop dataset default.",
    )
    parser.add_argument("--min-crop-size", type=int, default=128)
    parser.add_argument("--max-predictions", type=int, help="Smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest_image_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {Path(row["image_path"]).name: row["image_path"] for row in rows if row.get("image_path")}


def load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "images" not in payload:
        raise ValueError("Expected raw prediction JSON with top-level 'images'")
    if not isinstance(payload["images"], list):
        raise ValueError("predictions['images'] must be a list")
    return payload


def make_crop_box(
    box: list[float],
    width: int,
    height: int,
    expand: float,
    min_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {box}")
    side = max(x2 - x1, y2 - y1) * expand
    side = min(max(side, float(min_size)), float(max(width, height)))
    crop_w = min(side, float(width))
    crop_h = min(side, float(height))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = min(max(cx - crop_w / 2.0, 0.0), float(width) - crop_w)
    top = min(max(cy - crop_h / 2.0, 0.0), float(height) - crop_h)
    return (
        int(math.floor(left)),
        int(math.floor(top)),
        int(math.ceil(left + crop_w)),
        int(math.ceil(top + crop_h)),
    )


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(image.height, image.width, 3).permute(2, 0, 1).float().div_(255.0)
    return (data - IMAGENET_MEAN) / IMAGENET_STD


class PredictionCropDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        image_map: dict[str, str],
        *,
        image_size: int,
        crop_expand: float,
        min_crop_size: int,
        min_det_score: float,
        max_predictions: int | None = None,
    ) -> None:
        self.items = []
        self.image_size = image_size
        self.crop_expand = crop_expand
        self.min_crop_size = min_crop_size

        for image_index, record in enumerate(records):
            image_id = Path(str(record["image_id"])).name
            source = record.get("source_path") or image_map.get(image_id)
            if not source:
                raise FileNotFoundError(f"No source_path or manifest entry for {image_id}")
            source_path = Path(source)
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing source image for {image_id}: {source_path}")
            for pred_index, prediction in enumerate(record.get("predictions", [])):
                if float(prediction.get("score", 0.0)) < min_det_score:
                    continue
                self.items.append(
                    {
                        "image_index": image_index,
                        "pred_index": pred_index,
                        "image_id": image_id,
                        "source_path": source_path,
                        "width": int(record.get("width", 0) or 0),
                        "height": int(record.get("height", 0) or 0),
                        "bbox": prediction.get("bbox_xyxy", prediction.get("bbox")),
                    }
                )
                if max_predictions is not None and len(self.items) >= max_predictions:
                    return

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(item["source_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width = item["width"] or image.width
            height = item["height"] or image.height
            crop_box = make_crop_box(item["bbox"], width, height, self.crop_expand, self.min_crop_size)
            crop = image.crop(crop_box)
            tensor = image_to_tensor(crop, self.image_size)
        return tensor, index


def collate(batch):
    images, indices = zip(*batch)
    return torch.stack(list(images), dim=0), torch.tensor(indices, dtype=torch.long)


def load_classifier(checkpoint: Path, dinov2_weights: Path, classifier_classes: list[str], device: torch.device):
    model = Dinov2CropClassifier(dinov2_weights, len(classifier_classes), dropout=0.0)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    ckpt_classes = payload.get("classes")
    if ckpt_classes is not None and list(ckpt_classes) != classifier_classes:
        raise ValueError("Classifier checkpoint classes differ from classes config")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def should_replace(
    mode: str,
    original: str,
    candidate: str,
    confidence: float,
    *,
    min_conf: float,
    qilie_promote_conf: float,
    target_classes: set[str],
    frozen_source_classes: set[str],
    blocked_target_classes: set[str],
    protect_qilie_jieba: bool,
    allowed_reclass_pairs: set[tuple[str, str]],
    blocked_reclass_pairs: set[tuple[str, str]],
) -> bool:
    if mode == "replace_all":
        return True
    if mode == "rescore_only":
        return False
    if candidate == original:
        return False
    pair = (original, candidate)
    if pair in blocked_reclass_pairs:
        return False
    if allowed_reclass_pairs and pair not in allowed_reclass_pairs:
        return False
    if original in frozen_source_classes:
        return False
    if candidate in blocked_target_classes:
        return False
    if confidence < min_conf:
        return False
    if original == "qilie" and candidate == "jieba" and protect_qilie_jieba:
        return False
    if candidate == "qilie" and confidence < qilie_promote_conf:
        return False
    return original in target_classes or candidate in target_classes


def parse_reclass_pairs(values: list[str], classes: list[str], *, option_name: str) -> set[tuple[str, str]]:
    class_set = set(classes)
    pairs = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"{option_name} entries must be formatted as FROM:TO, got {value!r}")
        source, target = [part.strip() for part in value.split(":", 1)]
        if not source or not target:
            raise ValueError(f"{option_name} entries must be formatted as FROM:TO, got {value!r}")
        unknown = [name for name in (source, target) if name not in class_set]
        if unknown:
            raise ValueError(f"{option_name} contains unknown classes {unknown} in {value!r}")
        if source == target:
            raise ValueError(f"{option_name} must contain class-change pairs, got {value!r}")
        pairs.add((source, target))
    return pairs


def fuse_score(det_score: float, cls_prob: float, mode: str) -> float:
    det_score = float(det_score)
    cls_prob = float(cls_prob)
    if mode == "keep":
        return det_score
    if mode == "multiply":
        return det_score * cls_prob
    if mode == "soft":
        return det_score * (0.5 + 0.5 * cls_prob)
    if mode == "soft_light":
        return det_score * (0.8 + 0.2 * cls_prob)
    if mode == "soft_lighter":
        return det_score * (0.9 + 0.1 * cls_prob)
    raise ValueError(mode)


def fuse_score_for_decision(det_score: float, cls_prob: float, mode: str, *, replace: bool) -> float:
    """Fuse scores after the reclassification decision is made.

    The disagreement_* modes are intentionally conservative: if the crop
    classifier does not trigger a class change, preserve the detector score to
    avoid pushing borderline true positives below the evaluation threshold.
    """
    if mode == "disagreement_soft_light":
        return fuse_score(det_score, cls_prob, "soft_light") if replace else float(det_score)
    if mode == "disagreement_soft_lighter":
        return fuse_score(det_score, cls_prob, "soft_lighter") if replace else float(det_score)
    return fuse_score(det_score, cls_prob, mode)


@torch.inference_mode()
def run_classifier(model, dataset, args, device):
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
    )
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    outputs = {}
    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.amp and device.type == "cuda"):
            logits = model(images)
        probs = logits.float().softmax(dim=1).cpu()
        for row, index in zip(probs, indices.tolist()):
            outputs[index] = row
    return outputs


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Pass --overwrite.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.output.with_name(args.output.stem + "_report.json")

    detector_classes = load_classes(args.classes)
    classifier_classes = load_classes(args.classifier_classes or args.classes)
    class_to_id = {name: idx for idx, name in enumerate(detector_classes)}
    classifier_class_to_id = {name: idx for idx, name in enumerate(classifier_classes)}
    unknown_targets = sorted(set(args.target_classes) - set(detector_classes))
    if unknown_targets:
        raise ValueError(f"Unknown target classes: {unknown_targets}")
    unknown_frozen = sorted(set(args.frozen_source_classes) - set(detector_classes))
    if unknown_frozen:
        raise ValueError(f"Unknown frozen source classes: {unknown_frozen}")
    unknown_blocked = sorted(set(args.blocked_target_classes) - set(detector_classes))
    if unknown_blocked:
        raise ValueError(f"Unknown blocked target classes: {unknown_blocked}")
    if args.background_class in detector_classes:
        raise ValueError("--background-class must not be a detector output class")
    if args.classifier_classes is not None and args.background_class not in classifier_classes:
        raise ValueError(f"--background-class {args.background_class!r} not found in classifier classes")
    if not 0 <= args.background_threshold <= 1:
        raise ValueError("--background-threshold must be in [0, 1]")
    if not 0 <= args.background_score_scale <= 1:
        raise ValueError("--background-score-scale must be in [0, 1]")
    missing_classifier_classes = sorted(set(detector_classes) - set(classifier_classes))
    if missing_classifier_classes:
        raise ValueError(f"Classifier classes miss detector classes: {missing_classifier_classes}")
    payload = load_payload(args.predictions)
    metadata_classes = payload.get("metadata", {}).get("classes")
    if metadata_classes is not None and list(metadata_classes) != detector_classes:
        raise ValueError("Prediction metadata classes differ from classes config")

    records = payload["images"]
    image_map = load_manifest_image_map(args.manifest)
    dataset = PredictionCropDataset(
        records,
        image_map,
        image_size=args.image_size,
        crop_expand=args.crop_expand,
        min_crop_size=args.min_crop_size,
        min_det_score=args.min_det_score_for_reclass,
        max_predictions=args.max_predictions,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_classifier(args.classifier, args.dinov2_weights, classifier_classes, device)
    probs_by_index = run_classifier(model, dataset, args, device)

    target_classes = set(args.target_classes)
    frozen_source_classes = set(args.frozen_source_classes)
    blocked_target_classes = set(args.blocked_target_classes)
    allowed_reclass_pairs = parse_reclass_pairs(
        args.allowed_reclass_pairs, detector_classes, option_name="--allowed-reclass-pairs")
    blocked_reclass_pairs = parse_reclass_pairs(
        args.blocked_reclass_pairs, detector_classes, option_name="--blocked-reclass-pairs")
    overlap = allowed_reclass_pairs & blocked_reclass_pairs
    if overlap:
        raise ValueError(f"Pairs cannot be both allowed and blocked: {sorted(overlap)}")
    changed = Counter()
    background_suppressed = 0
    original_counts = Counter()
    output_counts = Counter()
    score_sum_delta = 0.0
    processed = 0
    skipped_by_limit = 0
    changed_examples = []

    # Map dataset item index back to image/prediction positions.
    item_lookup = {index: item for index, item in enumerate(dataset.items)}
    for item_index, probs in probs_by_index.items():
        item = item_lookup[item_index]
        prediction = records[item["image_index"]]["predictions"][item["pred_index"]]
        original_name = str(prediction["category_name"])
        if original_name not in class_to_id:
            raise ValueError(f"Unknown prediction class: {original_name}")
        original_id = class_to_id[original_name]
        candidate_id = int(torch.argmax(probs).item())
        candidate_name = classifier_classes[candidate_id]
        candidate_conf = float(probs[candidate_id].item())
        original_prob = float(probs[classifier_class_to_id[original_name]].item())
        suppress_background = (
            candidate_name == args.background_class
            and candidate_conf >= args.background_threshold
        )
        replace = False
        if not suppress_background and candidate_name in class_to_id:
            replace = should_replace(
                args.mode,
                original_name,
                candidate_name,
                candidate_conf,
                min_conf=args.min_cls_conf,
                qilie_promote_conf=args.qilie_promote_conf,
                target_classes=target_classes,
                frozen_source_classes=frozen_source_classes,
                blocked_target_classes=blocked_target_classes,
                protect_qilie_jieba=args.protect_qilie_jieba,
                allowed_reclass_pairs=allowed_reclass_pairs,
                blocked_reclass_pairs=blocked_reclass_pairs,
            )
        final_name = candidate_name if replace else original_name
        final_id = class_to_id[final_name]
        final_prob = candidate_conf if replace else original_prob
        old_score = float(prediction["score"])
        if suppress_background:
            new_score = old_score * args.background_score_scale
            background_suppressed += 1
        else:
            new_score = fuse_score_for_decision(old_score, final_prob, args.score_fusion, replace=replace)

        prediction["original_class_id"] = int(prediction.get("class_id", original_id))
        prediction["original_category_name"] = original_name
        prediction["original_score"] = old_score
        prediction["class_id"] = final_id
        prediction["category_name"] = final_name
        prediction["score"] = max(0.0, min(1.0, float(new_score)))
        prediction["crop_classifier"] = {
            "top_class_id": candidate_id,
            "top_category_name": candidate_name,
            "top_confidence": candidate_conf,
            "original_class_probability": original_prob,
            "reclassified": bool(replace),
            "background_suppressed": bool(suppress_background),
        }

        processed += 1
        original_counts[original_name] += 1
        output_counts[final_name] += 1
        score_sum_delta += prediction["score"] - old_score
        if replace:
            changed[(original_name, final_name)] += 1
            if len(changed_examples) < 20:
                changed_examples.append(
                    {
                        "image_id": item["image_id"],
                        "from": original_name,
                        "to": final_name,
                        "det_score": old_score,
                        "new_score": prediction["score"],
                        "classifier_conf": candidate_conf,
                        "background_suppressed": bool(suppress_background),
                    }
                )

    if args.max_predictions is not None:
        total_predictions = sum(len(record.get("predictions", [])) for record in records)
        skipped_by_limit = max(0, total_predictions - processed)

    payload.setdefault("metadata", {})
    payload["metadata"]["reclassification"] = {
        "classifier": str(args.classifier.resolve()),
        "dinov2_weights": str(args.dinov2_weights.resolve()),
        "detector_classes": detector_classes,
        "classifier_classes": classifier_classes,
        "classifier_classes_path": str((args.classifier_classes or args.classes).resolve()),
        "mode": args.mode,
        "min_cls_conf": args.min_cls_conf,
        "min_det_score_for_reclass": args.min_det_score_for_reclass,
        "qilie_promote_conf": args.qilie_promote_conf,
        "target_classes": sorted(target_classes),
        "frozen_source_classes": sorted(frozen_source_classes),
        "blocked_target_classes": sorted(blocked_target_classes),
        "allowed_reclass_pairs": [f"{source}->{target}" for source, target in sorted(allowed_reclass_pairs)],
        "blocked_reclass_pairs": [f"{source}->{target}" for source, target in sorted(blocked_reclass_pairs)],
        "protect_qilie_jieba": args.protect_qilie_jieba,
        "score_fusion": args.score_fusion,
        "background_class": args.background_class,
        "background_threshold": args.background_threshold,
        "background_score_scale": args.background_score_scale,
        "crop_expand": args.crop_expand,
        "min_crop_size": args.min_crop_size,
        "image_size": args.image_size,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = {
        "input_predictions": str(args.predictions.resolve()),
        "output_predictions": str(args.output.resolve()),
        "classifier": str(args.classifier.resolve()),
        "detector_classes": detector_classes,
        "classifier_classes": classifier_classes,
        "classifier_classes_path": str((args.classifier_classes or args.classes).resolve()),
        "mode": args.mode,
        "score_fusion": args.score_fusion,
        "background_class": args.background_class,
        "background_threshold": args.background_threshold,
        "background_score_scale": args.background_score_scale,
        "min_det_score_for_reclass": args.min_det_score_for_reclass,
        "processed_predictions": processed,
        "skipped_by_limit": skipped_by_limit,
        "changed_predictions": int(sum(changed.values())),
        "background_suppressed": background_suppressed,
        "changed_pairs": {f"{src}->{dst}": count for (src, dst), count in changed.items()},
        "original_class_counts": dict(original_counts),
        "output_class_counts": dict(output_counts),
        "average_score_delta": score_sum_delta / processed if processed else 0.0,
        "changed_examples": changed_examples,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
