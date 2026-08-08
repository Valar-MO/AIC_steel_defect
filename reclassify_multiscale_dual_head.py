#!/usr/bin/env python3
"""Re-score/reclassify detector predictions with a multi-scale dual-head verifier."""

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
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_crop_classifier import IMAGENET_MEAN, IMAGENET_STD, load_classes  # noqa: E402
from train_multiscale_dual_head_verifier import (  # noqa: E402
    MultiScaleDualHeadVerifier,
    expanded_crop_box,
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
    parser.add_argument("--checkpoint", "--classifier", dest="checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-weights", type=Path, default=Path("/root/autodl-tmp/weights/dinov2-small"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--tight-expand", type=float, default=1.0)
    parser.add_argument("--context-expand", type=float, default=1.8)
    parser.add_argument("--min-crop-size", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--min-det-score-for-reclass", type=float, default=0.03)
    parser.add_argument("--objectness-threshold", type=float, default=0.50)
    parser.add_argument("--min-cls-conf", type=float, default=0.65)
    parser.add_argument("--background-score-scale", type=float, default=0.10)
    parser.add_argument(
        "--score-mode",
        choices=("objectness", "objectness_class", "objectness_soft_class", "keep"),
        default="objectness_class",
        help=(
            "objectness: det*defect_prob; objectness_class: det*defect_prob*class_prob; "
            "objectness_soft_class: det*defect_prob*(0.5+0.5*class_prob); keep: class changes only."
        ),
    )
    parser.add_argument("--target-classes", nargs="*", default=list(DEFAULT_TARGET_CLASSES))
    parser.add_argument("--frozen-source-classes", nargs="*", default=["qilie", "jieba"])
    parser.add_argument("--blocked-target-classes", nargs="*", default=["qilie", "jieba"])
    parser.add_argument("--max-predictions", type=int)
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
        raise ValueError("Expected prediction JSON with top-level 'images'")
    return payload


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(image.height, image.width, 3).permute(2, 0, 1).float().div_(255.0)
    return (data - IMAGENET_MEAN) / IMAGENET_STD


class PredictionMultiScaleDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        image_map: dict[str, str],
        *,
        image_size: int,
        tight_expand: float,
        context_expand: float,
        min_crop_size: int,
        min_det_score: float,
        max_predictions: int | None = None,
    ) -> None:
        self.items = []
        self.image_size = image_size
        self.tight_expand = tight_expand
        self.context_expand = context_expand
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
                self.items.append({
                    "image_index": image_index,
                    "pred_index": pred_index,
                    "image_id": image_id,
                    "source_path": source_path,
                    "width": int(record.get("width", 0) or 0),
                    "height": int(record.get("height", 0) or 0),
                    "bbox": prediction.get("bbox_xyxy", prediction.get("bbox")),
                })
                if max_predictions is not None and len(self.items) >= max_predictions:
                    return

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(item["source_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width = item["width"] or image.width
            height = item["height"] or image.height
            tight = image.crop(expanded_crop_box(item["bbox"], width, height, self.tight_expand, self.min_crop_size))
            context = image.crop(expanded_crop_box(item["bbox"], width, height, self.context_expand, self.min_crop_size))
        return image_to_tensor(tight, self.image_size), image_to_tensor(context, self.image_size), index


def collate(batch):
    tight, context, indices = zip(*batch)
    return torch.stack(list(tight), 0), torch.stack(list(context), 0), torch.tensor(indices, dtype=torch.long)


def load_model(checkpoint: Path, dinov2_weights: Path, classes: list[str], device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    ckpt_classes = payload.get("classes")
    if ckpt_classes is not None and list(ckpt_classes) != classes:
        raise ValueError("Checkpoint classes differ from --classes")
    model = MultiScaleDualHeadVerifier(dinov2_weights, len(classes), dropout=0.0)
    model.load_state_dict(payload.get("model", payload), strict=True)
    model.to(device).eval()
    return model, payload


@torch.inference_mode()
def run_model(model, dataset, args, device):
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, collate_fn=collate)
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    outputs = {}
    for tight, context, indices in loader:
        tight = tight.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.amp and device.type == "cuda"):
            result = model(tight, context)
        defect_probs = torch.sigmoid(result["objectness_logits"].float()).cpu()
        class_probs = result["class_logits"].float().softmax(dim=1).cpu()
        for obj, cls, index in zip(defect_probs, class_probs, indices.tolist()):
            outputs[index] = (obj, cls)
    return outputs


def should_replace(original: str, candidate: str, candidate_conf: float, args, target_classes, frozen, blocked):
    if candidate == original:
        return False
    if original in frozen:
        return False
    if candidate in blocked:
        return False
    if candidate_conf < args.min_cls_conf:
        return False
    return original in target_classes or candidate in target_classes


def fuse_score(det_score: float, defect_prob: float, class_prob: float, mode: str):
    if mode == "keep":
        return det_score
    if mode == "objectness":
        return det_score * defect_prob
    if mode == "objectness_class":
        return det_score * defect_prob * class_prob
    if mode == "objectness_soft_class":
        return det_score * defect_prob * (0.5 + 0.5 * class_prob)
    raise ValueError(mode)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Pass --overwrite.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.output.with_name(args.output.stem + "_report.json")
    if not 0 <= args.objectness_threshold <= 1 or not 0 <= args.background_score_scale <= 1:
        raise ValueError("threshold/scale must be in [0, 1]")

    classes = load_classes(args.classes)
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    payload = load_payload(args.predictions)
    metadata_classes = payload.get("metadata", {}).get("classes")
    if metadata_classes is not None and list(metadata_classes) != classes:
        raise ValueError("Prediction metadata classes differ from classes config")

    records = payload["images"]
    image_map = load_manifest_image_map(args.manifest)
    dataset = PredictionMultiScaleDataset(
        records,
        image_map,
        image_size=args.image_size,
        tight_expand=args.tight_expand,
        context_expand=args.context_expand,
        min_crop_size=args.min_crop_size,
        min_det_score=args.min_det_score_for_reclass,
        max_predictions=args.max_predictions,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, checkpoint_payload = load_model(args.checkpoint, args.dinov2_weights, classes, device)
    outputs = run_model(model, dataset, args, device)

    target_classes = set(args.target_classes)
    frozen = set(args.frozen_source_classes)
    blocked = set(args.blocked_target_classes)
    item_lookup = {index: item for index, item in enumerate(dataset.items)}
    changed = Counter()
    suppressed = 0
    score_sum_delta = 0.0
    processed = 0
    changed_examples = []

    for item_index, (defect_prob_t, class_probs) in outputs.items():
        item = item_lookup[item_index]
        prediction = records[item["image_index"]]["predictions"][item["pred_index"]]
        original_name = str(prediction["category_name"])
        if original_name not in class_to_id:
            raise ValueError(f"Unknown prediction class: {original_name}")
        candidate_id = int(torch.argmax(class_probs).item())
        candidate_name = classes[candidate_id]
        candidate_conf = float(class_probs[candidate_id].item())
        original_prob = float(class_probs[class_to_id[original_name]].item())
        defect_prob = float(defect_prob_t.item())
        old_score = float(prediction["score"])

        replace = defect_prob >= args.objectness_threshold and should_replace(
            original_name, candidate_name, candidate_conf, args, target_classes, frozen, blocked
        )
        final_name = candidate_name if replace else original_name
        final_id = class_to_id[final_name]
        final_prob = candidate_conf if replace else original_prob
        if defect_prob < args.objectness_threshold:
            new_score = old_score * args.background_score_scale
            suppressed += 1
        else:
            new_score = fuse_score(old_score, defect_prob, final_prob, args.score_mode)

        prediction["original_class_id"] = int(prediction.get("class_id", class_to_id[original_name]))
        prediction["original_category_name"] = original_name
        prediction["original_score"] = old_score
        prediction["class_id"] = final_id
        prediction["category_name"] = final_name
        prediction["score"] = max(0.0, min(1.0, float(new_score)))
        prediction["dual_head_verifier"] = {
            "defect_probability": defect_prob,
            "top_class_id": candidate_id,
            "top_category_name": candidate_name,
            "top_confidence": candidate_conf,
            "original_class_probability": original_prob,
            "reclassified": bool(replace),
            "background_suppressed": bool(defect_prob < args.objectness_threshold),
        }
        processed += 1
        score_sum_delta += prediction["score"] - old_score
        if replace:
            changed[(original_name, final_name)] += 1
            if len(changed_examples) < 20:
                changed_examples.append({
                    "image_id": item["image_id"],
                    "from": original_name,
                    "to": final_name,
                    "det_score": old_score,
                    "new_score": prediction["score"],
                    "defect_probability": defect_prob,
                    "class_confidence": candidate_conf,
                })

    payload.setdefault("metadata", {})
    payload["metadata"]["dual_head_reclassification"] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "classes": classes,
        "model_type": checkpoint_payload.get("model_type", "multiscale_dual_head_verifier"),
        "min_det_score_for_reclass": args.min_det_score_for_reclass,
        "objectness_threshold": args.objectness_threshold,
        "background_score_scale": args.background_score_scale,
        "score_mode": args.score_mode,
        "tight_expand": args.tight_expand,
        "context_expand": args.context_expand,
        "image_size": args.image_size,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = {
        "input_predictions": str(args.predictions.resolve()),
        "output_predictions": str(args.output.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "classes": classes,
        "processed_predictions": processed,
        "background_suppressed": suppressed,
        "changed_predictions": int(sum(changed.values())),
        "changed_pairs": {f"{src}->{dst}": count for (src, dst), count in changed.items()},
        "average_score_delta": score_sum_delta / processed if processed else 0.0,
        "objectness_threshold": args.objectness_threshold,
        "background_score_scale": args.background_score_scale,
        "score_mode": args.score_mode,
        "changed_examples": changed_examples,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
