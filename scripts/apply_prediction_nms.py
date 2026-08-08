#!/usr/bin/env python3
"""Apply deterministic image-level NMS to an existing raw prediction JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, required=True)
    parser.add_argument(
        "--class-aware", action="store_true",
        help="Suppress only predictions with the same top-1 class. The default "
             "is class-agnostic competition, matching class-agnostic defectness.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    if not 0 <= args.nms_iou <= 1:
        raise ValueError("--nms-iou must be in [0, 1]")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    before = after = 0
    records = []
    for record in payload["images"]:
        predictions = sorted(
            record.get("predictions", []),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        before += len(predictions)
        kept = []
        for prediction in predictions:
            box = prediction.get("bbox_xyxy", prediction.get("bbox"))
            class_id = int(prediction["class_id"])
            if any(
                (not args.class_aware or int(selected["class_id"]) == class_id)
                and iou(
                    box,
                    selected.get("bbox_xyxy", selected.get("bbox")),
                ) > args.nms_iou
                for selected in kept
            ):
                continue
            kept.append(prediction)
        after += len(kept)
        updated = dict(record)
        updated["predictions"] = kept
        records.append(updated)

    metadata = dict(payload.get("metadata", {}))
    metadata.update({
        "nms_iou": args.nms_iou,
        "nms_class_aware": args.class_aware,
        "predictions_before_nms": before,
        "predictions_after_nms": after,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "images": records}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
