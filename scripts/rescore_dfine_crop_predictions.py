#!/usr/bin/env python3
"""Re-score already classified D-FINE proposals without rerunning DINOv2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("keep", "det_obj", "det_obj_softcls", "det_obj_cls"), required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def score(det: float, obj: float, cls: float, mode: str) -> float:
    if mode == "keep": return det
    if mode == "det_obj": return det * obj
    if mode == "det_obj_softcls": return det * obj * (0.5 + 0.5 * cls)
    if mode == "det_obj_cls": return det * obj * cls
    raise ValueError(mode)


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    count = 0
    for record in payload["images"]:
        for pred in record.get("predictions", []):
            detail = pred.get("crop_classifier")
            if not detail:
                raise ValueError("Input lacks crop_classifier probabilities")
            det = float(pred.get("original_score", pred["score"]))
            pred["score"] = max(0.0, min(1.0, score(
                det, float(detail["objectness"]), float(detail["class_probability"]), args.mode
            )))
            detail["score_mode"] = args.mode
            count += 1
    payload.setdefault("metadata", {}).setdefault("dfine_crop_classifier", {})["score_mode"] = args.mode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mode": args.mode, "predictions": count, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
