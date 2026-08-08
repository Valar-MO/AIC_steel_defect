#!/usr/bin/env python3
"""Print compact official metric rows for cached ROI epoch evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--epochs", type=int, required=True)
    args = p.parse_args()
    print("epoch,tp,fp,fn,precision,recall,f2,ap50,pure_bg,duplicate,recall_huashang,recall_qilie,recall_zonglie")
    for epoch in range(1, args.epochs + 1):
        root = args.root / f"epoch_{epoch:02d}"
        report = json.loads((root / "evaluation" / "evaluation_report.json").read_text(encoding="utf-8"))
        audit = json.loads((root / "audit" / "summary.json").read_text(encoding="utf-8"))
        overall, classes = report["overall"], report["per_class"]
        row = [epoch, overall["tp"], overall["fp"], overall["fn"], overall["precision"],
               overall["recall"], overall["f2"], overall["map50"],
               audit["cumulative_by_threshold"][0]["background"],
               audit["cumulative_by_threshold"][0]["duplicate"],
               classes["huashang"]["recall"], classes["qilie"]["recall"], classes["zonglie"]["recall"]]
        print(",".join(str(value) for value in row))


if __name__ == "__main__": main()
