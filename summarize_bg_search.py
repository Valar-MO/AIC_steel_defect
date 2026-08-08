import csv
import json
from pathlib import Path

root = Path("/root/autodl-tmp/evaluations/bg_verifier_threshold_search_min003")
rows = []
for d in sorted(root.glob("thr*_scale*")):
    er = d / "metrics" / "evaluation_report.json"
    rr = d / "reclass_report.json"
    if not er.exists() or not rr.exists():
        continue
    e = json.loads(er.read_text())["overall"]
    r = json.loads(rr.read_text())
    name = d.name
    threshold_text, scale_text = name.split("_scale")
    rows.append({
        "name": name,
        "background_threshold": float(threshold_text.replace("thr", "")),
        "background_score_scale": float(scale_text),
        "map50": e["map50"],
        "map50_95": e["map50_95"],
        "precision": e["precision"],
        "recall": e["recall"],
        "f1": e["f1"],
        "f2": e["f2"],
        "tp": e["tp"],
        "fp": e["fp"],
        "fn": e["fn"],
        "prediction_count": e["prediction_count"],
        "background_suppressed": r.get("background_suppressed", 0),
        "changed_predictions": r.get("changed_predictions", 0),
    })

rows_by_map = sorted(rows, key=lambda item: item["map50_95"], reverse=True)
if rows_by_map:
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_by_map[0].keys()))
        writer.writeheader()
        writer.writerows(rows_by_map)
    (root / "summary.json").write_text(
        json.dumps(rows_by_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

print("TOP_BY_MAP50_95")
for row in rows_by_map:
    print(json.dumps(row, ensure_ascii=False))

print("\nTOP_BY_F2")
for row in sorted(rows, key=lambda item: item["f2"], reverse=True)[:6]:
    print(json.dumps(row, ensure_ascii=False))
