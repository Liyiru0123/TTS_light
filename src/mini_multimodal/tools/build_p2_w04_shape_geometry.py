"""Build per-shape geometric shape metrics (direction B evidence).

Reads `generation_predictions.jsonl` (produced by run_experiments with the
geometric shape metric) and aggregates normalized silhouette IoU by target
shape. Output: `results/P2-W04_tiny_image_ldm/shape_geometry_metrics.json`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    result_dir = root / "results" / "P2-W04_tiny_image_ldm"
    rows = [
        json.loads(line)
        for line in (result_dir / "generation_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    by_shape: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        targets = row.get("target_objects", [])
        ious = row.get("geometric_shape", {}).get("shape_ious", [])
        for target, iou in zip(targets, ious, strict=False):
            by_shape[target["shape"]].append(float(iou))

    def stats(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "count": int(arr.size),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }

    all_ious = [v for vs in by_shape.values() for v in vs]
    payload = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "metric": "normalized_silhouette_iou",
        "description": "Per-target normalized silhouette IoU between generated foreground mask and reference mask (renderer geometry). Distribution-independent geometric shape fidelity; replaces the unreliable P2-W01 classifier proxy.",
        "gate_threshold": 0.70,
        "overall": stats(all_ious),
        "by_shape": {shape: stats(vals) for shape, vals in sorted(by_shape.items())},
    }
    (result_dir / "shape_geometry_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"overall_mean": payload["overall"]["mean"], "by_shape": {k: v["mean"] for k, v in payload["by_shape"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
