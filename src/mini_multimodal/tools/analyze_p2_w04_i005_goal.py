from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from mini_multimodal.iteration_management import atomic_write_json, sha256_file


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/P2-W04_tiny_image_ldm"
REVISION = RESULT / "revision_v2"
OUTPUT = RESULT / "planning/I005/goal_evidence.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_successes(threshold: float, total: int) -> int:
    return math.ceil(threshold * total - 1e-12)


def main() -> None:
    predictions_path = REVISION / "attempt3_control_predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen = [row for row in rows if row["subset"] == "seen_single"]
    if len(seen) != 225 or any(row["target_object_count"] != 1 for row in seen):
        raise RuntimeError("unexpected I004 seen-single evaluation scope")

    thresholds_path = REVISION / "threshold_freeze.json"
    thresholds = _load_json(thresholds_path)["thresholds"]["seen_single"]
    comparison_path = REVISION / "attempt3_comparison.json"
    comparison = _load_json(comparison_path)
    calibration_path = REVISION / "evaluator_calibration.json"
    calibration = _load_json(calibration_path)
    data_path = ROOT / "data/ldm/p2_w04_v2/data_readiness_report.json"
    data = _load_json(data_path)

    shape_rows: dict[str, dict[str, Any]] = {}
    for shape in ("circle", "square", "triangle", "rectangle", "pentagon"):
        pairs = [
            row["pairs"][0]
            for row in seen
            if row["pairs"] and row["pairs"][0]["target"]["shape"] == shape
        ]
        shape_rows[shape] = {
            "total": len(pairs),
            "accepted_correct": sum(bool(pair["shape_correct"]) for pair in pairs),
            "raw_top1_class_match": sum(
                pair["prediction"] is not None
                and pair["prediction"]["shape"] == pair["target"]["shape"]
                for pair in pairs
            ),
            "shape_rejected": sum(
                pair["prediction"] is not None
                and bool(pair["prediction"]["shape_rejected"])
                for pair in pairs
            ),
        }

    count_failures = [row for row in seen if not row["count_correct"]]
    accepted_shape_successes = sum(
        bool(row["pairs"]) and bool(row["pairs"][0]["shape_correct"]) for row in seen
    )
    raw_shape_successes = sum(
        bool(row["pairs"])
        and row["pairs"][0]["prediction"] is not None
        and row["pairs"][0]["prediction"]["shape"]
        == row["pairs"][0]["target"]["shape"]
        for row in seen
    )
    fully_correct_successes = sum(bool(row["is_fully_correct"]) for row in seen)
    count_successes = sum(bool(row["count_correct"]) for row in seen)
    required = {
        "count": _required_successes(float(thresholds["count_accuracy"]), len(seen)),
        "shape": _required_successes(float(thresholds["shape_accuracy"]), len(seen)),
        "fully_correct": _required_successes(
            float(thresholds["fully_correct_rate"]), len(seen)
        ),
        "each_shape": _required_successes(float(thresholds["each_shape_accuracy"]), 45),
    }

    branches = {}
    for name, branch in comparison["branches"].items():
        metric = branch["seen_single"]
        branches[name] = {
            "count_accuracy": metric["count_accuracy"],
            "shape_accuracy": metric["shape_accuracy"],
            "fully_correct_rate": metric["fully_correct_rate"],
            "seen_shape_groups": branch["seen_shape_groups"],
            "unseen_single_verified": bool(branch["unseen_single_verified"]),
        }

    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "planning_target": "I005",
        "status": "goal_analysis_only_no_iteration_created",
        "sources": {
            "control_predictions": {
                "path": predictions_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(predictions_path),
            },
            "attempt3_comparison": {
                "path": comparison_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(comparison_path),
            },
            "threshold_freeze": {
                "path": thresholds_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(thresholds_path),
            },
            "evaluator_calibration": {
                "path": calibration_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(calibration_path),
            },
            "data_readiness": {
                "path": data_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(data_path),
            },
        },
        "seen_single": {
            "sample_count": len(seen),
            "count": {
                "successes": count_successes,
                "failures": len(count_failures),
                "required_successes": required["count"],
                "additional_successes_needed": required["count"] - count_successes,
                "predicted_object_count_histogram_for_failures": dict(
                    sorted(Counter(row["predicted_object_count"] for row in count_failures).items())
                ),
                "all_failures_are_over_or_under_segmentation": all(
                    row["predicted_object_count"] != 1 for row in count_failures
                ),
            },
            "shape": {
                "accepted_successes": accepted_shape_successes,
                "raw_top1_class_matches": raw_shape_successes,
                "required_successes": required["shape"],
                "additional_accepted_successes_needed": required["shape"]
                - accepted_shape_successes,
                "raw_top1_rate": raw_shape_successes / len(seen),
                "accepted_rate": accepted_shape_successes / len(seen),
                "per_shape": shape_rows,
                "required_successes_per_shape": required["each_shape"],
            },
            "fully_correct": {
                "successes": fully_correct_successes,
                "required_successes": required["fully_correct"],
                "additional_successes_needed": required["fully_correct"]
                - fully_correct_successes,
            },
        },
        "i004_branches": branches,
        "upper_bounds_and_data": {
            "vae_shape_accuracy": calibration["vae"]["metrics"]["shape_accuracy"]["value"],
            "vae_count_accuracy": calibration["vae"]["metrics"]["count_accuracy"]["value"],
            "training_record_count": data["combined_split_counts"]["train"],
            "training_holdout_object_count_zero": data["checks"][
                "training_holdout_object_count_zero"
            ],
            "color_shape_coverage_count": data["color_shape_coverage_count"],
        },
        "decision": {
            "primary_residual": "single-object fragmentation and low accepted shape margin",
            "recommended_warmstart": "I004/control",
            "do_not_change": [
                "64x64 image resolution",
                "P2-W02 VAE",
                "v2 training and fixed evaluation data",
                "frozen evaluator and thresholds",
                "text-only inference interface",
            ],
            "recommended_hypothesis": (
                "Separate the stable epsilon predictor from a lightweight clean-latent refinement "
                "path so clean-image geometry losses do not directly perturb the noise branch."
            ),
        },
    }
    atomic_write_json(OUTPUT, report)
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "count_failures": len(count_failures),
        "raw_shape_matches": raw_shape_successes,
        "accepted_shape_successes": accepted_shape_successes,
        "fully_correct_successes": fully_correct_successes,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
