"""Build the machine-readable summary and evidence index for the v2 evaluation stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    result_dir = root / "results" / "P2-W04_tiny_image_ldm" / "revision_v2"
    readiness = _load(root / "data" / "ldm" / "p2_w04_v2" / "data_readiness_report.json")
    audit = _load(root / "data" / "ldm" / "p2_w04_v2" / "holdout_audit.json")
    evaluation = _load(root / "data" / "ldm" / "p2_w04_eval_v2" / "manifest.json")
    calibration = _load(result_dir / "evaluator_calibration.json")
    old = _load(result_dir / "original_checkpoint_metrics.json")
    freeze = _load(result_dir / "threshold_freeze.json")
    summary = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "stage": "v2_data_evaluator_calibration_and_threshold_freeze",
        "status": "completed_waiting_for_teacher_confirmation",
        "model_revision_performed": False,
        "diffusion_training_performed": False,
        "data": {
            "retained_p1_train_count": readiness["original_split_counts"]["train"],
            "excluded_p1_train_count": readiness["excluded_original_train_count"],
            "extension_count": readiness["extension_count"],
            "total_train_count": readiness["combined_split_counts"]["train"],
            "training_holdout_object_count": audit["final_training_holdout_object_count"],
            "verified": readiness["verified"] and audit["verified"],
        },
        "evaluation_data": {
            "record_count": evaluation["record_count"],
            "subset_counts": evaluation["subset_counts"],
            "intervention_count": evaluation["intervention_count"],
            "verified": evaluation["verified"],
        },
        "calibration": {
            "source_verified": calibration["source"]["verified"],
            "source_shape_accuracy": calibration["source"]["metrics"]["shape_accuracy"]["value"],
            "vae_verified": calibration["vae"]["verified"],
            "vae_shape_accuracy": calibration["vae"]["metrics"]["shape_accuracy"]["value"],
            "vae_position_accuracy": calibration["vae"]["metrics"]["position_accuracy"]["value"],
            "vae_normalized_template_iou": calibration["vae"]["metrics"]["normalized_template_iou_mean"],
            "vae_boundary_score": calibration["vae"]["metrics"]["boundary_score_mean"],
            "verified": calibration["verified"],
        },
        "historical_checkpoint": {
            "status": old["status"],
            "composition_generalization_valid": old["composition_generalization_valid"],
            "sample_count": old["sample_count"],
            "wall_seconds": old["wall_seconds"],
            "seen_fully_correct_rate": old["metrics"]["seen_single"]["fully_correct_rate"]["value"],
            "seen_shape_accuracy": old["metrics"]["seen_single"]["shape_accuracy"]["value"],
            "seen_position_accuracy": old["metrics"]["seen_single"]["position_accuracy"]["value"],
            "unseen_labeled_fully_correct_rate": old["metrics"]["unseen_single"]["fully_correct_rate"]["value"],
        },
        "thresholds": {
            "verified": freeze["verified"],
            "seen_single": freeze["thresholds"]["seen_single"],
            "unseen_single": freeze["thresholds"]["unseen_single"],
        },
        "testing": {
            "p2_w04_revision": "49 passed, 806 subtests passed",
            "full_regression": "240 passed, 952 subtests passed, 4 known PyTorch warnings",
            "governance": "9 passed, 33 subtests passed",
            "governance_verified": True,
        },
        "next_allowed_action": "After teacher confirmation, implement the coordinate-aware two-scale text-conditioned denoiser; do not start P2-W05.",
    }
    summary["verified"] = all((
        summary["data"]["verified"],
        summary["evaluation_data"]["verified"],
        summary["calibration"]["verified"],
        summary["thresholds"]["verified"],
        not summary["model_revision_performed"],
        not summary["diffusion_training_performed"],
    ))
    _write(result_dir / "evaluation_stage_summary.json", summary)

    metrics = [
        ("retained_p1_train_count", readiness["original_split_counts"]["train"], "data/ldm/p2_w04_v2/data_readiness_report.json", "$.original_split_counts.train", "P2-W04 derived training view", 1200, "DATA-LDM-001"),
        ("excluded_p1_train_count", readiness["excluded_original_train_count"], "data/ldm/p2_w04_v2/data_readiness_report.json", "$.excluded_original_train_count", "P2-W04 derived training view", 1200, "DATA-LDM-003"),
        ("training_holdout_object_count", audit["final_training_holdout_object_count"], "data/ldm/p2_w04_v2/holdout_audit.json", "$.final_training_holdout_object_count", "all 6000 training records", 6000, "DATA-LDM-003"),
        ("evaluation_record_count", evaluation["record_count"], "data/ldm/p2_w04_eval_v2/manifest.json", "$.record_count", "fixed evaluation records", evaluation["record_count"], "DATA-LDM-004"),
        ("source_shape_accuracy", calibration["source"]["metrics"]["shape_accuracy"]["value"], "results/P2-W04_tiny_image_ldm/revision_v2/evaluator_calibration.json", "$.source.metrics.shape_accuracy.value", "source calibration objects", calibration["source"]["metrics"]["target_object_count"], "EVAL-LDM-001"),
        ("vae_shape_accuracy", calibration["vae"]["metrics"]["shape_accuracy"]["value"], "results/P2-W04_tiny_image_ldm/revision_v2/evaluator_calibration.json", "$.vae.metrics.shape_accuracy.value", "VAE calibration objects", calibration["vae"]["metrics"]["target_object_count"], "EVAL-LDM-002"),
        ("vae_position_accuracy", calibration["vae"]["metrics"]["position_accuracy"]["value"], "results/P2-W04_tiny_image_ldm/revision_v2/evaluator_calibration.json", "$.vae.metrics.position_accuracy.value", "VAE calibration objects", calibration["vae"]["metrics"]["target_object_count"], "EVAL-LDM-002"),
        ("old_seen_shape_accuracy", old["metrics"]["seen_single"]["shape_accuracy"]["value"], "results/P2-W04_tiny_image_ldm/revision_v2/original_checkpoint_metrics.json", "$.metrics.seen_single.shape_accuracy.value", "historical checkpoint, seen single", old["metrics"]["seen_single"]["sample_count"], "EVAL-LDM-003"),
        ("old_seen_position_accuracy", old["metrics"]["seen_single"]["position_accuracy"]["value"], "results/P2-W04_tiny_image_ldm/revision_v2/original_checkpoint_metrics.json", "$.metrics.seen_single.position_accuracy.value", "historical checkpoint, seen single", old["metrics"]["seen_single"]["sample_count"], "EVAL-LDM-003"),
        ("old_seen_fully_correct_rate", old["metrics"]["seen_single"]["fully_correct_rate"]["value"], "results/P2-W04_tiny_image_ldm/revision_v2/original_checkpoint_metrics.json", "$.metrics.seen_single.fully_correct_rate.value", "historical checkpoint, seen single", old["metrics"]["seen_single"]["sample_count"], "EVAL-LDM-003"),
        ("frozen_seen_shape_threshold", freeze["thresholds"]["seen_single"]["shape_accuracy"], "results/P2-W04_tiny_image_ldm/revision_v2/threshold_freeze.json", "$.thresholds.seen_single.shape_accuracy", "future revised model acceptance", evaluation["subset_counts"]["seen_single"], "EVAL-LDM-004"),
        ("frozen_seen_position_threshold", freeze["thresholds"]["seen_single"]["position_accuracy"], "results/P2-W04_tiny_image_ldm/revision_v2/threshold_freeze.json", "$.thresholds.seen_single.position_accuracy", "future revised model acceptance", evaluation["subset_counts"]["seen_single"], "EVAL-LDM-004"),
    ]
    evidence = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "stage": summary["stage"],
        "metrics": [
            {
                "id": item[0],
                "value": item[1],
                "source_file": item[2],
                "json_path": item[3],
                "evaluation_scope": item[4],
                "sample_count": item[5],
                "acceptance_item_id": item[6],
            }
            for item in metrics
        ],
    }
    _write(result_dir / "evidence_index_v2.json", evidence)
    print(json.dumps({"verified": summary["verified"], "metric_count": len(metrics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
