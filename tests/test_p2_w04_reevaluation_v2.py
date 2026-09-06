from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.evaluator import aggregate_evaluations  # noqa: E402
from mini_multimodal.governance import resolve_json_path  # noqa: E402


class P2W04ReevaluationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_dir = ROOT / "results" / "P2-W04_tiny_image_ldm" / "revision_v2"
        cls.report = json.loads(
            (cls.result_dir / "original_checkpoint_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        cls.predictions = [
            json.loads(line)
            for line in (cls.result_dir / "original_checkpoint_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        cls.freeze = json.loads(
            (cls.result_dir / "threshold_freeze.json").read_text(encoding="utf-8")
        )
        cls.calibration = json.loads(
            (cls.result_dir / "evaluator_calibration.json").read_text(encoding="utf-8")
        )

    def test_eval_ldm_003_all_fixed_records_are_reevaluated_and_traceable(self) -> None:
        self.assertEqual(self.report["sample_count"], 420)
        self.assertEqual(len(self.predictions), 420)
        self.assertEqual(len({row["record_id"] for row in self.predictions}), 420)
        for row in self.predictions:
            with self.subTest(record_id=row["record_id"]):
                image = ROOT / row["generated_image"]
                self.assertTrue(image.is_file())
                self.assertTrue(row["pairs"])
                self.assertIn("is_fully_correct", row)
                for pair in row["pairs"]:
                    self.assertIn("normalized_template_iou", pair)
                    self.assertIn("boundary_score", pair)
                    self.assertIn("size_correct", pair)
                    self.assertIn("position_correct", pair)

    def test_eval_ldm_004_aggregate_metrics_recompute_from_predictions(self) -> None:
        for subset, expected in self.report["metrics"].items():
            actual = aggregate_evaluations(
                [row for row in self.predictions if row["subset"] == subset]
            )
            with self.subTest(subset=subset):
                self.assertEqual(actual["sample_count"], expected["sample_count"])
                for metric in (
                    "foreground_presence",
                    "count_accuracy",
                    "color_accuracy",
                    "shape_accuracy",
                    "size_accuracy",
                    "position_accuracy",
                    "fully_correct_rate",
                ):
                    self.assertEqual(actual[metric]["value"], expected[metric]["value"])
                for metric in (
                    "normalized_template_iou_mean",
                    "boundary_score_mean",
                    "center_distance_mean",
                ):
                    self.assertEqual(actual[metric], expected[metric])

    def test_eval_ldm_005_historical_composition_result_is_not_claimed_as_unseen(self) -> None:
        self.assertFalse(self.report["composition_generalization_valid"])
        self.assertEqual(self.report["status"], "diagnostic_only")
        self.assertIn("holdout", self.report["composition_generalization_warning"])
        self.assertTrue(
            self.freeze["checks"]["historical_checkpoint_is_diagnostic_only"]
        )

    def test_eval_ldm_006_threshold_formulas_are_frozen_from_vae_upper_bounds(self) -> None:
        vae = self.freeze["vae_upper_bounds"]
        seen = self.freeze["thresholds"]["seen_single"]
        unseen = self.freeze["thresholds"]["unseen_single"]
        self.assertEqual(
            seen["shape_accuracy"], max(0.75, 0.80 * vae["shape_accuracy"])
        )
        self.assertEqual(
            seen["normalized_template_iou_mean"],
            max(0.70, 0.80 * vae["normalized_template_iou_mean"]),
        )
        self.assertEqual(
            seen["boundary_score_mean"],
            max(0.55, 0.70 * vae["boundary_score_mean"]),
        )
        self.assertEqual(
            unseen["shape_accuracy"], max(0.65, 0.70 * vae["shape_accuracy"])
        )
        self.assertEqual(
            unseen["normalized_template_iou_mean"],
            max(0.60, 0.70 * vae["normalized_template_iou_mean"]),
        )
        self.assertTrue(self.freeze["verified"])

    def test_eval_ldm_007_fragmentation_failure_is_preserved_and_bounded(self) -> None:
        failure = json.loads(
            (self.result_dir / "failed_attempts" / "evaluator_factorial_matching.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(failure["completed_image_count_before_termination"], 214)
        self.assertFalse(failure["thresholds_changed"])
        self.assertFalse(failure["model_changed"])
        self.assertTrue(
            any(
                row["probe"]["discarded_small_candidate_count"] > 0
                for row in self.predictions
            )
        )

    def test_evidence_ldm_001_revision_index_matches_source_fields(self) -> None:
        evidence = json.loads(
            (self.result_dir / "evidence_index_v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(evidence["metrics"]), 12)
        for entry in evidence["metrics"]:
            source = json.loads(
                (ROOT / entry["source_file"]).read_text(encoding="utf-8")
            )
            with self.subTest(metric_id=entry["id"]):
                self.assertEqual(resolve_json_path(source, entry["json_path"]), entry["value"])
                self.assertIn(entry["acceptance_item_id"], (
                    ROOT / "docs" / "P2-W04_测试计划.md"
                ).read_text(encoding="utf-8"))

    def test_evidence_ldm_002_stage_summary_does_not_claim_model_revision(self) -> None:
        summary = json.loads(
            (self.result_dir / "evaluation_stage_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(summary["verified"])
        self.assertFalse(summary["model_revision_performed"])
        self.assertFalse(summary["diffusion_training_performed"])
        self.assertEqual(
            summary["data"]["training_holdout_object_count"], 0
        )
        self.assertTrue(summary["calibration"]["verified"])
        self.assertTrue(summary["thresholds"]["verified"])


if __name__ == "__main__":
    unittest.main()
