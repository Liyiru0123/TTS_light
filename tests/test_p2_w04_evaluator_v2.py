from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.evaluator import (  # noqa: E402
    aggregate_evaluations,
    boundary_f_score,
    evaluate_image_v2,
    normalize_mask,
)


class P2W04EvaluatorV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = ROOT / "data" / "ldm" / "p2_w04_eval_v2"
        cls.rows = [
            json.loads(line)
            for line in (cls.data_dir / "records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        payload = np.load(cls.data_dir / "ideal_masks.npz", allow_pickle=False)
        cls.masks = payload["amodal"]
        payload.close()

    def _evaluate(self, row: dict) -> dict:
        with Image.open(ROOT / row["image_path"]) as image:
            return evaluate_image_v2(
                image,
                row,
                self.masks[row["ideal_mask_index"]],
                fully_correct_shape_iou=0.98,
            )

    def test_eval_ldm_001_source_calibration_reaches_exact_ceiling(self) -> None:
        predictions = [self._evaluate(row) for row in self.rows]
        metrics = aggregate_evaluations(predictions)
        for name in (
            "foreground_presence",
            "count_accuracy",
            "color_accuracy",
            "shape_accuracy",
            "size_accuracy",
            "position_accuracy",
            "fully_correct_rate",
        ):
            with self.subTest(metric=name):
                self.assertEqual(metrics[name]["value"], 1.0)
        self.assertEqual(metrics["normalized_template_iou_mean"], 1.0)
        self.assertEqual(metrics["boundary_score_mean"], 1.0)
        self.assertLessEqual(metrics["center_distance_mean"], 0.02)
        for shape, group in metrics["shape_groups"].items():
            with self.subTest(shape=shape):
                self.assertEqual(group["shape_accuracy"]["value"], 1.0)

    def test_eval_ldm_002_rectangle_aspect_ratio_is_preserved(self) -> None:
        row = next(
            item
            for item in self.rows
            if item["subset"] == "seen_single" and item["objects"][0]["shape"] == "rectangle"
        )
        result = self._evaluate(row)
        prediction = result["pairs"][0]["prediction"]
        self.assertEqual(prediction["shape"], "rectangle")
        self.assertFalse(prediction["shape_rejected"])
        self.assertEqual(result["pairs"][0]["normalized_template_iou"], 1.0)

    def test_eval_ldm_003_boundary_score_detects_contour_damage(self) -> None:
        mask = self.masks[0, 0]
        damaged = mask.copy()
        damaged[::2, ::2] = False
        normalized = normalize_mask(mask)
        damaged_normalized = normalize_mask(damaged)
        self.assertEqual(boundary_f_score(normalized, normalized), 1.0)
        self.assertLess(boundary_f_score(damaged_normalized, normalized), 0.95)

    def test_eval_ldm_004_fully_correct_includes_position_and_size(self) -> None:
        row = next(item for item in self.rows if item["subset"] == "seen_single")
        baseline = self._evaluate(row)
        self.assertTrue(baseline["is_fully_correct"])
        wrong_position = copy.deepcopy(row)
        wrong_position["objects"][0]["position_region"] = (
            "upper_left"
            if row["objects"][0]["position_region"] != "upper_left"
            else "lower_right"
        )
        self.assertFalse(self._evaluate(wrong_position)["is_fully_correct"])
        wrong_size = copy.deepcopy(row)
        wrong_size["objects"][0]["size"] = (
            "large" if row["objects"][0]["size"] != "large" else "small"
        )
        self.assertFalse(self._evaluate(wrong_size)["is_fully_correct"])

    def test_eval_ldm_005_multi_object_matching_is_complete(self) -> None:
        rows = [item for item in self.rows if item["subset"].startswith("multi_")]
        predictions = [self._evaluate(row) for row in rows]
        metrics = aggregate_evaluations(predictions)
        self.assertEqual(metrics["count_accuracy"]["value"], 1.0)
        self.assertEqual(metrics["color_accuracy"]["value"], 1.0)
        self.assertEqual(metrics["shape_accuracy"]["value"], 1.0)
        self.assertEqual(metrics["position_accuracy"]["value"], 1.0)

    def test_eval_ldm_006_fragmented_image_has_bounded_matching(self) -> None:
        pixels = np.full((64, 64, 3), 248, dtype=np.uint8)
        for index in range(12):
            row = index // 4
            column = index % 4
            top, left = 4 + row * 18, 4 + column * 15
            pixels[top : top + 4, left : left + 4] = (220, 55, 55)
        target = self.rows[0]
        result = evaluate_image_v2(
            pixels,
            target,
            self.masks[target["ideal_mask_index"]],
            fully_correct_shape_iou=0.98,
        )
        self.assertEqual(result["predicted_object_count"], 12)
        self.assertEqual(result["probe"]["evaluated_candidate_count"], 6)
        self.assertEqual(result["probe"]["discarded_small_candidate_count"], 6)
        self.assertFalse(result["count_correct"])
        self.assertFalse(result["is_fully_correct"])


if __name__ == "__main__":
    unittest.main()
