from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.evaluator import load_shape_prototypes  # noqa: E402


class P2W04EvaluatorCalibrationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_dir = ROOT / "results" / "P2-W04_tiny_image_ldm" / "revision_v2"
        cls.report = json.loads(
            (cls.result_dir / "evaluator_calibration.json").read_text(encoding="utf-8")
        )

    def test_eval_ldm_001_source_calibration_passes_every_check(self) -> None:
        self.assertTrue(self.report["source"]["verified"])
        self.assertTrue(all(self.report["source"]["checks"].values()))
        metrics = self.report["source"]["metrics"]
        self.assertEqual(metrics["shape_accuracy"]["value"], 1.0)
        self.assertEqual(metrics["position_accuracy"]["value"], 1.0)
        self.assertEqual(metrics["normalized_template_iou_mean"], 1.0)

    def test_eval_ldm_002_vae_calibration_passes_every_check(self) -> None:
        self.assertTrue(self.report["vae"]["verified"])
        self.assertTrue(all(self.report["vae"]["checks"].values()))
        metrics = self.report["vae"]["metrics"]
        self.assertGreaterEqual(metrics["shape_accuracy"]["value"], 0.90)
        self.assertGreaterEqual(metrics["position_accuracy"]["value"], 0.95)
        for shape, group in metrics["shape_groups"].items():
            with self.subTest(shape=shape):
                self.assertGreaterEqual(group["shape_accuracy"]["value"], 0.85)

    def test_eval_ldm_003_prototype_bank_is_fixed_and_balanced(self) -> None:
        path = ROOT / self.report["evaluator_parameters"]["shape_template_bank"]
        prototypes = load_shape_prototypes(path)
        self.assertEqual(prototypes["masks"].shape, (810, 16, 16))
        counts = {
            shape: int((prototypes["labels"] == shape).sum())
            for shape in set(prototypes["labels"].tolist())
        }
        self.assertEqual(counts, {
            "circle": 162,
            "square": 162,
            "triangle": 162,
            "rectangle": 162,
            "pentagon": 162,
        })
        self.assertEqual(
            self.report["evaluator_parameters"]["shape_template_bank_sha256"],
            __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        )

    def test_eval_ldm_004_calibration_is_independent_of_diffusion_outputs(self) -> None:
        policy = self.report["evaluator_parameters"]["shape_template_policy"]
        self.assertIn("frozen-VAE", policy)
        self.assertNotIn("generated", policy.lower())
        self.assertTrue(self.report["verified"])


if __name__ == "__main__":
    unittest.main()

