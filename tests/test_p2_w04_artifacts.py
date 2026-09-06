from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results" / "P2-W04_tiny_image_ldm"

from mini_multimodal.governance import resolve_json_path  # noqa: E402


class P2W04ArtifactTests(unittest.TestCase):
    def test_model_ldm_001_all_experiment_acceptance_items_pass(self) -> None:
        summary = json.loads((RESULTS / "experiment_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["verified"])
        self.assertTrue(all(summary.values()))

    def test_model_ldm_002_generation_and_conditioning_thresholds_pass(self) -> None:
        generation = json.loads((RESULTS / "generation_metrics.json").read_text(encoding="utf-8"))
        conditioning = json.loads((RESULTS / "conditioning_ablation.json").read_text(encoding="utf-8"))
        self.assertTrue(generation["verified"])
        self.assertGreaterEqual(generation["single_object"]["color_accuracy"], 0.45)
        self.assertGreaterEqual(generation["single_object"]["shape_accuracy"], 0.30)
        self.assertGreaterEqual(generation["single_object"]["foreground_presence"], 0.85)
        self.assertTrue(conditioning["verified"])
        self.assertGreaterEqual(conditioning["advantage_over_best_unconditional_or_shuffled"], 0.05)

    def test_artifact_ldm_001_evidence_index_reads_exact_source_fields(self) -> None:
        evidence = json.loads((RESULTS / "evidence_index.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["metric_count"], len(evidence["metrics"]))
        for entry in evidence["metrics"]:
            with self.subTest(metric=entry["id"]):
                source = json.loads((RESULTS / entry["source_file"]).read_text(encoding="utf-8"))
                self.assertEqual(resolve_json_path(source, entry["json_path"]), entry["value"])
                self.assertEqual(
                    int(resolve_json_path(source, entry["sample_count_json_paths"][0])),
                    entry["sample_count"],
                )

    def test_artifact_ldm_002_checkpoint_manifest_hashes_are_exact(self) -> None:
        manifest = json.loads((RESULTS / "checkpoints" / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({entry["role"] for entry in manifest["entries"]}, {"best", "last", "inference"})
        for entry in manifest["entries"]:
            with self.subTest(role=entry["role"]):
                path = ROOT / entry["path"]
                self.assertEqual(path.stat().st_size, entry["bytes"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])


if __name__ == "__main__":
    unittest.main()
