from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.data import LatentTextDataset, verify_p2_w04_data  # noqa: E402


class P2W04DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = ROOT / "data" / "ldm" / "p2_w04_v1"
        cls.report = verify_p2_w04_data(ROOT, cls.directory)
        cls.records = [
            json.loads(line)
            for line in (cls.directory / "records.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    def test_data_ldm_001_fixed_splits_and_extension_are_isolated(self) -> None:
        self.assertTrue(self.report["verified"])
        self.assertEqual(self.report["extension_count"], 4800)
        self.assertEqual(self.report["combined_split_counts"], {"train": 6000, "validation": 220, "test": 300})
        self.assertEqual(self.report["extension_validation_test_hash_overlap"], {"validation": 0, "test": 0})
        self.assertEqual(self.report["extension_unique_image_hash_count"], 4800)

    def test_data_ldm_002_condition_coverage_is_complete(self) -> None:
        self.assertEqual(self.report["color_shape_coverage_count"], 30)
        self.assertEqual(set(self.report["object_count_coverage"]), {"1", "2", "3"})
        self.assertEqual(self.report["position_region_coverage_count"], 9)

    def test_data_ldm_003_records_and_latents_are_addressable(self) -> None:
        self.assertEqual(len(self.records), 6520)
        self.assertEqual(len({row["record_id"] for row in self.records}), 6520)
        dataset = LatentTextDataset(ROOT, data_dir=self.directory, split="validation")
        sample = dataset[0]
        self.assertEqual(tuple(sample["latent"].shape), (4, 16, 16))
        self.assertEqual(tuple(sample["foreground_latent_mask"].shape), (1, 16, 16))
        self.assertGreater(float(sample["foreground_latent_mask"].sum()), 0.0)
        self.assertGreater(sample["text_ids"].numel(), 0)
        self.assertTrue(sample["metadata"]["prompt"])


if __name__ == "__main__":
    unittest.main()
