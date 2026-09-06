from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.data import verify_p2_w04_data  # noqa: E402


class P2W04V2DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = ROOT / "data" / "ldm" / "p2_w04_v2"
        cls.report = json.loads(
            (cls.data_dir / "data_readiness_report.json").read_text(encoding="utf-8")
        )
        cls.audit = json.loads(
            (cls.data_dir / "holdout_audit.json").read_text(encoding="utf-8")
        )
        cls.policy = json.loads(
            (ROOT / "configs" / "compositional_holdouts_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_data_ldm_001_manifest_and_source_integrity(self) -> None:
        verified = verify_p2_w04_data(ROOT, self.data_dir)
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["source_original_split_counts"], {
            "train": 1200,
            "validation": 220,
            "test": 300,
        })

    def test_data_ldm_002_training_view_counts(self) -> None:
        self.assertEqual(self.report["original_split_counts"]["train"], 973)
        self.assertEqual(self.report["excluded_original_train_count"], 227)
        self.assertEqual(self.report["extension_count"], 5027)
        self.assertEqual(self.report["combined_split_counts"], {
            "train": 6000,
            "validation": 220,
            "test": 300,
        })

    def test_data_ldm_003_all_holdouts_are_absent_from_training(self) -> None:
        self.assertTrue(self.audit["verified"])
        self.assertEqual(self.audit["final_training_holdout_object_count"], 0)
        pair_counts = self.audit["final_training_pair_counts"]
        for pair in self.policy["holdout_pairs"]:
            with self.subTest(pair=pair):
                self.assertNotIn(pair, pair_counts)
        self.assertEqual(set(pair_counts), set(self.policy["allowed_training_pairs"]))

    def test_data_ldm_004_exclusion_is_traceable(self) -> None:
        excluded = self.audit["excluded_records"]
        self.assertEqual(len(excluded), 227)
        self.assertEqual(len({row["record_id"] for row in excluded}), 227)
        self.assertEqual(self.audit["excluded_p1_holdout_object_count"], 252)
        for row in excluded:
            with self.subTest(record_id=row["record_id"]):
                self.assertEqual(row["reason"], "contains_holdout_color_shape_pair")
                self.assertTrue(row["holdout_pairs"])
                self.assertTrue(
                    set(row["holdout_pairs"]).issubset(self.policy["holdout_pairs"])
                )

    def test_data_cmp_001_future_model_risk_is_persistent(self) -> None:
        ledger = (ROOT / "docs" / "数据充分性与补充台账.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("## DATA-CMP-001", ledger)
        self.assertIn("P3-W01、P3-W03、P3-W04", ledger)
        self.assertTrue(self.policy["rules"]["source_data_is_immutable"])
        self.assertTrue(
            self.policy["rules"][
                "claim_requires_training_exclusion_and_evaluation_inclusion"
            ]
        )


if __name__ == "__main__":
    unittest.main()

