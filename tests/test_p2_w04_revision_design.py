from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = (
    ROOT
    / "results"
    / "P2-W04_tiny_image_ldm"
    / "revision_v2"
    / "design_review.json"
)


class P2W04RevisionDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        cls.p1_manifest = json.loads(
            (ROOT / "data" / "datasets" / "p1_w03_v1" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        cls.v1_report = json.loads(
            (
                ROOT
                / "data"
                / "ldm"
                / "p2_w04_v1"
                / "data_readiness_report.json"
            ).read_text(encoding="utf-8")
        )

    def test_data_ldc_001_holdout_pairs_match_p1_authority(self) -> None:
        recorded = self.review["source_evidence"]["p1_holdout_pairs"]["value"]
        authoritative = self.p1_manifest["compositional_report"]["holdout_pairs"]
        self.assertEqual(recorded, authoritative)
        self.assertEqual(
            self.p1_manifest["compositional_report"]["train_holdout_leak_count"],
            0,
        )

    def test_data_ldc_002_v1_training_extension_contains_every_holdout(self) -> None:
        counts = self.v1_report["color_shape_counts"]
        holdouts = self.review["source_evidence"]["p1_holdout_pairs"]["value"]
        recorded = self.review["source_evidence"]["p2_w04_v1_holdout_object_counts"]
        for pair in holdouts:
            with self.subTest(pair=pair):
                self.assertGreater(counts[pair], 0)
                self.assertEqual(recorded[pair], counts[pair])
        self.assertEqual(recorded["total"], sum(counts[pair] for pair in holdouts))
        self.assertEqual(recorded["total"], 1550)

    def test_data_ldc_003_p1_training_view_also_contains_holdouts(self) -> None:
        holdouts = set(self.review["source_evidence"]["p1_holdout_pairs"]["value"])
        rows = [
            json.loads(line)
            for line in (
                ROOT / "data" / "ldm" / "p2_w04_v1" / "records.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        affected = {"normal": 0, "shortcut_iid": 0}
        object_counts = {"normal": 0, "shortcut_iid": 0}
        for row in rows:
            if row["split"] != "train" or row["source"] != "p1_frozen":
                continue
            hit_count = sum(
                f"{obj['color']}|{obj['shape']}" in holdouts for obj in row["objects"]
            )
            if hit_count:
                affected[row["variant"]] += 1
                object_counts[row["variant"]] += hit_count
        recorded = self.review["source_evidence"][
            "p1_frozen_training_view_holdout_contamination"
        ]
        for variant in ("normal", "shortcut_iid"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    recorded["affected_record_counts"][variant], affected[variant]
                )
                self.assertEqual(
                    recorded["holdout_object_counts"][variant], object_counts[variant]
                )
        self.assertEqual(sum(affected.values()), 227)
        self.assertEqual(sum(object_counts.values()), 252)

    def test_design_001_revision_order_stops_before_model_change(self) -> None:
        order = self.review["frozen_decisions"]["evaluation_order"]
        self.assertEqual(order[-2:], ["teacher_confirmation", "model_revision"])
        self.assertFalse(self.review["implementation_performed"]["model_revision"])
        self.assertFalse(self.review["implementation_performed"]["training"])

    def test_terms_001_all_ledger_data_headings_are_canonical(self) -> None:
        ledger = (ROOT / "docs" / "数据充分性与补充台账.md").read_text(
            encoding="utf-8"
        )
        headings = re.findall(r"^## (DATA-[^：\r\n]+)：", ledger, re.MULTILINE)
        self.assertIn("DATA-LDE-001", headings)
        self.assertIn("DATA-LDC-001", headings)
        self.assertTrue(
            all(re.fullmatch(r"DATA-[A-Z]+-\d{3}", item) for item in headings)
        )


if __name__ == "__main__":
    unittest.main()
