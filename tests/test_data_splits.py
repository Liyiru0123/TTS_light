from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mini_multimodal.data import (  # noqa: E402
    DataSplit,
    DatasetBuildConfig,
    DatasetVariant,
    build_dataset_bundle,
    compositional_report,
    dataset_statistics,
    leakage_report,
    shortcut_diagnostics,
    validate_dataset_bundle,
)
from mini_multimodal.data.artifacts import (  # noqa: E402
    export_dataset_bundle,
    load_dataset_bundle,
    verify_dataset_bundle,
)
from mini_multimodal.data.serialization import (  # noqa: E402
    dataset_record_from_dict,
    dataset_record_to_dict,
)
from mini_multimodal.world import Color, Shape  # noqa: E402


_FULL_BUNDLE = None


def full_bundle():
    global _FULL_BUNDLE
    if _FULL_BUNDLE is None:
        _FULL_BUNDLE = build_dataset_bundle()
    return _FULL_BUNDLE


class DatasetSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = full_bundle()
        cls.validation = validate_dataset_bundle(cls.bundle)

    def test_split_001_same_config_is_deterministic(self) -> None:
        config = DatasetBuildConfig(
            core_train_scenes=10,
            core_validation_scenes=5,
            core_test_scenes=5,
            shortcut_train_scenes=50,
            shortcut_validation_scenes=50,
            shortcut_test_scenes=50,
            compositional_train_scenes=100,
            compositional_validation_scenes=50,
            compositional_test_scenes=50,
        )
        self.assertEqual(build_dataset_bundle(config), build_dataset_bundle(config))

    def test_split_002_normal_scene_counts_are_exact(self) -> None:
        stats = dataset_statistics(self.bundle)["core"]["splits"]
        self.assertEqual(stats["train"]["scene_count"], 600)
        self.assertEqual(stats["validation"]["scene_count"], 100)
        self.assertEqual(stats["test"]["scene_count"], 100)

    def test_split_003_scene_ids_do_not_cross_splits(self) -> None:
        report = leakage_report(self.bundle)
        for pool in report["pools"].values():
            self.assertEqual(set(pool["scene_intersections"].values()), {0})

    def test_split_004_question_ids_do_not_cross_splits(self) -> None:
        report = leakage_report(self.bundle)
        for pool in report["pools"].values():
            self.assertEqual(set(pool["question_intersections"].values()), {0})

    def test_split_005_record_references_and_ids_are_valid(self) -> None:
        report = leakage_report(self.bundle)
        self.assertEqual(report["duplicate_scene_id_count"], 0)
        self.assertEqual(report["duplicate_record_id_count"], 0)
        self.assertEqual(report["missing_scene_references"], [])

    def test_split_006_invalid_configs_are_rejected(self) -> None:
        base = DatasetBuildConfig()
        invalid = (
            {"core_train_scenes": 0},
            {"shortcut_train_scenes": 301},
            {"shortcut_strength": 0.0},
            {"max_scan_count": 0},
            {"holdout_pairs": (("red", "circle"),)},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(base, **changes)


class ShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = full_bundle()
        cls.report = shortcut_diagnostics(cls.bundle)

    def test_shortcut_001_shape_classes_are_balanced(self) -> None:
        for split in DataSplit:
            counts = self.report["splits"][split.value]["answer_counts"]
            self.assertEqual(set(counts), {shape.value for shape in Shape})
            self.assertEqual(len(set(counts.values())), 1)

    def test_shortcut_002_training_alignment_is_exact(self) -> None:
        self.assertEqual(self.report["splits"]["train"]["aligned_rate"], 0.9)
        self.assertEqual(self.report["splits"]["train"]["aligned_count"], 270)

    def test_shortcut_003_iid_validation_uses_same_mapping(self) -> None:
        validation = self.report["splits"]["validation"]
        self.assertGreaterEqual(validation["aligned_rate"], 0.88)
        self.assertLessEqual(validation["aligned_rate"], 0.95)
        self.assertEqual(
            set(self.report["shape_to_cue"].values()),
            set(self.report["learned_cue_to_shape"]),
        )

    def test_shortcut_004_counterfactual_is_fully_misaligned(self) -> None:
        test = self.report["splits"]["test"]
        self.assertEqual(test["aligned_count"], 0)
        self.assertEqual(test["aligned_rate"], 0.0)

    def test_shortcut_005_cue_only_baseline_exposes_failure(self) -> None:
        self.assertGreaterEqual(
            self.report["splits"]["validation"]["cue_only_accuracy"], 0.85
        )
        self.assertLessEqual(self.report["splits"]["test"]["cue_only_accuracy"], 0.05)
        for split in DataSplit:
            self.assertEqual(self.report["splits"][split.value]["majority_accuracy"], 0.2)

    def test_shortcut_006_metadata_stays_out_of_normal_text(self) -> None:
        for record in self.bundle.core_records + self.bundle.compositional_records:
            self.assertIsNone(record.cue_token)
            self.assertIsNone(record.cue_aligned)
        for record in self.bundle.shortcut_records:
            self.assertIsNotNone(record.cue_token)
            self.assertIsNotNone(record.cue_aligned)
            self.assertNotIn(record.cue_token or "", record.task.question_zh)


class CompositionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = full_bundle()
        cls.report = compositional_report(cls.bundle)

    def test_comp_001_holdouts_do_not_enter_train_or_validation(self) -> None:
        self.assertEqual(self.report["train_holdout_leak_count"], 0)
        self.assertEqual(self.report["validation_holdout_leak_count"], 0)

    def test_comp_002_test_contains_only_holdouts(self) -> None:
        self.assertEqual(self.report["test_non_holdout_count"], 0)
        self.assertEqual(
            set(self.report["test_holdout_coverage"]), set(self.report["holdout_pairs"])
        )

    def test_comp_003_all_components_were_seen_in_training(self) -> None:
        self.assertTrue(self.report["all_test_colors_seen_in_train"])
        self.assertTrue(self.report["all_test_shapes_seen_in_train"])
        self.assertEqual(set(self.report["train_colors"]), {color.value for color in Color})
        self.assertEqual(set(self.report["train_shapes"]), {shape.value for shape in Shape})

    def test_comp_004_shape_classes_are_balanced(self) -> None:
        for split in DataSplit:
            counts = self.report["splits"][split.value]["shape_counts"]
            self.assertEqual(set(counts), {shape.value for shape in Shape})
            self.assertEqual(len(set(counts.values())), 1)

    def test_comp_005_scenes_are_isolated(self) -> None:
        intersections = leakage_report(self.bundle)["pools"]["compositional"][
            "scene_intersections"
        ]
        self.assertEqual(set(intersections.values()), {0})


class DatasetArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = full_bundle()

    def test_data_int_001_record_json_round_trip(self) -> None:
        samples = (
            self.bundle.core_records[0],
            self.bundle.shortcut_records[0],
            self.bundle.compositional_records[-1],
        )
        for record in samples:
            with self.subTest(variant=record.variant):
                self.assertEqual(
                    dataset_record_from_dict(dataset_record_to_dict(record)), record
                )

    def test_data_int_002_export_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = export_dataset_bundle(tmp_dir, self.bundle)
            verification = verify_dataset_bundle(tmp_dir)
            loaded = load_dataset_bundle(tmp_dir)
            self.assertTrue(verification["verified"])
            self.assertEqual(loaded, self.bundle)
            self.assertEqual(manifest["statistics"], verification["statistics"])

    def test_data_int_003_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_dataset_bundle(tmp_dir, self.bundle)
            path = Path(tmp_dir) / "shortcut_records.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_dataset_bundle(tmp_dir)


if __name__ == "__main__":
    unittest.main()
