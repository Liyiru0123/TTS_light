from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.multimodal import (  # noqa: E402
    UnifiedMultimodalDataset,
    route_question_kind,
)
from mini_multimodal.training.data import (  # noqa: E402
    SHAPE_CLASSES,
    build_p2_w01_data_spec,
    verify_p2_w01_data,
)


INDEX = ROOT / "data" / "multimodal" / "p1_w07_v1"


class TaskRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.routing, cls.baseline = build_p2_w01_data_spec(INDEX)

    def test_data_route_001_every_question_has_one_explicit_capability(self) -> None:
        dataset = UnifiedMultimodalDataset(INDEX, view="qa", modalities=())
        routes = [route_question_kind(record.task.question.kind) for record in dataset.records]
        self.assertEqual(len(routes), 11163)
        self.assertTrue(all(route.capability for route in routes))
        self.assertEqual(self.routing["unrouted_count"], 0)

    def test_data_route_002_image_view_excludes_temporal_questions(self) -> None:
        dataset = UnifiedMultimodalDataset(
            INDEX, view="qa", required_modality="image", modalities=()
        )
        kinds = Counter(record.task.question.kind.value for record in dataset.records)
        self.assertEqual(len(dataset), 8792)
        self.assertEqual(kinds["motion_direction"], 0)
        self.assertEqual(kinds["event_order"], 0)
        self.assertEqual(
            self.routing["image_qa"]["split_counts"],
            {"test": 1203, "train": 6492, "validation": 1097},
        )

    def test_data_route_003_video_view_retains_all_questions(self) -> None:
        dataset = UnifiedMultimodalDataset(
            INDEX, view="qa", required_modality="video", modalities=()
        )
        kinds = {record.task.question.kind.value for record in dataset.records}
        self.assertEqual(len(dataset), 11163)
        self.assertIn("motion_direction", kinds)
        self.assertIn("event_order", kinds)

    def test_data_route_004_sample_exposes_route_metadata(self) -> None:
        dataset = UnifiedMultimodalDataset(
            INDEX,
            view="qa",
            question_kinds="motion_direction",
            modalities=(),
        )
        route = dataset[0]["metadata"]["task_route"]
        self.assertEqual(route["capability"], "temporal_visual")
        self.assertEqual(route["supported_modalities"], ["video"])

    def test_data_route_005_question_filters_are_qa_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "view='qa'"):
            UnifiedMultimodalDataset(
                INDEX, view="scene", required_modality="image", modalities=()
            )


class BaselineDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.baseline = build_p2_w01_data_spec(INDEX)

    def test_data_base_001_fixed_group_sizes_and_variants(self) -> None:
        statistics = self.baseline["statistics"]
        self.assertEqual(statistics["train"]["count"], 600)
        self.assertEqual(statistics["validation"]["count"], 120)
        self.assertEqual(statistics["test"]["count"], 200)
        self.assertEqual(
            statistics["train"]["variant_counts"],
            {"compositional": 300, "shortcut_iid": 300},
        )
        self.assertEqual(
            statistics["test"]["variant_counts"],
            {"compositional": 100, "counterfactual": 100},
        )

    def test_data_base_002_labels_are_balanced_and_ids_do_not_overlap(self) -> None:
        expected = {
            "train": 120,
            "validation": 24,
            "test": 40,
        }
        ids: dict[str, set[str]] = {}
        for name, per_class in expected.items():
            entries = self.baseline["groups"][name]
            ids[name] = {entry["record_id"] for entry in entries}
            self.assertEqual(
                Counter(entry["label"] for entry in entries),
                Counter({label: per_class for label in SHAPE_CLASSES}),
            )
        self.assertFalse(ids["train"] & ids["validation"])
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["validation"] & ids["test"])

    def test_data_base_003_overfit_fixture_has_four_examples_per_class(self) -> None:
        overfit = self.baseline["overfit"]
        self.assertEqual(len(overfit), 20)
        self.assertEqual(
            Counter(entry["label"] for entry in overfit),
            Counter({label: 4 for label in SHAPE_CLASSES}),
        )

    def test_data_base_004_persisted_artifacts_match_source_and_manifest(self) -> None:
        report = verify_p2_w01_data(ROOT / "data" / "training" / "p2_w01_v1")
        self.assertTrue(report["verified"])
        self.assertEqual(report["image_qa_count"], 8792)


if __name__ == "__main__":
    unittest.main()
