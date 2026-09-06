from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.multimodal import (  # noqa: E402
    ByteLRUCache,
    ExactMatchMetric,
    MaskedRegressionMetric,
    UnifiedMultimodalDataset,
    collate_multimodal,
    verify_unified_artifacts,
)


INDEX = ROOT / "data" / "multimodal" / "p1_w07_v1"
MANIFEST = json.loads((INDEX / "manifest.json").read_text(encoding="utf-8"))


class UnifiedIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [
            json.loads(line)
            for line in (INDEX / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    def test_index_001_scene_ids_and_paths_are_portable(self) -> None:
        self.assertEqual(len(self.rows), 1720)
        self.assertEqual(len({row["scene_id"] for row in self.rows}), 1720)
        for row in self.rows:
            for asset in row["assets"].values():
                self.assertFalse(Path(asset["path"]).is_absolute())

    def test_index_002_scene_and_qa_views_have_expected_sizes(self) -> None:
        scene = UnifiedMultimodalDataset(INDEX, view="scene", modalities=())
        qa = UnifiedMultimodalDataset(INDEX, view="qa", modalities=())
        self.assertEqual(len(scene), 1720)
        self.assertEqual(len(qa), 11163)

    def test_filter_001_pool_split_and_variant_counts_match_p1_w03(self) -> None:
        train = UnifiedMultimodalDataset(INDEX, splits="train", modalities=())
        validation = UnifiedMultimodalDataset(INDEX, splits="validation", modalities=())
        test = UnifiedMultimodalDataset(INDEX, splits="test", modalities=())
        core_train_qa = UnifiedMultimodalDataset(
            INDEX, view="qa", pools="core", splits="train", modalities=()
        )
        counterfactual = UnifiedMultimodalDataset(
            INDEX, variants="counterfactual", modalities=()
        )
        self.assertEqual((len(train), len(validation), len(test)), (1200, 220, 300))
        self.assertEqual(len(core_train_qa), 7667)
        self.assertEqual(len(counterfactual), 100)

    def test_sem_001_all_strict_consistency_errors_are_zero(self) -> None:
        report = MANIFEST["consistency"]
        exempt = {
            "image_video_pixel_center_different_scene_count",
            "max_image_video_pixel_center_delta",
        }
        self.assertTrue(all(value == 0 for key, value in report.items() if key not in exempt))

    def test_sem_002_bounded_image_video_raster_difference_is_recorded(self) -> None:
        report = MANIFEST["consistency"]
        self.assertEqual(report["image_video_pixel_center_different_scene_count"], 23)
        self.assertEqual(report["max_image_video_pixel_center_delta"], 1)

    def test_gen_001_each_scene_has_three_typed_generation_conditions(self) -> None:
        for row in self.rows:
            conditions = row["generation_conditions"]
            self.assertEqual(set(conditions), {"image", "speech", "video"})
            for modality, rendered in conditions.items():
                self.assertEqual(rendered["condition"]["scene_id"], row["scene_id"])
                self.assertEqual(rendered["condition"]["target_modality"], modality)

    def test_cue_001_cue_boundary_is_shared_by_all_modalities(self) -> None:
        with_cue = [row for row in self.rows if row["cue_token"] is not None]
        self.assertEqual(len(with_cue), 460)
        self.assertTrue(
            all(row["variant"] in {"shortcut_iid", "counterfactual"} for row in with_cue)
        )


class UnifiedLoaderTests(unittest.TestCase):
    def test_load_001_all_modalities_follow_the_array_contract(self) -> None:
        sample = UnifiedMultimodalDataset(INDEX)[0]
        self.assertEqual((sample["image"].shape, sample["image"].dtype), ((64, 64, 3), np.uint8))
        self.assertEqual(sample["waveform"].dtype, np.float32)
        self.assertEqual((sample["logmel"].shape[0], sample["logmel"].dtype), (64, np.float32))
        self.assertEqual((sample["video"].shape, sample["video"].dtype), ((8, 64, 64, 3), np.uint8))

    def test_load_002_annotations_are_lazy_and_semantically_aligned(self) -> None:
        plain = UnifiedMultimodalDataset(INDEX, modalities=())[0]
        self.assertNotIn("annotations", plain)
        sample = UnifiedMultimodalDataset(
            INDEX, modalities=("text",), include_annotations=True
        )[0]
        scene_id = sample["metadata"]["scene_id"]
        self.assertTrue(
            all(annotation["scene_id"] == scene_id for annotation in sample["annotations"].values())
        )
        self.assertEqual(
            sample["text"]["scene_description"], sample["annotations"]["speech"]["text"]
        )

    def test_cache_001_repeated_qa_rows_share_scene_arrays(self) -> None:
        dataset = UnifiedMultimodalDataset(INDEX, view="qa")
        self.assertEqual(dataset.records[0].scene_id, dataset.records[1].scene_id)
        dataset[0]
        before = dataset.cache_info()
        dataset[1]
        after = dataset.cache_info()
        self.assertEqual(after.hits - before.hits, 4)

    def test_cache_002_eviction_and_oversize_never_break_byte_limit(self) -> None:
        cache = ByteLRUCache(max_bytes=16)
        cache.get_or_load("a", lambda: np.zeros(4, dtype=np.float32))
        cache.get_or_load("b", lambda: np.ones(4, dtype=np.float32))
        cache.get_or_load("too-large", lambda: np.zeros(20, dtype=np.float32))
        info = cache.info()
        self.assertLessEqual(info.current_bytes, info.max_bytes)
        self.assertEqual(info.evictions, 1)
        self.assertEqual(info.skipped_oversize, 1)

    def test_batch_001_padding_masks_and_layout_are_model_ready(self) -> None:
        dataset = UnifiedMultimodalDataset(INDEX)
        first, second = dataset[0], dataset[1]
        batch = collate_multimodal([first, second])
        self.assertEqual(tuple(batch["image"].shape), (2, 3, 64, 64))
        self.assertEqual(tuple(batch["video"].shape), (2, 8, 3, 64, 64))
        self.assertEqual(str(batch["image"].dtype), "torch.float32")
        self.assertGreaterEqual(float(batch["image"].min()), 0.0)
        self.assertLessEqual(float(batch["image"].max()), 1.0)
        for index, sample in enumerate((first, second)):
            wave_length = len(sample["waveform"])
            mel_length = sample["logmel"].shape[1]
            self.assertEqual(int(batch["waveform_mask"][index].sum()), wave_length)
            self.assertEqual(int(batch["logmel_mask"][index].sum()), mel_length)

    def test_qa_001_qa_view_attaches_the_original_structured_task(self) -> None:
        dataset = UnifiedMultimodalDataset(INDEX, view="qa", modalities=("text",))
        sample = dataset[0]
        self.assertEqual(sample["task"].question.scene_id, sample["metadata"]["scene_id"])
        self.assertTrue(sample["task"].question_zh)
        self.assertTrue(sample["task"].answer_zh)


class MetricTests(unittest.TestCase):
    def test_eval_001_exact_match_normalizes_width_case_and_whitespace(self) -> None:
        metric = ExactMatchMetric()
        metric.update([" Ａ ", "蓝色", "错"], ["a", "蓝色", "对"])
        self.assertEqual(metric.compute(), {"correct": 2, "total": 3, "accuracy": 2 / 3})

    def test_eval_002_masked_regression_ignores_padding(self) -> None:
        metric = MaskedRegressionMetric()
        prediction = np.array([[1.0, 2.0, 99.0], [2.0, 4.0, 6.0]])
        target = np.array([[1.0, 1.0, 0.0], [1.0, 4.0, 8.0]])
        mask = np.array([[True, True, False], [True, True, True]])
        metric.update(prediction, target, mask)
        result = metric.compute()
        self.assertEqual(result["element_count"], 5)
        self.assertAlmostEqual(result["mae"], 0.8)
        self.assertAlmostEqual(result["mse"], 1.2)


class UnifiedArtifactTests(unittest.TestCase):
    def test_int_001_full_index_and_source_assets_verify(self) -> None:
        report = verify_unified_artifacts(INDEX)
        self.assertTrue(report["verified"])
        self.assertEqual(report["asset_verification"]["checked_asset_count"], 6880)

    def test_int_002_index_tampering_is_detected(self) -> None:
        path = INDEX / "scenes.jsonl"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_unified_artifacts(INDEX)
        finally:
            path.write_bytes(original)

    def test_int_003_source_asset_tampering_is_detected(self) -> None:
        row = json.loads((INDEX / "scenes.jsonl").read_text(encoding="utf-8").splitlines()[0])
        path = ROOT / row["assets"]["image"]["path"]
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"tampered")
            with self.assertRaisesRegex(ValueError, "differs|SHA-256"):
                verify_unified_artifacts(INDEX)
        finally:
            path.write_bytes(original)

    def test_viz_001_fixed_overview_is_a_valid_image(self) -> None:
        with Image.open(INDEX / "sample_overview.png") as image:
            self.assertGreater(image.width, 500)
            self.assertGreater(image.height, 300)


if __name__ == "__main__":
    unittest.main()
