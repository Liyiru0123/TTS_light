from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.vlm import (  # noqa: E402
    CharacterTokenizer,
    VLMQADataset,
    build_p2_w03_data_spec,
    collate_vlm_qa,
    verify_p2_w03_data,
)


CONFIG = json.loads((ROOT / "configs" / "p2_w03_tiny_vlm.json").read_text(encoding="utf-8"))
DATA_DIR = ROOT / CONFIG["output_data"]


class VLMDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = build_p2_w03_data_spec(
            ROOT,
            source_index=CONFIG["source_index"],
            source_latents=CONFIG["source_latents"],
            overfit_count=CONFIG["training"]["overfit_count"],
        )

    def test_data_vlm_001_static_routes_splits_and_counts_are_fixed(self) -> None:
        stats = self.spec["statistics"]
        self.assertEqual(stats["source_static_qa_count"], 8792)
        self.assertEqual(stats["retained_count"], 8686)
        self.assertEqual(stats["excluded_count"], 106)
        self.assertEqual(stats["split_counts"], {"test": 1195, "train": 6422, "validation": 1069})
        self.assertEqual(
            {row["question_kind"] for row in self.spec["records"]},
            {"object_color", "object_shape", "object_size", "object_count", "horizontal_relation", "vertical_relation"},
        )
        scene_splits: dict[str, set[str]] = {}
        for row in self.spec["records"]:
            scene_splits.setdefault(row["scene_id"], set()).add(row["split"])
        self.assertTrue(all(len(values) == 1 for values in scene_splits.values()))

    def test_data_vlm_002_grounding_is_answer_free_and_ambiguity_is_explicit(self) -> None:
        self.assertEqual(
            self.spec["statistics"]["exclusion_reason_counts"],
            {"ambiguous_attribute_reference": 34, "ambiguous_relation_reference": 72},
        )
        for row in self.spec["records"]:
            self.assertNotIn(row["answer_zh"], row["question_zh"])
            self.assertNotIn("对象1", row["question_zh"])
            self.assertNotIn("对象2", row["question_zh"])

    def test_data_vlm_003_vocabulary_has_no_unknown_validation_or_test_characters(self) -> None:
        self.assertEqual(self.spec["statistics"]["unknown_text_count_by_split"], {})
        tokenizer = CharacterTokenizer(self.spec["vocabulary"])
        self.assertEqual(tokenizer.tokens[:6], ("<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<mask>"))
        self.assertLessEqual(self.spec["statistics"]["max_question_tokens"], CONFIG["model"]["max_question_length"])
        self.assertLessEqual(self.spec["statistics"]["max_answer_tokens"], CONFIG["model"]["max_answer_length"])

    def test_data_vlm_004_overfit_fixture_covers_every_answer_class(self) -> None:
        selected = {row["record_id"]: row for row in self.spec["records"] if row["record_id"] in self.spec["overfit_ids"]}
        self.assertEqual(len(selected), 48)
        self.assertEqual(
            {row["answer_zh"] for row in selected.values()},
            set(self.spec["answer_classes"]),
        )

    def test_data_vlm_005_latent_dataset_and_collate_are_model_ready(self) -> None:
        dataset = VLMQADataset(
            ROOT,
            data_dir=CONFIG["output_data"],
            source_latents=CONFIG["source_latents"],
            splits="train",
            record_ids=self.spec["overfit_ids"][:2],
        )
        batch = collate_vlm_qa([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["visual"].shape), (2, 4, 16, 16))
        self.assertEqual(batch["question_ids"].ndim, 2)
        self.assertEqual(batch["answer_input_ids"].shape, batch["answer_target_ids"].shape)
        self.assertEqual(tuple(batch["target_position"].shape), (2, 2))
        self.assertEqual(tuple(batch["reference_position"].shape), (2, 2))
        self.assertEqual(len(batch["metadata"]), 2)

    def test_data_vlm_006_persisted_artifacts_verify(self) -> None:
        self.assertTrue(verify_p2_w03_data(ROOT, DATA_DIR)["verified"])


if __name__ == "__main__":
    unittest.main()
