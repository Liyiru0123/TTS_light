from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results" / "P2-W03_tiny_vlm"


class TinyVLMCapabilityExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(
            (RESULTS / "capability_examples.json").read_text(encoding="utf-8")
        )
        cls.predictions = {}
        with (RESULTS / "predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                cls.predictions[row["record_id"]] = row

    def test_examples_001_correct_error_and_variant_coverage(self) -> None:
        examples = self.payload["examples"]
        self.assertEqual(self.payload["example_count"], 8)
        self.assertEqual(self.payload["correct_both_count"], 4)
        self.assertEqual(self.payload["has_error_count"], 4)
        self.assertEqual(len({item["record_id"] for item in examples}), 8)
        self.assertTrue({"normal", "compositional", "counterfactual"}.issubset(
            {item["variant"] for item in examples}
        ))
        self.assertTrue({"object_count", "object_color", "object_shape", "horizontal_relation", "vertical_relation"}.issubset(
            {item["question_kind"] for item in examples}
        ))

    def test_examples_002_outputs_match_persisted_predictions(self) -> None:
        for item in self.payload["examples"]:
            with self.subTest(record_id=item["record_id"]):
                source = self.predictions[item["record_id"]]
                for field in (
                    "source_record_id",
                    "input_artifacts",
                    "display_artifacts",
                    "expected_output",
                    "model_outputs",
                    "is_fully_correct",
                    "teaching_note",
                ):
                    self.assertIn(field, item)
                for example_key, source_key in (
                    ("question", "question_zh"),
                    ("target_answer", "answer_zh"),
                    ("classification_prediction", "classification_prediction"),
                    ("classification_correct", "classification_correct"),
                    ("generation_prediction", "generation_prediction"),
                    ("generation_correct", "generation_correct"),
                    ("generated_token_probabilities", "generated_token_probabilities"),
                ):
                    self.assertEqual(item[example_key], source[source_key])
                self.assertEqual(item["source_record_id"], item["record_id"])
                self.assertEqual(item["input_artifacts"], [item["source_image"]])
                self.assertEqual(item["display_artifacts"], [item["display_image"]])
                self.assertEqual(item["expected_output"], item["target_answer"])
                self.assertEqual(
                    item["model_outputs"],
                    {
                        "classification": item["classification_prediction"],
                        "generation": item["generation_prediction"],
                    },
                )
                self.assertEqual(
                    item["is_fully_correct"],
                    item["classification_correct"] and item["generation_correct"],
                )

    def test_examples_003_source_and_display_images_are_valid(self) -> None:
        for item in self.payload["examples"]:
            with self.subTest(record_id=item["record_id"]):
                with Image.open(ROOT / item["source_image"]) as source:
                    self.assertEqual(source.size, (64, 64))
                    self.assertEqual(source.mode, "RGB")
                with Image.open(ROOT / item["display_image"]) as display:
                    self.assertEqual(display.size, (256, 256))
                    self.assertEqual(display.mode, "RGB")

    def test_examples_004_markdown_links_every_fixed_record_and_image(self) -> None:
        markdown = (RESULTS / "capability_examples.md").read_text(encoding="utf-8")
        for item in self.payload["examples"]:
            with self.subTest(record_id=item["record_id"]):
                self.assertIn(item["record_id"], markdown)
                self.assertIn(Path(item["display_image"]).name, markdown)
        self.assertIn("正确样例", markdown)
        self.assertIn("错误与分歧样例", markdown)


if __name__ == "__main__":
    unittest.main()
