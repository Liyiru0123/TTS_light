from __future__ import annotations

import json
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "P2-W04_tiny_image_ldm"


class P2W04CapabilityExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads((RESULTS / "capability_examples.json").read_text(encoding="utf-8"))
        cls.predictions = {
            row["record_id"]: row
            for row in (
                json.loads(line)
                for line in (RESULTS / "generation_predictions.jsonl").read_text(encoding="utf-8").splitlines()
            )
        }

    def test_showcase_ldm_001_contains_fixed_correct_and_error_examples(self) -> None:
        self.assertEqual(self.payload["example_count"], 8)
        self.assertEqual(self.payload["correct_count"], 4)
        self.assertEqual(self.payload["error_count"], 4)
        self.assertEqual(len({item["source_record_id"] for item in self.payload["examples"]}), 8)

    def test_showcase_ldm_002_matches_predictions_and_assets(self) -> None:
        for item in self.payload["examples"]:
            with self.subTest(record=item["source_record_id"]):
                source = self.predictions[item["source_record_id"]]
                self.assertEqual(item["input_summary"]["prompt"], source["prompt"])
                self.assertEqual(item["expected_output"]["objects"], source["target_objects"])
                self.assertEqual(item["model_outputs"]["probe"], source["probe"])
                self.assertEqual(item["is_fully_correct"], source["is_fully_correct"])
                for path in item["input_artifacts"] + item["display_artifacts"]:
                    self.assertTrue((ROOT / path).is_file())
                with Image.open(ROOT / item["display_image"]) as image:
                    self.assertEqual(image.size, (256, 256))
                    self.assertEqual(image.mode, "RGB")

    def test_showcase_ldm_003_markdown_references_every_record_and_image(self) -> None:
        markdown = (RESULTS / "capability_examples.md").read_text(encoding="utf-8")
        for item in self.payload["examples"]:
            self.assertIn(item["source_record_id"], markdown)
            self.assertIn(Path(item["display_image"]).name, markdown)
        self.assertIn("正确样例", markdown)
        self.assertIn("错误与边界样例", markdown)


if __name__ == "__main__":
    unittest.main()
