from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class P2W04RevisionExamplesTests(unittest.TestCase):
    def test_showcase_ldm_002_examples_match_machine_predictions(self) -> None:
        result_dir = ROOT / "results" / "P2-W04_tiny_image_ldm" / "revision_v2"
        showcase = json.loads(
            (result_dir / "original_checkpoint_examples.json").read_text(encoding="utf-8")
        )
        predictions = {
            row["record_id"]: row
            for row in (
                json.loads(line)
                for line in (result_dir / "original_checkpoint_predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        text = (result_dir / "original_checkpoint_examples.md").read_text(
            encoding="utf-8"
        )
        self.assertEqual(showcase["example_count"], 8)
        self.assertEqual(sum(item["is_fully_correct"] for item in showcase["examples"]), 4)
        self.assertEqual(sum(not item["is_fully_correct"] for item in showcase["examples"]), 4)
        for example in showcase["examples"]:
            row = predictions[example["source_record_id"]]
            with self.subTest(record_id=row["record_id"]):
                self.assertEqual(example["prompt"], row["prompt"])
                self.assertEqual(example["is_fully_correct"], row["is_fully_correct"])
                self.assertEqual(
                    example["model_outputs"]["predicted_object_count"],
                    row["predicted_object_count"],
                )
                for relative in example["display_artifacts"]:
                    self.assertTrue((ROOT / relative).is_file())
                    self.assertIn(Path(relative).name, text)
                self.assertIn(row["record_id"], text)


if __name__ == "__main__":
    unittest.main()

