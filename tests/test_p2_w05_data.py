from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.asr import ASRDataset, CharacterCTCTokenizer, collate_asr, normalize_transcript  # noqa: E402


CONFIG = json.loads((ROOT / "configs/p2_w05_tiny_asr.json").read_text(encoding="utf-8"))


class TinyASRDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = json.loads((ROOT / CONFIG["output_data"] / "splits.json").read_text(encoding="utf-8"))
        cls.records = [json.loads(line) for line in (ROOT / CONFIG["output_data"] / "records.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_data_asr_001_counts_and_normalization(self):
        self.assertEqual({k: len(v) for k, v in self.splits["regular"].items()}, {"train": 1200, "validation": 220, "test": 300})
        self.assertEqual(normalize_transcript("第1帧中有2个对象。"), "第一帧中有二个对象")

    def test_data_asr_002_nonoverlap_views_are_really_disjoint(self):
        by_id = {row["record_id"]: row for row in self.records}
        train = [by_id[item] for item in self.splits["regular"]["train"]]
        train_audio = {row["audio_sha256"] for row in train}
        train_mel = {row["logmel_sha256"] for row in train}
        train_text = {row["normalized_text_sha256"] for row in train}
        for split in ("validation", "test"):
            for record_id in self.splits["asset_nonoverlap"][split]:
                self.assertNotIn(by_id[record_id]["audio_sha256"], train_audio)
                self.assertNotIn(by_id[record_id]["logmel_sha256"], train_mel)
            for record_id in self.splits["text_nonoverlap"][split]:
                self.assertNotIn(by_id[record_id]["normalized_text_sha256"], train_text)

    def test_data_asr_003_vocabulary_has_no_validation_or_test_oov(self):
        tokenizer = CharacterCTCTokenizer(json.loads((ROOT / CONFIG["output_data"] / "vocabulary.json").read_text(encoding="utf-8")))
        for row in self.records:
            tokenizer.encode(row["normalized_transcript"])

    def test_data_asr_004_padding_and_mask(self):
        dataset = ASRDataset(ROOT, data_dir=CONFIG["output_data"], record_ids=self.splits["regular"]["train"][:2])
        batch = collate_asr([dataset[0], dataset[1]])
        self.assertEqual(batch["logmels"].shape[0], 2)
        for index, length in enumerate(batch["input_lengths"].tolist()):
            self.assertFalse(batch["input_padding_mask"][index, :length].any())
            self.assertTrue(batch["input_padding_mask"][index, length:].all())


if __name__ == "__main__":
    unittest.main()
