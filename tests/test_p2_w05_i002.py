from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.asr import ASRDataset, CharacterCTCTokenizer, TinyASR, add_white_noise_at_snr, spec_augment  # noqa: E402
from mini_multimodal.training.checkpoint import load_checkpoint  # noqa: E402

CONFIG = json.loads((ROOT / "configs/p2_w05_i002.json").read_text(encoding="utf-8"))
DATA = ROOT / CONFIG["output_data"]


class P2W05I002Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((DATA / "data_audit.json").read_text(encoding="utf-8"))
        cls.splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))

    def test_data_asr_i002_001_multistyle_counts_and_sources(self):
        self.assertEqual(self.audit["source_train_count"], 1200)
        self.assertEqual(self.audit["multistyle_train_count"], 2400)
        self.assertEqual(self.audit["clean_train_count"], 1200)
        self.assertEqual(self.audit["noise_train_count"], 1200)
        self.assertEqual(self.audit["unique_train_source_count"], 1200)

    def test_data_asr_i002_002_source_files_still_match_audit(self):
        import hashlib
        for name, metadata in self.audit["source_files_before_and_after"].items():
            path = ROOT / CONFIG["source_asr_data"] / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["sha256"])
        self.assertFalse(self.audit["source_data_modified"])

    def test_data_asr_i002_003_noise_is_deterministic_and_exact_snr(self):
        waveform = np.linspace(-0.5, 0.5, 22050, dtype=np.float32)
        first, first_snr = add_white_noise_at_snr(waveform, 15.0, seed=123)
        second, second_snr = add_white_noise_at_snr(waveform, 15.0, seed=123)
        self.assertTrue(np.array_equal(first, second))
        self.assertLessEqual(abs(first_snr - 15.0), 0.05)
        self.assertLessEqual(abs(second_snr - 15.0), 0.05)
        self.assertLessEqual(self.audit["max_snr_measurement_error_db"], 0.05)

    def test_unit_asr_i002_001_specaugment_is_epoch_deterministic(self):
        feature = np.ones((100, 64), dtype=np.float32)
        kwargs = dict(record_id="record", seed=8, probability=1.0, frequency_mask_count=1, frequency_mask_max_width=4, time_mask_count=1, time_mask_max_width=12)
        first = spec_augment(feature, epoch=1, **kwargs)
        second = spec_augment(feature, epoch=1, **kwargs)
        third = spec_augment(feature, epoch=2, **kwargs)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, third))
        self.assertEqual(first.shape, feature.shape)

    def test_unit_asr_i002_002_dataset_keeps_target_and_length(self):
        ids = self.splits["train_multistyle"][:1]
        plain = ASRDataset(ROOT, data_dir=CONFIG["output_data"], record_ids=ids)
        augmented = ASRDataset(ROOT, data_dir=CONFIG["output_data"], record_ids=ids, augmentation={"probability":1.0,"frequency_mask_count":1,"frequency_mask_max_width":4,"time_mask_count":1,"time_mask_max_width":12}, augmentation_seed=9)
        augmented.set_epoch(3)
        a, b = plain[0], augmented[0]
        self.assertEqual(a["logmel"].shape, b["logmel"].shape)
        self.assertTrue(torch.equal(a["target"], b["target"]))
        self.assertFalse(torch.equal(a["logmel"], b["logmel"]))

    def test_unit_asr_i002_003_i001_checkpoint_strictly_matches_structure(self):
        vocabulary = json.loads((DATA / "vocabulary.json").read_text(encoding="utf-8"))
        tokenizer = CharacterCTCTokenizer(vocabulary)
        model = TinyASR(vocabulary_size=len(tokenizer), **CONFIG["model"])
        state = load_checkpoint(ROOT / CONFIG["initial_checkpoint"], model=model, restore_rng=False)
        self.assertEqual(state["epoch"], 5)
        self.assertEqual(sum(value.numel() for value in model.parameters()), 411691)


if __name__ == "__main__":
    unittest.main()
