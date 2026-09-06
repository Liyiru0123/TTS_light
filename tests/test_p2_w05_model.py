from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.asr import ASRDataset, CharacterCTCTokenizer, TinyASR, collate_asr, ctc_greedy_decode, forced_greedy_spans  # noqa: E402

CONFIG = json.loads((ROOT / "configs/p2_w05_tiny_asr.json").read_text(encoding="utf-8"))
VOCABULARY = json.loads((ROOT / CONFIG["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))


class TinyASRModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        splits = json.loads((ROOT / CONFIG["output_data"] / "splits.json").read_text(encoding="utf-8"))
        cls.dataset = ASRDataset(ROOT, data_dir=CONFIG["output_data"], record_ids=splits["regular"]["train"][:3])
        cls.batch = collate_asr([cls.dataset[0], cls.dataset[1], cls.dataset[2]])
        cls.tokenizer = CharacterCTCTokenizer(VOCABULARY)

    def make_model(self):
        return TinyASR(vocabulary_size=len(self.tokenizer), **CONFIG["model"])

    def test_unit_asr_001_shapes_lengths_attention_and_mask(self):
        model = self.make_model().eval()
        with torch.no_grad():
            output = model(self.batch["logmels"], self.batch["input_lengths"], return_attention=True)
        self.assertEqual(output.logits.shape[0], 3)
        self.assertEqual(output.logits.shape[2], len(self.tokenizer))
        self.assertTrue((output.output_lengths >= self.batch["target_lengths"]).all())
        self.assertEqual(output.padding_mask.shape, output.logits.shape[:2])
        self.assertEqual(output.attention.shape[:2], (3, CONFIG["model"]["num_heads"]))

    def test_unit_asr_002_padding_does_not_change_valid_logits(self):
        model = self.make_model().eval()
        one = collate_asr([self.dataset[0]])
        padded = one["logmels"].new_zeros(1, one["logmels"].shape[1] + 120, one["logmels"].shape[2])
        padded[:, : one["logmels"].shape[1]] = one["logmels"]
        with torch.no_grad():
            first = model(one["logmels"], one["input_lengths"]).logits
            second = model(padded, one["input_lengths"]).logits
        valid = int(model(one["logmels"], one["input_lengths"]).output_lengths[0])
        self.assertTrue(torch.allclose(first[:, :valid], second[:, :valid], atol=1e-5))

    def test_unit_asr_003_ctc_loss_reaches_all_core_modules(self):
        model = self.make_model()
        output = model(self.batch["logmels"], self.batch["input_lengths"])
        loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(output.logits.log_softmax(-1).transpose(0, 1), self.batch["targets"], output.output_lengths, self.batch["target_lengths"])
        loss.backward()
        for name, module in {"subsampling": model.subsampling, "attention": model.blocks[0].attention, "convolution": model.blocks[0].convolution, "classifier": model.classifier}.items():
            with self.subTest(module=name):
                gradients = [value.grad for value in module.parameters() if value.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_unit_asr_004_decode_and_spans_are_observable(self):
        logits = torch.full((1, 6, len(self.tokenizer)), -10.0)
        ids = [0, self.tokenizer.encode("红")[0], self.tokenizer.encode("红")[0], 0, self.tokenizer.encode("色")[0], 0]
        for step, token_id in enumerate(ids):
            logits[0, step, token_id] = 10.0
        self.assertEqual(ctc_greedy_decode(logits, torch.tensor([6]), self.tokenizer), ["红色"])
        self.assertEqual([item["token"] for item in forced_greedy_spans(logits[0], 6, self.tokenizer)], ["红", "色"])


if __name__ == "__main__":
    unittest.main()
