from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.asr import CharacterCTCTokenizer, TinyASR  # noqa: E402

RESULT = ROOT / "results/P2-W05_tiny_asr"


class TinyASRArtifactTests(unittest.TestCase):
    def test_art_asr_001_contract_and_inventory(self):
        contract = json.loads((RESULT / "artifact_contract.json").read_text(encoding="utf-8"))
        inventory_rows = json.loads((RESULT / "artifact_inventory.json").read_text(encoding="utf-8"))["files"]
        inventory = {item["path"]: item for item in inventory_rows}
        required = [item["path"] for item in contract["required_artifacts"]]
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file())
                prefix = "results/P2-W05_tiny_asr/"
                if relative.startswith(prefix):
                    self.assertIn(relative, inventory)

    def test_art_asr_002_checkpoint_hashes_and_inference_strict_load(self):
        checkpoint_root = RESULT / "iterations/I001/branches/baseline/checkpoints"
        manifest = json.loads((checkpoint_root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        for name, metadata in manifest.items():
            self.assertEqual(hashlib.sha256((checkpoint_root / name).read_bytes()).hexdigest(), metadata["sha256"])
        package = torch.load(RESULT / "iterations/I001/branches/baseline/inference_package.pt", map_location="cpu", weights_only=False)
        tokenizer = CharacterCTCTokenizer(package["vocabulary"])
        model = TinyASR(vocabulary_size=len(tokenizer), **package["model_config"])
        model.load_state_dict(package["model_state"], strict=True)

    def test_ped_asr_001_examples_include_machine_correct_and_error_cases(self):
        examples = json.loads((RESULT / "capability_examples.json").read_text(encoding="utf-8"))["examples"]
        self.assertGreaterEqual(len(examples), 10)
        self.assertTrue(any(item["is_fully_correct"] for item in examples))
        self.assertTrue(any(not item["is_fully_correct"] for item in examples))
        for item in examples:
            self.assertTrue((ROOT / item["display_artifacts"][0]).is_file())
            self.assertIn("greedy_spans", item["model_outputs"])

    def test_art_asr_003_candidate_and_release_are_not_confused(self):
        candidate = json.loads((RESULT / "current_candidate.json").read_text(encoding="utf-8"))["current_candidate"]
        release = json.loads((RESULT / "official_release.json").read_text(encoding="utf-8"))["official_release"]
        acceptance = json.loads((RESULT / "iterations/I001/branches/baseline/results/acceptance_report.json").read_text(encoding="utf-8"))
        self.assertEqual((candidate["iteration_id"], candidate["branch_id"]), ("I002", "robust_finetune"))
        self.assertIsNone(release)
        self.assertFalse(acceptance["all_blocking_acceptance_items_passed"])
        self.assertEqual(set(acceptance["failed_items"]), {"MODEL-ASR-001", "ROB-ASR-001"})


if __name__ == "__main__":
    unittest.main()
