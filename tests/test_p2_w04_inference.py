from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.inference import (  # noqa: E402
    generate_images_from_package,
    load_inference_package,
    load_vae_decoder,
)
from mini_multimodal.ldm.model import TinyImageDenoiser  # noqa: E402
from mini_multimodal.training import load_checkpoint  # noqa: E402


class P2W04InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = load_inference_package(
            ROOT / "results" / "P2-W04_tiny_image_ldm" / "checkpoints" / "tiny_image_ldm_inference.pt"
        )
        cls.vae = load_vae_decoder(ROOT, cls.package["payload"]["vae_checkpoint"])

    def test_int_ldm_001_inference_package_generates_seeded_image_and_traces(self) -> None:
        prompt = "第1帧中有1个可见对象。对象1是红色的大圆形，位于中央区域。"
        first = generate_images_from_package(self.package, self.vae, [prompt], seed=23, sample_steps=3)
        second = generate_images_from_package(self.package, self.vae, [prompt], seed=23, sample_steps=3)
        torch.testing.assert_close(first["images"], second["images"])
        self.assertEqual(tuple(first["images"].shape), (1, 3, 64, 64))
        self.assertEqual(len(first["latent_trajectory"]), 4)
        self.assertEqual(len(first["attention_trajectory"]), 3)

    def test_int_ldm_002_last_checkpoint_restores_optimizer_and_epoch(self) -> None:
        config = json.loads((ROOT / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
        model = TinyImageDenoiser(
            vocab_size=len(self.package["tokenizer"]),
            **config["model"],
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
        state = load_checkpoint(
            ROOT / config["output_results"] / "training" / "checkpoints" / "last.pt",
            model=model,
            optimizer=optimizer,
            restore_rng=False,
        )
        self.assertEqual(state["epoch"], 29)
        self.assertTrue(optimizer.state)


if __name__ == "__main__":
    unittest.main()
