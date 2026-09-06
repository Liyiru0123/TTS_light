from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.vlm import (  # noqa: E402
    encode_raw_image_with_vae,
    first_cached_latent,
    load_tiny_vlm_inference_bundle,
    predict_from_latent,
)


class TinyVLMInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_tiny_vlm_inference_bundle(
            ROOT / "results/P2-W03_tiny_vlm/checkpoints/tiny_vlm_inference.pt"
        )
        cls.cache_path = ROOT / "data/latents/p2_w02_v1/posterior_means.npz"

    def test_infer_vlm_001_cached_latent_predicts_fixed_answer(self) -> None:
        latent = first_cached_latent(self.cache_path, scene_id="core_scene_000700")
        result = predict_from_latent(self.loaded, latent, "图中有几个可见对象？")
        self.assertEqual(result["classification_answer"], "1个")
        self.assertEqual(result["generated_answer"], "1个")

    def test_infer_vlm_002_raw_image_matches_cached_posterior_mean(self) -> None:
        cached = first_cached_latent(self.cache_path, scene_id="core_scene_000700")
        encoded = encode_raw_image_with_vae(
            ROOT / "data/images/p1_w04_v1/images/core/test/core_scene_000700_f000.png",
            vae_checkpoint=ROOT / "results/P2-W02_image_vae/checkpoints/best.pt",
            vae_config=ROOT / "configs/p2_w02_image_vae.json",
        )
        self.assertTrue(torch.allclose(encoded, cached, atol=1e-3, rtol=0.0))
        result = predict_from_latent(self.loaded, encoded, "图中有几个可见对象？")
        self.assertEqual(result["generated_answer"], "1个")

    def test_infer_vlm_003_invalid_shapes_and_scene_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            predict_from_latent(self.loaded, torch.zeros(4, 8, 8), "图中有几个可见对象？")
        with self.assertRaises(KeyError):
            first_cached_latent(self.cache_path, scene_id="missing_scene")


if __name__ == "__main__":
    unittest.main()
