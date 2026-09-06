from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.data import CharacterTokenizer  # noqa: E402
from mini_multimodal.ldm.inference import sample_latents  # noqa: E402
from mini_multimodal.ldm.model import TinyImageDenoiser  # noqa: E402
from mini_multimodal.ldm.schedule import DiffusionSchedule  # noqa: E402


class P2W04LDMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        vocabulary = json.loads(
            (ROOT / "data" / "ldm" / "p2_w04_v1" / "vocabulary.json").read_text(encoding="utf-8")
        )
        cls.tokenizer = CharacterTokenizer(vocabulary)

    def new_model(self) -> TinyImageDenoiser:
        return TinyImageDenoiser(vocab_size=len(self.tokenizer), base_channels=32, text_dim=64)

    def test_unit_ldm_001_forward_noise_matches_closed_form(self) -> None:
        schedule = DiffusionSchedule(timesteps=10, beta_start=1e-4, beta_end=0.02)
        clean = torch.ones(2, 4, 16, 16)
        noise = torch.full_like(clean, 0.5)
        timestep = torch.tensor([0, 9])
        actual = schedule.q_sample(clean, timestep, noise)
        expected = (
            schedule.sqrt_alpha_bars[timestep][:, None, None, None] * clean
            + schedule.sqrt_one_minus_alpha_bars[timestep][:, None, None, None] * noise
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        self.assertTrue(torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1]))
        previous = schedule.ddim_step(
            actual,
            timestep,
            torch.tensor([-1, 8]),
            noise,
        )
        self.assertEqual(tuple(previous.shape), tuple(clean.shape))
        torch.testing.assert_close(previous[0], clean[0], atol=1e-5, rtol=1e-5)

    def test_unit_ldm_002_time_text_attention_and_gradient_are_active(self) -> None:
        torch.manual_seed(3)
        model = self.new_model()
        latent = torch.randn(2, 4, 16, 16)
        ids = torch.tensor([
            self.tokenizer.encode("红色圆形", 32),
            self.tokenizer.encode("蓝色圆形", 32),
        ])
        mask = torch.ones_like(ids, dtype=torch.bool)
        first = model(latent, torch.tensor([1, 1]), ids, mask)
        later = model(latent, torch.tensor([8, 8]), ids, mask)
        swapped = model(latent, torch.tensor([1, 1]), ids.flip(0), mask)
        self.assertEqual(tuple(first.predicted_noise.shape), (2, 4, 16, 16))
        self.assertEqual(tuple(first.cross_attention.shape[:3]), (2, 4, 64))
        torch.testing.assert_close(first.cross_attention.sum(dim=-1), torch.ones_like(first.cross_attention.sum(dim=-1)), atol=1e-5, rtol=0)
        self.assertFalse(torch.allclose(first.predicted_noise, later.predicted_noise))
        self.assertFalse(torch.allclose(first.predicted_noise, swapped.predicted_noise))
        first.predicted_noise.square().mean().backward()
        self.assertGreater(float(model.cross_attention.in_proj_weight.grad.abs().sum()), 0.0)

    def test_unit_ldm_003_sampling_is_seeded_and_returns_trajectory(self) -> None:
        torch.manual_seed(4)
        model = self.new_model()
        schedule = DiffusionSchedule(timesteps=6)
        kwargs = dict(
            model=model,
            schedule=schedule,
            tokenizer=self.tokenizer,
            prompts=["红色圆形"],
            sample_steps=3,
            guidance_scale=1.5,
            seed=9,
        )
        first = sample_latents(**kwargs)
        second = sample_latents(**kwargs)
        torch.testing.assert_close(first["latent"], second["latent"])
        self.assertEqual(len(first["latent_trajectory"]), 4)
        self.assertEqual(len(first["attention_trajectory"]), 3)


if __name__ == "__main__":
    unittest.main()
