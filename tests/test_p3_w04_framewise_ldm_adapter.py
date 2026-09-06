from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.video_gen.framewise_ldm_adapter import FramewiseLDMNoiseAdapter  # noqa: E402


class FramewiseLDMNoiseAdapterTests(unittest.TestCase):
    def test_off_and_on_are_identity_at_initialization(self) -> None:
        noise = torch.randn(2, 4, 8, 16, 16)
        trajectory = torch.rand(2, 8, 2)
        for use_temporal in (False, True):
            model = FramewiseLDMNoiseAdapter(use_temporal=use_temporal)
            torch.testing.assert_close(model(noise, trajectory), noise)

    def test_on_contains_two_animate_diff_dimension_transform_modules(self) -> None:
        off = FramewiseLDMNoiseAdapter(use_temporal=False)
        on = FramewiseLDMNoiseAdapter(use_temporal=True)
        self.assertFalse(any(name.startswith("temporal") for name, _ in off.named_parameters()))
        self.assertTrue(hasattr(on, "temporal16"))
        self.assertTrue(hasattr(on, "temporal8"))

    def test_trajectory_contract_is_explicit(self) -> None:
        model = FramewiseLDMNoiseAdapter(use_temporal=True)
        with self.assertRaisesRegex(ValueError, "trajectory"):
            model(torch.randn(1, 4, 8, 16, 16), torch.rand(1, 7, 2))


if __name__ == "__main__":
    unittest.main()
