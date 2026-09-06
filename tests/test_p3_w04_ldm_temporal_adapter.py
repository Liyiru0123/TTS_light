from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.video_gen.ldm_temporal_adapter import LDMVideoAdapter  # noqa: E402
from mini_multimodal.video_gen.model import TemporalTransformer  # noqa: E402


class LDMVideoAdapterTests(unittest.TestCase):
    def test_unit_vidgen_f_01_on_preserves_video_shape_and_initial_identity(self) -> None:
        model = LDMVideoAdapter(use_temporal=True)
        value = torch.randn(2, 4, 8, 16, 16)
        output = model(value)
        self.assertEqual(tuple(output.shape), tuple(value.shape))
        torch.testing.assert_close(output, value)

    def test_unit_vidgen_f_01_off_is_separately_trainable_without_temporal_parameters(self) -> None:
        off = LDMVideoAdapter(use_temporal=False)
        on = LDMVideoAdapter(use_temporal=True)
        self.assertFalse(any(name.startswith("temporal") for name, _ in off.named_parameters()))
        self.assertTrue(any(name.startswith("temporal") for name, _ in on.named_parameters()))
        value = torch.randn(1, 4, 8, 16, 16, requires_grad=True)
        off(value).square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in off.parameters()))

    def test_invalid_video_shape_fails_explicitly(self) -> None:
        model = LDMVideoAdapter(use_temporal=True)
        with self.assertRaisesRegex(ValueError, r"\[B,C,8,16,16\]"):
            model(torch.randn(1, 4, 7, 16, 16))

    def test_motion_aligned_path_requires_and_preserves_trajectory_shape(self) -> None:
        model = LDMVideoAdapter(use_temporal=True, motion_align=True, residual_scale=0.1)
        value = torch.randn(1, 4, 8, 16, 16)
        with self.assertRaisesRegex(ValueError, "requires trajectory"):
            model(value)
        trajectory = torch.full((1, 8, 2), 0.5)
        output = model(value, trajectory)
        self.assertEqual(tuple(output.shape), tuple(value.shape))
        torch.testing.assert_close(output, value)

    def test_zero_initialized_motion_module_is_identity_but_trainable(self) -> None:
        module = TemporalTransformer(32, heads=4, max_frames=8, zero_init_gate=True)
        value = torch.randn(1, 32, 8, 4, 4, requires_grad=True)
        output = module(value)
        torch.testing.assert_close(output, value)
        output.square().mean().backward()
        self.assertIsNotNone(module.output_gate.grad)


if __name__ == "__main__":
    unittest.main()
