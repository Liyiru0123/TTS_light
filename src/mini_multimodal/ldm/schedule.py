"""Discrete Gaussian forward process and deterministic DDIM-style updates."""

from __future__ import annotations

import torch
from torch import nn


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        if timesteps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("invalid diffusion schedule")
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = int(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    @staticmethod
    def _extract(values: torch.Tensor, timestep: torch.Tensor, ndim: int) -> torch.Tensor:
        result = values.gather(0, timestep.long())
        return result.reshape(result.shape[0], *((1,) * (ndim - 1)))

    def q_sample(self, clean: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if clean.shape != noise.shape or timestep.shape != (clean.shape[0],):
            raise ValueError("q_sample shape mismatch")
        signal = self._extract(self.sqrt_alpha_bars, timestep, clean.ndim)
        sigma = self._extract(self.sqrt_one_minus_alpha_bars, timestep, clean.ndim)
        return signal * clean + sigma * noise

    def predict_clean(self, noisy: torch.Tensor, timestep: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        signal = self._extract(self.sqrt_alpha_bars, timestep, noisy.ndim)
        sigma = self._extract(self.sqrt_one_minus_alpha_bars, timestep, noisy.ndim)
        return (noisy - sigma * predicted_noise) / signal.clamp_min(1e-8)

    def predict_noise(self, noisy: torch.Tensor, timestep: torch.Tensor, predicted_clean: torch.Tensor) -> torch.Tensor:
        """Convert a clean-sample estimate back to the schedule's noise parameterization."""
        if noisy.shape != predicted_clean.shape or timestep.shape != (noisy.shape[0],):
            raise ValueError("predict_noise shape mismatch")
        signal = self._extract(self.sqrt_alpha_bars, timestep, noisy.ndim)
        sigma = self._extract(self.sqrt_one_minus_alpha_bars, timestep, noisy.ndim)
        return (noisy - signal * predicted_clean) / sigma.clamp_min(1e-8)

    def ddim_step(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        previous_timestep: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        clean = self.predict_clean(noisy, timestep, predicted_noise)
        previous_alpha = torch.where(
            (previous_timestep >= 0).reshape(previous_timestep.shape[0], *((1,) * (noisy.ndim - 1))),
            self._extract(self.alpha_bars, previous_timestep.clamp_min(0), noisy.ndim),
            torch.ones((noisy.shape[0], *((1,) * (noisy.ndim - 1))), device=noisy.device),
        )
        return previous_alpha.sqrt() * clean + (1.0 - previous_alpha).clamp_min(0).sqrt() * predicted_noise

    def sampling_timesteps(self, steps: int) -> list[int]:
        if not 2 <= steps <= self.timesteps:
            raise ValueError("sample steps must be within [2, train_timesteps]")
        values = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        return list(dict.fromkeys(int(value) for value in values))
