"""Trainable video-noise adapters for a frozen framewise image LDM."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .model import TemporalTransformer


class FramewiseLDMNoiseAdapter(nn.Module):
    """Correct frozen image-LDM noise predictions, optionally across time."""

    def __init__(self, *, channels: int = 16, use_temporal: bool) -> None:
        super().__init__()
        self.use_temporal = bool(use_temporal)
        self.input = nn.Conv2d(5, channels, 3, padding=1)
        self.spatial = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
        )
        if self.use_temporal:
            self.temporal16 = TemporalTransformer(channels, heads=4, max_frames=8, zero_init_gate=True)
            self.down = nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1)
            self.temporal8 = TemporalTransformer(channels * 2, heads=4, max_frames=8, zero_init_gate=True)
            self.up = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.output = nn.Conv2d(channels, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def trajectory_heatmap(trajectory: torch.Tensor, grid: int = 16, sigma: float = 1.5) -> torch.Tensor:
        batch, frames, _ = trajectory.shape
        coords = trajectory * (grid - 1)
        gy, gx = torch.meshgrid(
            torch.arange(grid, dtype=trajectory.dtype, device=trajectory.device),
            torch.arange(grid, dtype=trajectory.dtype, device=trajectory.device), indexing="ij",
        )
        px = coords[:, :, 0, None, None]
        py = coords[:, :, 1, None, None]
        value = torch.exp(-((gx[None, None] - px).square() + (gy[None, None] - py).square()) / (2 * sigma**2))
        return value.unsqueeze(1)

    @staticmethod
    def _framewise(module: nn.Module, video: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        flat = module(flat)
        return flat.reshape(batch, frames, flat.shape[1], flat.shape[2], flat.shape[3]).permute(0, 2, 1, 3, 4)

    def forward(self, predicted_noise: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        if predicted_noise.ndim != 5 or predicted_noise.shape[1:] != (4, 8, 16, 16):
            raise ValueError("predicted_noise must have shape [B,4,8,16,16]")
        if trajectory.shape != (predicted_noise.shape[0], 8, 2):
            raise ValueError("trajectory must have shape [B,8,2]")
        heatmap = self.trajectory_heatmap(trajectory)
        value = self._framewise(self.input, torch.cat((predicted_noise, heatmap), dim=1))
        value = self._framewise(self.spatial, value)
        if self.use_temporal:
            temporal16 = self.temporal16(value)
            low = self._framewise(self.down, value)
            temporal8 = self.temporal8(low)
            low_delta = temporal8 - low
            up_delta = self._framewise(self.up, self._framewise(
                nn.Upsample(scale_factor=2, mode="nearest"), low_delta
            ))
            value = temporal16 + up_delta
        correction = self._framewise(self.output, value)
        return predicted_noise + correction

