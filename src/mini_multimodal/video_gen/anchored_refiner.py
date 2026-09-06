"""Small appearance-anchored video refiner with an AnimateDiff motion module."""

from __future__ import annotations

import torch
from torch import nn

from .model import TemporalTransformer


class AnchoredTemporalRefiner(nn.Module):
    """Refine independently perturbed anchor frames through temporal attention."""

    def __init__(self, channels: int = 48, heads: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(24, channels, 4, stride=2, padding=1), nn.SiLU(),
        )
        self.motion_module = TemporalTransformer(channels, heads=heads, max_frames=8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 24, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=2, padding=1),
        )

    def forward(self, video: torch.Tensor, *, use_temporal: bool = True) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1:] != (3, 8, 64, 64):
            raise ValueError("video must have shape [B,3,8,64,64]")
        batch = video.shape[0]
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * 8, 3, 64, 64)
        hidden = self.encoder(flat)
        hidden = hidden.reshape(batch, 8, hidden.shape[1], 16, 16).permute(0, 2, 1, 3, 4)
        if use_temporal:
            hidden = self.motion_module(hidden)
        flat_hidden = hidden.permute(0, 2, 1, 3, 4).reshape(batch * 8, hidden.shape[1], 16, 16)
        correction = self.decoder(flat_hidden).reshape(batch, 8, 3, 64, 64).permute(0, 2, 1, 3, 4)
        return (video + correction).clamp(0.0, 1.0)


def perturb_independent_frames(video: torch.Tensor, *, generator: torch.Generator,
                               noise_std: float = 0.035, brightness: float = 0.025) -> torch.Tensor:
    """Simulate independent-frame generation variation with reproducible noise."""
    batch = video.shape[0]
    gain = 1.0 + (torch.rand((batch, 1, 8, 1, 1), generator=generator) * 2 - 1) * brightness
    noise = torch.randn(video.shape, generator=generator, dtype=video.dtype) * noise_std
    return (video * gain + noise).clamp(0.0, 1.0)
