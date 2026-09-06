"""A compact spatial AutoencoderKL retaining the essential VAE operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


@dataclass(slots=True)
class VAEOutput:
    reconstruction: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    latent: torch.Tensor


class TinyImageVAE(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 32,
        latent_channels: int = 4,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ) -> None:
        super().__init__()
        if base_channels % 8 != 0 or latent_channels <= 0:
            raise ValueError("base channels must be divisible by 8 and latent channels positive")
        self.latent_channels = latent_channels
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        hidden = base_channels * 2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, hidden, 4, stride=2, padding=1),
            ResidualBlock(hidden),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1),
            ResidualBlock(hidden),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, latent_channels * 2, 3, padding=1),
        )
        self.decoder_input = nn.Conv2d(latent_channels, hidden, 3, padding=1)
        self.decoder = nn.Sequential(
            ResidualBlock(hidden),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            ResidualBlock(hidden),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden, base_channels, 3, padding=1),
            ResidualBlock(base_channels),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1:] != (3, 64, 64):
            raise ValueError("TinyImageVAE expects [B,3,64,64]")
        moments = self.encoder(image)
        mean, logvar = moments.chunk(2, dim=1)
        return mean, logvar.clamp(self.logvar_min, self.logvar_max)

    @staticmethod
    def reparameterize(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return mean + torch.exp(0.5 * logvar) * noise

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != self.latent_channels:
            raise ValueError("latent must be [B,C,H,W] with configured channels")
        return self.decoder(self.decoder_input(latent))

    def forward(
        self,
        image: torch.Tensor,
        *,
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> VAEOutput:
        mean, logvar = self.encode(image)
        latent = (
            self.reparameterize(mean, logvar, generator=generator)
            if sample_posterior
            else mean
        )
        return VAEOutput(self.decode(latent), mean, logvar, latent)

