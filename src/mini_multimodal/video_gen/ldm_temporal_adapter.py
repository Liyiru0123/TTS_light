"""Latent video adapter for a frozen per-frame image LDM source."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .model import TemporalTransformer


class SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class LDMVideoAdapter(nn.Module):
    """Refine independently sampled image-LDM latents into a video sequence.

    ``use_temporal`` is fixed at construction time.  OFF and ON are therefore
    separately trainable models; OFF is not produced by bypassing an ON model.
    """

    def __init__(self, *, latent_channels: int = 4, base_channels: int = 32,
                 heads: int = 4, use_temporal: bool, motion_align: bool = False,
                 residual_scale: float = 1.0) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8")
        self.latent_channels = int(latent_channels)
        self.use_temporal = bool(use_temporal)
        self.motion_align = bool(motion_align)
        self.residual_scale = float(residual_scale)
        hidden = base_channels * 2
        self.input = nn.Conv2d(latent_channels, base_channels, 3, padding=1)
        self.spatial16 = SpatialResidualBlock(base_channels)
        self.down = nn.Conv2d(base_channels, hidden, 4, stride=2, padding=1)
        self.spatial8 = SpatialResidualBlock(hidden)
        if self.use_temporal:
            self.temporal16 = TemporalTransformer(base_channels, heads=heads, max_frames=8)
            self.temporal8 = TemporalTransformer(hidden, heads=heads, max_frames=8)
        self.up = nn.Conv2d(hidden, base_channels, 3, padding=1)
        self.merge = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.output = nn.Conv2d(base_channels, latent_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _framewise(module: nn.Module, video: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        flat = module(flat)
        out_channels, out_height, out_width = flat.shape[1:]
        return flat.reshape(batch, frames, out_channels, out_height, out_width).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _roll_by_trajectory(video: torch.Tensor, trajectory: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        """Align moving content to frame 0 with integer latent-grid shifts."""
        batch, _, frames, height, width = video.shape
        if trajectory.shape != (batch, frames, 2):
            raise ValueError("trajectory must have shape [B,8,2]")
        output = torch.empty_like(video)
        reference = trajectory[:, :1]
        offsets = reference - trajectory
        if inverse:
            offsets = -offsets
        for batch_index in range(batch):
            for frame_index in range(frames):
                dx = int(torch.round(offsets[batch_index, frame_index, 0] * (width - 1)).item())
                dy = int(torch.round(offsets[batch_index, frame_index, 1] * (height - 1)).item())
                output[batch_index, :, frame_index] = torch.roll(
                    video[batch_index, :, frame_index], shifts=(dy, dx), dims=(-2, -1)
                )
        return output

    def _temporal(self, module: nn.Module, value: torch.Tensor,
                  trajectory: torch.Tensor | None) -> torch.Tensor:
        if self.motion_align:
            if trajectory is None:
                raise ValueError("motion-aligned temporal adapter requires trajectory")
            aligned = self._roll_by_trajectory(value, trajectory, inverse=False)
            refined = module(aligned)
            return self._roll_by_trajectory(refined, trajectory, inverse=True)
        return module(value)

    def forward(self, video: torch.Tensor, trajectory: torch.Tensor | None = None) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1:] != (self.latent_channels, 8, 16, 16):
            raise ValueError("video must have shape [B,C,8,16,16]")
        value16 = self._framewise(self.input, video)
        value16 = self._framewise(self.spatial16, value16)
        if self.use_temporal:
            value16 = self._temporal(self.temporal16, value16, trajectory)
        value8 = self._framewise(self.down, value16)
        value8 = self._framewise(self.spatial8, value8)
        if self.use_temporal:
            value8 = self._temporal(self.temporal8, value8, trajectory)
        up = self._framewise(lambda value: F.interpolate(value, scale_factor=2, mode="nearest"), value8)
        up = self._framewise(self.up, up)
        merged = torch.cat((up, value16), dim=1)
        correction = self._framewise(self.output, self._framewise(self.merge, merged))
        return video + self.residual_scale * torch.tanh(correction)
