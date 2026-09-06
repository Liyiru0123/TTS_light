"""A two-scale U-Net denoiser with time injection and text cross-attention."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time dimension must be even and at least 4")
        self.dimension = dimension

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = timestep.float()[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class TimeResidualBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = hidden + self.time(F.silu(time))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return value + hidden


@dataclass(slots=True)
class DenoiserOutput:
    predicted_noise: torch.Tensor
    cross_attention: torch.Tensor


class TinyImageDenoiser(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        latent_channels: int = 4,
        base_channels: int = 32,
        text_dim: int = 64,
        time_dim: int = 64,
        attention_heads: int = 4,
        max_text_length: int = 128,
    ) -> None:
        super().__init__()
        if base_channels % 8 or text_dim != base_channels * 2:
            raise ValueError("base channels must divide groups and text_dim must equal 2*base_channels")
        hidden = base_channels * 2
        self.latent_channels = latent_channels
        self.max_text_length = max_text_length
        self.text_embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.text_position = nn.Parameter(torch.randn(max_text_length, text_dim) * 0.01)
        text_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=attention_heads,
            dim_feedforward=text_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=1)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.text_time = nn.Sequential(nn.Linear(text_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(latent_channels, base_channels, 3, padding=1)
        self.block16 = TimeResidualBlock(base_channels, time_dim)
        self.down = nn.Conv2d(base_channels, hidden, 4, stride=2, padding=1)
        self.block8a = TimeResidualBlock(hidden, time_dim)
        self.cross_attention = nn.MultiheadAttention(hidden, attention_heads, dropout=0.0, batch_first=True)
        self.block8b = TimeResidualBlock(hidden, time_dim)
        self.up_projection = nn.Conv2d(hidden, base_channels, 3, padding=1)
        self.merge = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.block_out = TimeResidualBlock(base_channels, time_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, 3, padding=1),
        )

    def encode_text(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        if text_ids.ndim != 2 or text_ids.shape != text_mask.shape:
            raise ValueError("text ids/mask shape mismatch")
        if text_ids.shape[1] > self.max_text_length or not text_mask.any(dim=1).all():
            raise ValueError("text sequence is empty or too long")
        value = self.text_embedding(text_ids) + self.text_position[: text_ids.shape[1]][None, :, :]
        return self.text_encoder(value, src_key_padding_mask=~text_mask)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        disable_cross_attention: bool = False,
    ) -> DenoiserOutput:
        if noisy_latent.ndim != 4 or noisy_latent.shape[1:] != (self.latent_channels, 16, 16):
            raise ValueError("denoiser expects [B,C,16,16] latent")
        if timestep.shape != (noisy_latent.shape[0],):
            raise ValueError("timestep shape mismatch")
        text = self.encode_text(text_ids, text_mask)
        weights_for_pool = text_mask.to(text.dtype).unsqueeze(-1)
        pooled_text = (text * weights_for_pool).sum(dim=1) / weights_for_pool.sum(dim=1).clamp_min(1.0)
        time = self.time_embedding(timestep) + self.text_time(pooled_text)
        skip = self.block16(self.input(noisy_latent), time)
        hidden = self.block8a(self.down(skip), time)
        queries = hidden.flatten(2).transpose(1, 2)
        if disable_cross_attention:
            attended = torch.zeros_like(queries)
            weights = torch.zeros(
                queries.shape[0],
                self.cross_attention.num_heads,
                queries.shape[1],
                text.shape[1],
                device=queries.device,
            )
        else:
            attended, weights = self.cross_attention(
                queries,
                text,
                text,
                key_padding_mask=~text_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        hidden = hidden + attended.transpose(1, 2).reshape_as(hidden)
        hidden = self.block8b(hidden, time)
        up = F.interpolate(hidden, scale_factor=2, mode="nearest")
        up = self.up_projection(up)
        merged = self.merge(torch.cat((up, skip), dim=1))
        result = self.output(self.block_out(merged, time))
        return DenoiserOutput(result, weights)
