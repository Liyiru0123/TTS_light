"""Coordinate-aware, multi-scale text-conditioned denoiser for P2-W04 v2.

The model remains a pure text-to-image latent diffusion denoiser.  Region words
receive a small, explicit 2-D token feature, while visual queries receive the
matching normalized coordinate system.  No target box or mask is consumed at
inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .model import SinusoidalTimeEmbedding, TimeResidualBlock


def coordinate_grid(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return an [1,2,H,W] grid in x/y order with values in [-1,1]."""
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0)


class CoordinateAwareCrossAttention(nn.Module):
    """Text cross-attention whose visual queries explicitly carry 2-D position."""

    def __init__(self, channels: int, text_dim: int, heads: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.visual_norm = nn.GroupNorm(min(8, channels), channels)
        self.text_norm = nn.LayerNorm(text_dim)
        self.coordinate_projection = nn.Conv2d(2, channels, 1, bias=False)
        self.attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=0.0,
            batch_first=True,
            kdim=text_dim,
            vdim=text_dim,
        )
        self.output_gate = nn.Parameter(torch.tensor(0.5))
        self.spatial_bias_strength = nn.Parameter(torch.tensor(4.0))

    def forward(
        self,
        value: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        token_coordinates: torch.Tensor,
        token_coordinate_mask: torch.Tensor,
        *,
        disable_attention: bool = False,
        disable_coordinates: bool = False,
        disable_spatial_attention_bias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = value.shape
        query_map = self.visual_norm(value)
        if not disable_coordinates:
            grid = coordinate_grid(height, width, device=value.device, dtype=value.dtype)
            query_map = query_map + self.coordinate_projection(grid)
        queries = query_map.flatten(2).transpose(1, 2)
        if disable_attention:
            attended = torch.zeros_like(queries)
            weights = torch.zeros(
                batch,
                self.attention.num_heads,
                height * width,
                text.shape[1],
                device=value.device,
                dtype=value.dtype,
            )
        else:
            attention_mask = None
            if not disable_coordinates and not disable_spatial_attention_bias:
                grid_xy = coordinate_grid(height, width, device=value.device, dtype=value.dtype)
                query_xy = grid_xy.flatten(2).transpose(1, 2).expand(batch, -1, -1)
                distance = (query_xy[:, :, None, :] - token_coordinates[:, None, :, :]).square().sum(dim=-1)
                locality = -F.softplus(self.spatial_bias_strength) * distance
                locality = locality * token_coordinate_mask.squeeze(-1)[:, None, :]
                attention_mask = locality[:, None, :, :].expand(
                    batch, self.attention.num_heads, height * width, text.shape[1]
                ).reshape(batch * self.attention.num_heads, height * width, text.shape[1])
            padding_mask: torch.Tensor = ~text_mask
            if attention_mask is not None:
                padding_mask = torch.zeros_like(text_mask, dtype=value.dtype).masked_fill(~text_mask, float("-inf"))
            attended, weights = self.attention(
                queries,
                self.text_norm(text),
                self.text_norm(text),
                key_padding_mask=padding_mask,
                attn_mask=attention_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        update = attended.transpose(1, 2).reshape(batch, self.channels, height, width)
        return value + torch.tanh(self.output_gate) * update, weights


@dataclass(slots=True)
class DenoiserOutputV2:
    predicted_noise: torch.Tensor
    cross_attention_16: torch.Tensor
    cross_attention_8: torch.Tensor

    @property
    def cross_attention(self) -> torch.Tensor:
        """Compatibility view used by the original sampler."""
        return self.cross_attention_8


class TinyImageDenoiserV2(nn.Module):
    """Two-scale U-Net with coordinate-aware text conditioning at both scales."""

    model_family = "TinyImageLDM-v2"

    def __init__(
        self,
        *,
        vocab_size: int,
        latent_channels: int = 4,
        base_channels: int = 48,
        text_dim: int = 96,
        time_dim: int = 96,
        attention_heads: int = 4,
        max_text_length: int = 128,
        spatial_token_coordinates: Mapping[int, Sequence[float]] | None = None,
    ) -> None:
        super().__init__()
        if base_channels % 8 or text_dim != base_channels * 2:
            raise ValueError("base channels must divide groups and text_dim must equal 2*base_channels")
        hidden = base_channels * 2
        self.latent_channels = int(latent_channels)
        self.max_text_length = int(max_text_length)
        self.text_embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.text_position = nn.Parameter(torch.randn(max_text_length, text_dim) * 0.01)
        coordinates = torch.zeros(vocab_size, 2)
        coordinate_mask = torch.zeros(vocab_size, 1)
        for token_id, xy in (spatial_token_coordinates or {}).items():
            if len(xy) != 2 or not 0 <= int(token_id) < vocab_size:
                raise ValueError("invalid spatial token coordinate")
            coordinates[int(token_id)] = torch.tensor([float(xy[0]), float(xy[1])])
            coordinate_mask[int(token_id)] = 1.0
        self.register_buffer("spatial_token_coordinates", coordinates, persistent=True)
        self.register_buffer("spatial_token_mask", coordinate_mask, persistent=True)
        self.text_coordinate_projection = nn.Linear(2, text_dim, bias=False)
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
        self.input_coordinate = nn.Conv2d(2, base_channels, 1, bias=False)
        self.text_spatial_16 = nn.Conv2d(1, base_channels, 3, padding=1, bias=False)
        self.block16a = TimeResidualBlock(base_channels, time_dim)
        self.cross_attention_16 = CoordinateAwareCrossAttention(base_channels, text_dim, attention_heads)
        self.block16b = TimeResidualBlock(base_channels, time_dim)
        self.down = nn.Conv2d(base_channels, hidden, 4, stride=2, padding=1)
        self.down_coordinate = nn.Conv2d(2, hidden, 1, bias=False)
        self.text_spatial_8 = nn.Conv2d(1, hidden, 3, padding=1, bias=False)
        self.block8a = TimeResidualBlock(hidden, time_dim)
        self.cross_attention_8 = CoordinateAwareCrossAttention(hidden, text_dim, attention_heads)
        self.block8b = TimeResidualBlock(hidden, time_dim)
        self.up_projection = nn.Conv2d(hidden, base_channels, 3, padding=1)
        self.merge = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.block_out = TimeResidualBlock(base_channels, time_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, 3, padding=1),
        )

    def encode_text(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        disable_coordinates: bool = False,
    ) -> torch.Tensor:
        if text_ids.ndim != 2 or text_ids.shape != text_mask.shape:
            raise ValueError("text ids/mask shape mismatch")
        if text_ids.shape[1] > self.max_text_length or not text_mask.any(dim=1).all():
            raise ValueError("text sequence is empty or too long")
        value = self.text_embedding(text_ids) + self.text_position[: text_ids.shape[1]][None, :, :]
        if not disable_coordinates:
            token_xy = self.spatial_token_coordinates[text_ids]
            token_has_xy = self.spatial_token_mask[text_ids]
            value = value + self.text_coordinate_projection(token_xy) * token_has_xy
        return self.text_encoder(value, src_key_padding_mask=~text_mask)

    def text_spatial_map(self, text_ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Convert region-word tokens into a smooth spatial field without target metadata."""
        token_xy = self.spatial_token_coordinates[text_ids]
        token_mask = self.spatial_token_mask[text_ids].squeeze(-1)
        grid = coordinate_grid(height, width, device=text_ids.device, dtype=token_xy.dtype)
        query_xy = grid.flatten(2).transpose(1, 2)
        distance = (query_xy[:, :, None, :] - token_xy[:, None, :, :]).square().sum(dim=-1)
        fields = torch.exp(-distance / (2.0 * 0.28**2)) * token_mask[:, None, :]
        spatial = fields.amax(dim=-1).reshape(text_ids.shape[0], 1, height, width)
        return spatial

    def object_grounding_feature(
        self,
        text_ids: torch.Tensor,
        encoded_text: torch.Tensor,
        height: int,
        width: int,
        *,
        scale: str,
    ) -> torch.Tensor | None:
        """Subclass hook; the first-round model has no object-level grounding."""
        return None

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        disable_cross_attention: bool = False,
        disable_coordinates: bool = False,
        disable_cross_attention_16: bool = False,
        disable_cross_attention_8: bool = False,
        disable_spatial_attention_bias: bool = False,
        disable_text_spatial_map: bool = False,
        disable_object_grounding: bool = False,
    ) -> DenoiserOutputV2:
        if noisy_latent.ndim != 4 or noisy_latent.shape[1:] != (self.latent_channels, 16, 16):
            raise ValueError("denoiser expects [B,C,16,16] latent")
        if timestep.shape != (noisy_latent.shape[0],):
            raise ValueError("timestep shape mismatch")
        text = self.encode_text(text_ids, text_mask, disable_coordinates=disable_coordinates)
        token_coordinates = self.spatial_token_coordinates[text_ids]
        token_coordinate_mask = self.spatial_token_mask[text_ids]
        weights_for_pool = text_mask.to(text.dtype).unsqueeze(-1)
        pooled_text = (text * weights_for_pool).sum(dim=1) / weights_for_pool.sum(dim=1).clamp_min(1.0)
        time = self.time_embedding(timestep) + self.text_time(pooled_text)

        grid16 = coordinate_grid(16, 16, device=noisy_latent.device, dtype=noisy_latent.dtype)
        value16 = self.input(noisy_latent)
        if not disable_coordinates:
            value16 = value16 + self.input_coordinate(grid16)
        if not disable_text_spatial_map:
            value16 = value16 + self.text_spatial_16(self.text_spatial_map(text_ids, 16, 16).to(value16.dtype))
        if not disable_object_grounding:
            grounded16 = self.object_grounding_feature(text_ids, text, 16, 16, scale="16x16")
            if grounded16 is not None:
                value16 = value16 + grounded16.to(value16.dtype)
        value16 = self.block16a(value16, time)
        value16, weights16 = self.cross_attention_16(
            value16,
            text,
            text_mask,
            token_coordinates,
            token_coordinate_mask,
            disable_attention=disable_cross_attention or disable_cross_attention_16,
            disable_coordinates=disable_coordinates,
            disable_spatial_attention_bias=disable_spatial_attention_bias,
        )
        skip = self.block16b(value16, time)

        hidden = self.down(skip)
        if not disable_coordinates:
            grid8 = coordinate_grid(8, 8, device=noisy_latent.device, dtype=noisy_latent.dtype)
            hidden = hidden + self.down_coordinate(grid8)
        if not disable_text_spatial_map:
            hidden = hidden + self.text_spatial_8(self.text_spatial_map(text_ids, 8, 8).to(hidden.dtype))
        if not disable_object_grounding:
            grounded8 = self.object_grounding_feature(text_ids, text, 8, 8, scale="8x8")
            if grounded8 is not None:
                hidden = hidden + grounded8.to(hidden.dtype)
        hidden = self.block8a(hidden, time)
        hidden, weights8 = self.cross_attention_8(
            hidden,
            text,
            text_mask,
            token_coordinates,
            token_coordinate_mask,
            disable_attention=disable_cross_attention or disable_cross_attention_8,
            disable_coordinates=disable_coordinates,
            disable_spatial_attention_bias=disable_spatial_attention_bias,
        )
        hidden = self.block8b(hidden, time)
        up = self.up_projection(F.interpolate(hidden, scale_factor=2, mode="nearest"))
        merged = self.merge(torch.cat((up, skip), dim=1))
        predicted = self.output(self.block_out(merged, time))
        return DenoiserOutputV2(predicted, weights16, weights8)


REGION_TOKEN_COORDINATES: dict[str, tuple[float, float]] = {
    "左上区域": (-0.7, -0.7),
    "上方区域": (0.0, -0.7),
    "右上区域": (0.7, -0.7),
    "左侧区域": (-0.7, 0.0),
    "中央区域": (0.0, 0.0),
    "右侧区域": (0.7, 0.0),
    "左下区域": (-0.7, 0.7),
    "下方区域": (0.0, 0.7),
    "右下区域": (0.7, 0.7),
}


def resolve_spatial_token_coordinates(token_to_id: Mapping[str, int]) -> dict[int, tuple[float, float]]:
    missing = sorted(set(REGION_TOKEN_COORDINATES) - set(token_to_id))
    if missing:
        raise ValueError(f"vocabulary lacks region tokens: {missing}")
    return {int(token_to_id[token]): xy for token, xy in REGION_TOKEN_COORDINATES.items()}
