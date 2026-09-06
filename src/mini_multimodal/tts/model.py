"""TinySpeechFlow: text -> duration/alignment -> frame-wise fusion -> conditional flow.

Frozen by docs/P3-W03_TinyTTS设计.md (2026-08-18).

Components:
- TinyTextEncoder: bidirectional Transformer, D_text=64
- DurationPredictor: h_i -> log(d_i+1)
- LengthRegulator: N -> T_hat (repeat), raises DurationOverflowError if T_hat > 64
- FrameWiseCondition: W: 64->80, y = x_t + C_64 (input-side one-shot injection)
- TemporalUNet: 1D U-Net 64->32->64, per-time channel norm, time embedding D_e=128
- TinySpeechFlow: full model (velocity field v_theta(x_t, t, c))

The latent from the frozen P3-W02 VAE is [B, 4, 20, 64] (channels, freq, time);
it is flattened to [B, 80, 64] (80 = 4*20) for the U-Net and reshaped back for
the frozen decoder.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class DurationOverflowError(RuntimeError):
    """T_hat > T_max: the text exceeds the teaching model's supported range."""

    def __init__(self, t_hat: int, t_max: int):
        super().__init__(f"duration_overflow: T_hat={t_hat} > T_max={t_max}")
        self.t_hat = int(t_hat)
        self.t_max = int(t_max)


class PerTimeLayerNorm(nn.Module):
    """LayerNorm over channels for each time step (avoids cross-T statistics)."""

    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TimeEmbedding(nn.Module):
    """e = MLP([t, sin(pi t), cos(pi t)]), t in [0, 1]."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        feat = torch.stack([t, torch.sin(t * math.pi), torch.cos(t * math.pi)], dim=-1)
        return self.mlp(feat)  # [B, dim]


class TemporalBlock(nn.Module):
    """Conv -> LN -> time add -> act -> mask -> Conv -> LN -> act -> mask -> add -> mask."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = PerTimeLayerNorm(channels)
        self.time_proj = nn.Linear(time_dim, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = PerTimeLayerNorm(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T], e: [B, time_dim], mask: [B, T] (1=valid)
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_proj(e)[:, :, None]
        h = self.act(h)
        h = h * mask[:, None, :]
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        h = h * mask[:, None, :]
        h = h + x
        h = h * mask[:, None, :]
        return h


class TemporalUNet(nn.Module):
    """1D temporal U-Net velocity field network v_theta(x_t, t, c).

    Time shape: 64 -> 32 -> 64 (true U-Net with skip).
    - downsample: Conv1d(stride=2)
    - upsample: nearest 2x + Conv1d
    - per-time channel normalization (no GroupNorm, avoids length coupling)
    - mask re-applied before each temporal-mixing Conv1d
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 128,
        down_channels: int = 192,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed = TimeEmbedding(time_embed_dim)
        self.stem = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        self.block0 = TemporalBlock(base_channels, time_embed_dim)
        self.down0 = nn.Conv1d(base_channels, down_channels, 3, stride=2, padding=1)
        self.block1 = TemporalBlock(down_channels, time_embed_dim)
        self.up0 = nn.Conv1d(down_channels, base_channels, 3, padding=1)
        self.block2 = TemporalBlock(base_channels, time_embed_dim)
        self.out = nn.Conv1d(base_channels, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, 80, 64], t: [B], mask: [B, 64]
        e = self.time_embed(t)  # [B, time_dim]
        x = x * mask[:, None, :]
        h = self.stem(x) * mask[:, None, :]  # [B, base, 64]
        h = self.block0(h, e, mask)
        skip = h
        h = self.down0(h * mask[:, None, :])  # [B, down, 32]
        t_valid = mask.sum(dim=1)  # [B]
        valid32 = (t_valid + 1) // 2  # ceil(T_valid/2)
        mask32 = (torch.arange(32, device=mask.device)[None, :] < valid32[:, None]).to(mask.dtype)
        h = self.block1(h, e, mask32)
        h = F.interpolate(h, size=64, mode="nearest")
        h = self.up0(h) * mask[:, None, :]  # [B, base, 64]
        h = h + skip
        h = self.block2(h, e, mask)
        v = self.out(h)  # [B, 80, 64]
        v = v * mask[:, None, :]
        return v


class TinyTextEncoder(nn.Module):
    """Bidirectional Transformer text encoder, D_text=64."""

    def __init__(
        self, vocab_size: int, d_text: int = 64, num_layers: int = 2,
        num_heads: int = 4, ff_dim: int = 128, dropout: float = 0.1, max_tokens: int = 40,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.d_text = d_text
        self.max_tokens = max_tokens
        self.embed = nn.Embedding(vocab_size, d_text)
        self.pos = nn.Parameter(torch.zeros(max_tokens, d_text))
        layer = nn.TransformerEncoderLayer(
            d_model=d_text, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_text)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [N] or [B, N]
        if token_ids.ndim == 0:
            raise ValueError("token_ids must be 1-D [N] or 2-D [B,N]")
        single = token_ids.ndim == 1
        if single:
            token_ids = token_ids[None]
        if token_ids.shape[1] > self.max_tokens:
            raise ValueError(f"text length {token_ids.shape[1]} exceeds max_tokens {self.max_tokens}")
        h = self.embed(token_ids) + self.pos[: token_ids.shape[1]]
        h = self.encoder(h)
        h = self.norm(h)
        return h[0] if single else h  # [N, D] or [B, N, D]


class DurationPredictor(nn.Module):
    """h_i -> log(d_i+1). Inference: d_hat = max(0, round(exp(log_d)-1))."""

    def __init__(self, d_text: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_text, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [N, D] or [B, N, D] -> log(d+1): [N, 1] or [B, N, 1]
        return self.net(h)

    def predict_durations(self, h: torch.Tensor) -> torch.Tensor:
        log_d = self.forward(h)
        d = torch.clamp(torch.round(torch.exp(log_d) - 1.0), min=0)
        return d.squeeze(-1)  # [N] or [B, N]


class LengthRegulator(nn.Module):
    """N -> T_hat by repeating each token d_i times. Raises on overflow."""

    def __init__(self, t_max: int = 64):
        super().__init__()
        self.t_max = t_max

    def forward(self, h: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        # h: [N, D], durations: [N] (int)
        if h.shape[0] != durations.shape[0]:
            raise ValueError("h and durations length mismatch")
        t_hat = int(durations.sum().item())
        if t_hat > self.t_max:
            raise DurationOverflowError(t_hat, self.t_max)
        if t_hat == 0:
            raise ValueError("regulator produced zero frames")
        idx = torch.repeat_interleave(torch.arange(h.shape[0], device=h.device), durations.to(h.device))
        return h[idx]  # [T_hat, D]


class FrameWiseCondition(nn.Module):
    """W: D_text -> latent_channels, pad to t_max, output [latent_channels, t_max]."""

    def __init__(self, d_text: int = 64, latent_channels: int = 80, t_max: int = 64):
        super().__init__()
        self.W = nn.Linear(d_text, latent_channels)
        self.t_max = t_max

    def forward(self, H: torch.Tensor, t_max: int | None = None) -> torch.Tensor:
        # H: [T_hat, D]
        t_max = t_max or self.t_max
        C = self.W(H).T  # [latent_channels, T_hat]
        if C.shape[1] < t_max:
            pad = torch.zeros(C.shape[0], t_max - C.shape[1], device=C.device, dtype=C.dtype)
            C = torch.cat([C, pad], dim=1)
        return C  # [latent_channels, t_max]


class TinySpeechFlow(nn.Module):
    """Full TinyTTS model: text -> duration -> condition -> velocity field."""

    def __init__(
        self,
        vocab_size: int,
        text_cfg: dict,
        duration_cfg: dict,
        condition_cfg: dict,
        unet_cfg: dict,
        t_max: int = 64,
    ):
        super().__init__()
        self.t_max = t_max
        self.text_encoder = TinyTextEncoder(vocab_size, **text_cfg)
        self.duration = DurationPredictor(**duration_cfg)
        self.regulator = LengthRegulator(t_max=t_max)
        self.condition = FrameWiseCondition(**condition_cfg, t_max=t_max)
        self.unet = TemporalUNet(**unet_cfg)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(token_ids)  # [N, D] or [B, N, D]

    @torch.inference_mode()
    def plan_duration(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (h, durations, H) for inference."""
        h = self.encode_text(token_ids)  # [N, D]
        durations = self.duration.predict_durations(h)  # [N]
        H = self.regulator(h, durations)  # [T_hat, D]
        return h, durations, H

    def condition_frames(self, H: torch.Tensor) -> torch.Tensor:
        return self.condition(H)  # [80, t_max]

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, C: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x_t: [B, 80, 64], t: [B], C: [B, 80, 64], mask: [B, 64]
        y = (x_t + C) * mask[:, None, :]
        return self.unet(y, t, mask)  # [B, 80, 64]

    def count_parameters(self, trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (not trainable) or p.requires_grad)
