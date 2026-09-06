"""TinyT2V (P3-W04): frozen spatial denoiser + trainable temporal modules.

AnimateDiff-style:
- spatial U-Net (P2-W04 TinyImageDenoiser weights) frozen, run per frame
- TemporalTransformer (dimension reshape + 1D attention over frames) inserted
  at the 16x16 and 8x8 resolution levels, trainable only
- text embedding extended with motion tokens; base rows frozen, new rows trainable
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from mini_multimodal.ldm.model import DenoiserOutput, SinusoidalTimeEmbedding, TimeResidualBlock


class SinusoidalCoordinateEmbedding(nn.Module):
    """Embed per-frame (x, y) normalized coordinates into a conditioning vector."""

    def __init__(self, out_dim: int, coord_dim: int = 2, num_frequencies: int = 8) -> None:
        super().__init__()
        if out_dim % (coord_dim * 2 * num_frequencies) != 0:
            raise ValueError("out_dim must be divisible by coord_dim*2*num_frequencies")
        self.coord_dim = coord_dim
        self.num_frequencies = num_frequencies
        self.proj = nn.Linear(coord_dim * 2 * num_frequencies, out_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N, coord_dim] in [0, 1]
        freqs = 2.0 ** torch.arange(self.num_frequencies, device=coords.device, dtype=torch.float32)
        angles = coords[:, :, None] * math.pi * freqs[None, None, :]  # [N, coord_dim, F]
        feats = torch.cat((angles.sin(), angles.cos()), dim=1).flatten(1)  # [N, coord_dim*2F]
        return self.proj(feats)


class MotionPredictor(nn.Module):
    """Text (pooled encoding) -> per-frame object trajectory [B, F, 2] in [0,1].

    The video-generation analogue of TinyTTS's DurationPredictor: structured
    correspondence (text -> motion) learned by a dedicated predictor.
    """

    def __init__(self, text_dim: int = 96, hidden: int = 128, frames: int = 8) -> None:
        super().__init__()
        self.frames = frames
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, frames * 2),
        )

    def forward(self, pooled_text: torch.Tensor) -> torch.Tensor:
        value = self.mlp(pooled_text).reshape(-1, self.frames, 2)
        return torch.sigmoid(value)


class TemporalTransformer(nn.Module):
    """AnimateDiff motion module: dimension reshape + 1D temporal attention.

    forward x: [B, C, F, H, W]
      reshape ① -> [B*H*W, C, F] -> [B*H*W, F, C]
      temporal self-attention over F (with sinusoidal position embedding)
      optional text cross-attention (motion semantics injected at temporal level)
      reshape ② -> [B, C, F, H, W]
    """

    def __init__(self, channels: int, heads: int = 4, ff_mult: int = 2,
                 num_layers: int = 1, max_frames: int = 8, text_dim: int | None = None,
                 zero_init_gate: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.max_frames = max_frames
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult), nn.GELU(),
            nn.Linear(channels * ff_mult, channels),
        )
        self.text_dim = text_dim
        self.zero_init_gate = bool(zero_init_gate)
        if self.zero_init_gate:
            self.output_gate = nn.Parameter(torch.tensor(0.0))
        if text_dim is not None:
            self.text_proj = nn.Linear(text_dim, channels)
            self.norm_text = nn.LayerNorm(channels)
            self.cross_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        position = torch.zeros(max_frames, channels)
        for frame in range(max_frames):
            for i in range(0, channels, 2):
                angle = frame / (10000.0 ** (i / max(channels, 1)))
                position[frame, i] = math.sin(angle)
                if i + 1 < channels:
                    position[frame, i + 1] = math.cos(angle)
        self.register_buffer("position", position)

    def forward(self, value: torch.Tensor, text: torch.Tensor | None = None) -> torch.Tensor:
        batch, channels, frames, height, width = value.shape
        # dimension reshape ① : [B, C, F, H, W] -> [B*H*W, F, C]
        reshaped = value.permute(0, 3, 4, 1, 2).reshape(batch * height * width, channels, frames)
        base = reshaped.transpose(1, 2)
        sequence = base + self.position[:frames]  # [B*H*W, F, C]
        hidden = self.norm1(sequence)
        attended, _ = self.attn(hidden, hidden, hidden)
        sequence = sequence + attended
        if text is not None and self.cross_attn is not None:
            # text cross-attention: [B, H*W*F, C] attends over text tokens [B, L, C]
            text_h = self.text_proj(text)
            seq_b = sequence.reshape(batch, height * width * frames, channels)
            seq_b = seq_b + self.cross_attn(self.norm_text(seq_b), text_h, text_h)[0]
            sequence = seq_b.reshape(batch * height * width, frames, channels)
        sequence = sequence + self.ff(self.norm2(sequence))
        if self.zero_init_gate:
            sequence = base + torch.tanh(self.output_gate) * (sequence - base)
        # dimension reshape ② : [B*H*W, F, C] -> [B, C, F, H, W]
        restored = sequence.transpose(1, 2).reshape(batch, height, width, channels, frames)
        return restored.permute(0, 3, 4, 1, 2)


class TinyVideoDenoiser(nn.Module):
    """Video latent denoiser: [B, 4, F, 16, 16] -> predicted noise.

    Spatial submodules share TinyImageDenoiser's structure/names so P2-W04
    weights load directly; they are frozen. Temporal modules + extended
    text-embedding rows are the only trainable parameters.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        base_vocab_size: int,
        latent_channels: int = 4,
        base_channels: int = 48,
        text_dim: int = 96,
        time_dim: int = 96,
        attention_heads: int = 4,
        max_text_length: int = 128,
        temporal_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.base_vocab_size = int(base_vocab_size)
        self.latent_channels = latent_channels
        self.max_text_length = int(max_text_length)
        hidden = base_channels * 2
        self.hidden_channels = hidden
        # text embedding: base rows frozen, extended (motion) rows trainable
        self.embedding_base = nn.Embedding(base_vocab_size, text_dim, padding_idx=0)
        self.embedding_ext = nn.Embedding(max(1, vocab_size - base_vocab_size), text_dim)
        self.text_position = nn.Parameter(torch.randn(max_text_length, text_dim) * 0.01)
        text_layer = nn.TransformerEncoderLayer(
            d_model=text_dim, nhead=attention_heads, dim_feedforward=text_dim * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=1)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.text_time = nn.Sequential(nn.Linear(text_dim, time_dim), nn.SiLU(),
                                       nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(latent_channels + 1, base_channels, 3, padding=1)
        self.block16 = TimeResidualBlock(base_channels, time_dim)
        self.down = nn.Conv2d(base_channels, hidden, 4, stride=2, padding=1)
        self.block8a = TimeResidualBlock(hidden, time_dim)
        self.cross_attention = nn.MultiheadAttention(hidden, attention_heads, dropout=0.0,
                                                     batch_first=True)
        self.block8b = TimeResidualBlock(hidden, time_dim)
        self.up_projection = nn.Conv2d(hidden, base_channels, 3, padding=1)
        self.merge = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.block_out = TimeResidualBlock(base_channels, time_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels), nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, 3, padding=1),
        )
        cfg = temporal_cfg or {}
        self.text_dim = text_dim
        self.temporal16 = TemporalTransformer(
            base_channels, heads=int(cfg.get("heads", 4)), ff_mult=int(cfg.get("ff_mult", 2)),
            num_layers=int(cfg.get("num_layers", 1)), max_frames=int(cfg.get("max_frames", 8)),
            text_dim=int(cfg.get("text_dim", text_dim)),
            zero_init_gate=bool(cfg.get("zero_init_gate", False)))
        self.temporal8 = TemporalTransformer(
            hidden, heads=int(cfg.get("heads", 4)), ff_mult=int(cfg.get("ff_mult", 2)),
            num_layers=int(cfg.get("num_layers", 1)), max_frames=int(cfg.get("max_frames", 8)),
            text_dim=int(cfg.get("text_dim", text_dim)),
            zero_init_gate=bool(cfg.get("zero_init_gate", False)))
        self.multi_scale_trajectory = bool(cfg.get("multi_scale_trajectory", False))
        if self.multi_scale_trajectory:
            # ControlNet-style lightweight residual adapters.  The original
            # heatmap input is retained; these paths prevent spatial control
            # from being washed out before/after downsampling.
            self.trajectory16 = nn.Conv2d(1, base_channels, 3, padding=1)
            self.trajectory8 = nn.Conv2d(1, hidden, 3, padding=1)
        # per-frame position conditioning: spatial Gaussian heatmap (trajectory channel)
        # (the position_embedding module is removed; heatmap is the spatial condition)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        base_ids = token_ids.clamp(max=self.base_vocab_size - 1)
        ext_ids = (token_ids - self.base_vocab_size).clamp(min=0)
        ext_mask = (token_ids >= self.base_vocab_size).float().unsqueeze(-1)
        return self.embedding_base(base_ids) + self.embedding_ext(ext_ids) * ext_mask

    def encode_text(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape != token_mask.shape:
            raise ValueError("text ids/mask shape mismatch")
        if token_ids.shape[1] > self.max_text_length or not token_mask.any(dim=1).all():
            raise ValueError("text sequence is empty or too long")
        value = self.embed(token_ids) + self.text_position[: token_ids.shape[1]][None, :, :]
        return self.text_encoder(value, src_key_padding_mask=~token_mask)

    @staticmethod
    def _temporal(module: nn.Module, value: torch.Tensor, batch: int, frames: int,
                  text: torch.Tensor | None = None) -> torch.Tensor:
        channels, height, width = value.shape[1:]
        video = value.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        video = module(video, text)
        return video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)

    def pool_text(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        text = self.encode_text(token_ids, token_mask)
        weights = token_mask.to(text.dtype).unsqueeze(-1)
        return (text * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def trajectory_heatmap(trajectory: torch.Tensor, grid: int = 16, sigma: float = 1.5) -> torch.Tensor:
        """Trajectory [B,F,2] in [0,1] -> Gaussian heatmap [B,1,F,grid,grid]."""
        batch, frames, _ = trajectory.shape
        coords = (trajectory * (grid - 1)).reshape(batch, frames, 2)  # [B,F,2]
        gy, gx = torch.meshgrid(torch.arange(grid, dtype=torch.float32, device=trajectory.device),
                                torch.arange(grid, dtype=torch.float32, device=trajectory.device),
                                indexing="ij")
        px = coords[:, :, 0].view(batch, frames, 1, 1)
        py = coords[:, :, 1].view(batch, frames, 1, 1)
        heatmap = torch.exp(-((gx[None, None] - px) ** 2 + (gy[None, None] - py) ** 2) / (2 * sigma * sigma))
        return heatmap.unsqueeze(1)  # [B,1,F,grid,grid]

    def forward(
        self,
        noisy_video: torch.Tensor,
        timestep: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        disable_cross_attention: bool = False,
        use_temporal: bool = True,
        trajectory: torch.Tensor | None = None,
    ) -> DenoiserOutput:
        if noisy_video.ndim != 5 or noisy_video.shape[1] != self.latent_channels:
            raise ValueError("denoiser expects [B,C,F,16,16] video latent")
        batch, _, frames, height, width = noisy_video.shape
        if height != 16 or width != 16:
            raise ValueError("video latent spatial size must be 16x16")
        if timestep.shape != (batch,):
            raise ValueError("timestep shape mismatch")
        if trajectory is None:
            trajectory = torch.zeros(batch, frames, 2, device=noisy_video.device, dtype=noisy_video.dtype)
        heatmap = self.trajectory_heatmap(trajectory)
        noisy_video = torch.cat((noisy_video, heatmap), dim=1)  # [B,5,F,16,16]
        text = self.encode_text(token_ids, token_mask)  # [B, L, text_dim]
        pooled_text = self.pool_text(token_ids, token_mask)
        # per-frame spatial processing (frozen, shared weights)
        x = noisy_video.permute(0, 2, 1, 3, 4).reshape(batch * frames, self.latent_channels + 1, 16, 16)
        time_f = timestep.repeat_interleave(frames)
        time = self.time_embedding(time_f) + self.text_time(pooled_text.repeat_interleave(frames, dim=0))
        skip = self.block16(self.input(x), time)
        if self.multi_scale_trajectory:
            condition16 = heatmap.permute(0, 2, 1, 3, 4).reshape(batch * frames, 1, 16, 16)
            skip = skip + self.trajectory16(condition16)
        if use_temporal:
            skip = self._temporal(self.temporal16, skip, batch, frames, text)
        hidden = self.block8a(self.down(skip), time)
        if self.multi_scale_trajectory:
            condition8 = F.avg_pool2d(condition16, kernel_size=2)
            hidden = hidden + self.trajectory8(condition8)
        queries = hidden.flatten(2).transpose(1, 2)
        text_f = text.repeat_interleave(frames, dim=0)
        mask_f = token_mask.repeat_interleave(frames, dim=0)
        if disable_cross_attention:
            attended = torch.zeros_like(queries)
            weights = torch.zeros(
                queries.shape[0], self.cross_attention.num_heads, queries.shape[1],
                text.shape[1], device=queries.device,
            )
        else:
            attended, weights = self.cross_attention(
                queries, text_f, text_f, key_padding_mask=~mask_f,
                need_weights=True, average_attn_weights=False,
            )
        hidden = hidden + attended.transpose(1, 2).reshape_as(hidden)
        hidden = self.block8b(hidden, time)
        if use_temporal:
            hidden = self._temporal(self.temporal8, hidden, batch, frames, text)
        up = F.interpolate(hidden, scale_factor=2, mode="nearest")
        up = self.up_projection(up)
        merged = self.merge(torch.cat((up, skip), dim=1))
        result = self.output(self.block_out(merged, time))
        result = result.reshape(batch, frames, self.latent_channels, 16, 16).permute(0, 2, 1, 3, 4)
        return DenoiserOutput(result, weights)

    def load_frozen_spatial(self, spatial_state: dict, *, freeze_spatial: bool = True) -> None:
        """Load P2-W04 TinyImageDenoiser state into spatial submodules.

        freeze_spatial=True  -> only temporal/embedding_ext/cross_attention trainable
        freeze_spatial=False -> warm start + full fine-tune (I006: spatial heatmap trajectory
                               conditioning needs the spatial blocks to place content at the
                               conditioned position)
        """
        own = self.state_dict()
        compatible = {key: value for key, value in spatial_state.items()
                      if key in own and key not in ("text_embedding.weight", "input.weight", "input.bias")}
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        if unexpected:
            raise ValueError(f"unexpected state keys: {sorted(unexpected)[:10]}")
        with torch.no_grad():
            if "text_embedding.weight" in spatial_state:
                base_weight = spatial_state["text_embedding.weight"]
                n = min(base_weight.shape[0], self.embedding_base.weight.shape[0])
                self.embedding_base.weight[:n].copy_(base_weight[:n])
            # input conv: warm-start first 4 channels, zero the trajectory heatmap channel
            if "input.weight" in spatial_state:
                base_input = spatial_state["input.weight"]  # [base, 4, 3, 3]
                self.input.weight[:, :4].copy_(base_input)
                self.input.weight[:, 4:].zero_()
            if "input.bias" in spatial_state:
                self.input.bias.copy_(spatial_state["input.bias"])
        # verify all spatial keys were covered (temporal/embedding_ext are intentionally new)
        spatial_keys = set(spatial_state) - {"text_embedding.weight"}
        own_keys = set(self.state_dict())
        uncovered = spatial_keys - own_keys
        if uncovered:
            raise ValueError(f"spatial state keys not present in model: {sorted(uncovered)[:10]}")
        # freeze policy
        for name, parameter in self.named_parameters():
            if not freeze_spatial:
                parameter.requires_grad = True
            elif name.startswith("temporal"):
                parameter.requires_grad = True
            elif name == "embedding_ext.weight":
                parameter.requires_grad = True
            elif name.startswith("cross_attention"):
                parameter.requires_grad = True
            elif name.startswith("trajectory16") or name.startswith("trajectory8"):
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
