"""A CPU-sized convolution + Conformer-style encoder with a CTC head."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def _conv_length(lengths: torch.Tensor, kernel: int = 5, stride: int = 2, padding: int = 2) -> torch.Tensor:
    return torch.div(lengths + 2 * padding - (kernel - 1) - 1, stride, rounding_mode="floor") + 1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        position = torch.arange(length, device=x.device, dtype=x.dtype).unsqueeze(1)
        scale = torch.exp(torch.arange(0, self.d_model, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / self.d_model))
        encoding = torch.zeros(length, self.d_model, device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        return x + encoding.unsqueeze(0)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.channel_norm = nn.LayerNorm(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = nn.functional.glu(self.pointwise_in(y), dim=1)
        y = y.masked_fill(padding_mask.unsqueeze(1), 0.0)
        y = self.depthwise(y).transpose(1, 2)
        y = nn.functional.silu(self.channel_norm(y)).transpose(1, 2)
        y = self.dropout(self.pointwise_out(y)).transpose(1, 2)
        return y.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class TinyConformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_dim, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.convolution = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_dim, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, *, return_attention: bool = False):
        x = x + 0.5 * self.ff1(x)
        normalized = self.attn_norm(x)
        attended, weights = self.attention(normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=return_attention, average_attn_weights=False)
        x = x + self.attn_dropout(attended)
        x = x + self.convolution(x, padding_mask)
        x = self.final_norm(x + 0.5 * self.ff2(x))
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0), weights


@dataclass
class TinyASROutput:
    logits: torch.Tensor
    output_lengths: torch.Tensor
    padding_mask: torch.Tensor
    attention: torch.Tensor | None


class TinyASR(nn.Module):
    def __init__(self, *, vocabulary_size: int, n_mels: int = 64, d_model: int = 96, num_heads: int = 4, ff_dim: int = 192, num_blocks: int = 2, conv_kernel_size: int = 15, dropout: float = 0.1, subsampling_factor: int = 8):
        super().__init__()
        if subsampling_factor != 8:
            raise ValueError("This teaching model fixes three stride-2 layers (factor 8)")
        self.subsampling = nn.ModuleList([
            nn.Conv1d(n_mels, d_model, 5, stride=2, padding=2),
            nn.Conv1d(d_model, d_model, 5, stride=2, padding=2),
            nn.Conv1d(d_model, d_model, 5, stride=2, padding=2),
        ])
        self.position = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([TinyConformerBlock(d_model, num_heads, ff_dim, conv_kernel_size, dropout) for _ in range(num_blocks)])
        self.classifier = nn.Linear(d_model, vocabulary_size)

    def forward(self, logmels: torch.Tensor, input_lengths: torch.Tensor, *, return_attention: bool = False) -> TinyASROutput:
        if logmels.ndim != 3:
            raise ValueError("logmels must have shape [batch,time,n_mels]")
        output_lengths = input_lengths.clone()
        encoded = logmels.transpose(1, 2)
        for convolution in self.subsampling:
            encoded = nn.functional.gelu(convolution(encoded))
            output_lengths = _conv_length(output_lengths)
            layer_time = torch.arange(encoded.shape[2], device=encoded.device).unsqueeze(0)
            encoded = encoded.masked_fill((layer_time >= output_lengths.unsqueeze(1)).unsqueeze(1), 0.0)
        encoded = encoded.transpose(1, 2)
        time = torch.arange(encoded.shape[1], device=encoded.device).unsqueeze(0)
        padding_mask = time >= output_lengths.unsqueeze(1)
        encoded = self.position(encoded).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        attention = None
        for index, block in enumerate(self.blocks):
            encoded, weights = block(encoded, padding_mask, return_attention=return_attention and index == len(self.blocks) - 1)
            if weights is not None:
                attention = weights
        return TinyASROutput(self.classifier(encoded), output_lengths, padding_mask, attention)
