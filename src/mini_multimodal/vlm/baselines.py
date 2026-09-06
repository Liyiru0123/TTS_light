"""Trainable single-modality baselines for P2-W03."""

from __future__ import annotations

import torch
from torch import nn

from .model import VisualConnector


class TextOnlyVQABaseline(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        answer_class_count: int,
        pad_id: int,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        max_question_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=pad_id)
        self.position = nn.Embedding(max_question_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, answer_class_count)

    def forward(self, question_ids: torch.Tensor) -> torch.Tensor:
        padding = question_ids.eq(self.pad_id)
        positions = torch.arange(question_ids.shape[1], device=question_ids.device)[None]
        hidden = self.embedding(question_ids) + self.position(positions)
        hidden = self.norm(self.encoder(hidden, src_key_padding_mask=padding))
        return self.classifier(hidden[:, 0])


class ImageOnlyVQABaseline(nn.Module):
    def __init__(
        self,
        *,
        answer_class_count: int,
        d_model: int,
        num_heads: int,
        visual_channels: int,
        visual_height: int,
        visual_width: int,
    ) -> None:
        super().__init__()
        self.connector = VisualConnector(visual_channels, d_model)
        self.row = nn.Parameter(torch.randn(visual_height, d_model) * 0.02)
        self.column = nn.Parameter(torch.randn(visual_width, d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, answer_class_count)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        tokens = self.connector(visual).flatten(2).transpose(1, 2)
        position = (self.row[:, None, :] + self.column[None, :, :]).reshape(1, tokens.shape[1], -1)
        tokens = tokens + position
        query = self.query.expand(visual.shape[0], -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.classifier(self.norm(pooled[:, 0]))
