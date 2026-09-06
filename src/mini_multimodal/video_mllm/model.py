"""A compact, explicitly temporal video-language model for P3-W01."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class SharedFrameEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv2d(24, 40, 3, stride=2, padding=1),
            nn.GroupNorm(5, 40),
            nn.SiLU(),
            nn.Conv2d(40, d_model, 3, stride=2, padding=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or tuple(video.shape[2:]) != (3, 64, 64):
            raise ValueError("video must have shape [B,T,3,64,64]")
        batch, frames = video.shape[:2]
        encoded = self.network(video.reshape(batch * frames, 3, 64, 64))
        if tuple(encoded.shape[-2:]) != (8, 8):
            raise AssertionError("shared frame encoder did not produce an 8x8 grid")
        return encoded.reshape(batch, frames, encoded.shape[1], 8, 8)


class TemporalBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm(value)
        attended, weights = self.attention(
            normed, normed, normed, need_weights=True, average_attn_weights=False
        )
        hidden = value + attended
        return hidden + self.ff(self.ff_norm(hidden)), weights


class CausalVideoDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

    def forward(
        self,
        value: torch.Tensor,
        memory: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        value_padding_mask: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.self_norm(value)
        attended, _ = self.self_attention(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            key_padding_mask=value_padding_mask,
            need_weights=False,
        )
        hidden = value + attended
        cross, weights = self.cross_attention(
            self.cross_norm(hidden),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = hidden + cross
        return hidden + self.ff(self.ff_norm(hidden)), weights


@dataclass(slots=True)
class TinyVideoMLLMOutput:
    classification_logits: torch.Tensor
    answer_logits: torch.Tensor | None
    spatial_tokens: torch.Tensor
    frame_tokens: torch.Tensor
    temporal_tokens: torch.Tensor
    text_tokens: torch.Tensor
    spatial_attention: torch.Tensor
    temporal_attention: torch.Tensor
    decoder_attention: torch.Tensor | None


class TinyVideoMLLM(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        answer_class_count: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        mask_id: int,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        text_layers: int = 1,
        temporal_layers: int = 2,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        spatial_grid: int = 8,
        max_question_length: int = 48,
        max_answer_length: int = 8,
        frame_count: int = 8,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if spatial_grid != 8 or frame_count != 8:
            raise ValueError("validated P3-W01 configuration requires 8x8 grids and 8 frames")
        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.mask_id = int(mask_id)
        self.d_model = int(d_model)
        self.frame_count = int(frame_count)
        self.spatial_grid = int(spatial_grid)
        self.max_question_length = int(max_question_length)
        self.max_answer_length = int(max_answer_length)

        self.frame_encoder = SharedFrameEncoder(d_model)
        self.row_embedding = nn.Parameter(torch.empty(spatial_grid, d_model))
        self.column_embedding = nn.Parameter(torch.empty(spatial_grid, d_model))
        self.video_modality = nn.Parameter(torch.empty(1, 1, d_model))
        self.null_video = nn.Parameter(torch.empty(1, 1, d_model))

        self.token_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=pad_id)
        self.text_position = nn.Embedding(max_question_length, d_model)
        self.answer_position = nn.Embedding(max_answer_length, d_model)
        self.text_modality = nn.Parameter(torch.empty(1, 1, d_model))
        text_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(
            text_layer, num_layers=text_layers, enable_nested_tensor=False
        )
        self.text_norm = nn.LayerNorm(d_model)

        self.spatial_cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.frame_query = nn.Parameter(torch.empty(1, 1, d_model))
        self.frame_query_norm = nn.LayerNorm(d_model)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_layout_norm = nn.LayerNorm(spatial_grid * spatial_grid * d_model)
        self.spatial_layout_projection = nn.Linear(
            spatial_grid * spatial_grid * d_model, d_model
        )
        self.motion_delta_norm = nn.LayerNorm(spatial_grid * spatial_grid * d_model)
        self.motion_delta_projection = nn.Linear(
            spatial_grid * spatial_grid * d_model, d_model
        )
        self.motion_token_type = nn.Parameter(torch.empty(1, 1, d_model))
        self.frame_ff_norm = nn.LayerNorm(d_model)
        self.frame_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model)
        )
        self.time_embedding = nn.Parameter(torch.empty(frame_count, d_model))
        self.temporal_cls = nn.Parameter(torch.empty(1, 1, d_model))
        self.temporal_blocks = nn.ModuleList(
            TemporalBlock(d_model, num_heads, ff_dim, dropout) for _ in range(temporal_layers)
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, answer_class_count),
        )

        self.decoder = nn.ModuleList(
            CausalVideoDecoderBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocabulary_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The answer projection shares this matrix with the input embedding.
        # PyTorch's default Embedding initialisation has unit variance, which
        # makes the tied projection overwhelmingly favour the current input
        # token at the start of training.  Transformer-style small-variance
        # initialisation keeps the initial next-token distribution trainable.
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.pad_id].zero_()
        for value in (
            self.row_embedding,
            self.column_embedding,
            self.video_modality,
            self.null_video,
            self.text_modality,
            self.frame_query,
            self.time_embedding,
            self.temporal_cls,
            self.motion_token_type,
        ):
            nn.init.normal_(value, std=0.02)

    def encode_text(self, question_ids: torch.Tensor, *, mask_text: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if question_ids.ndim != 2 or question_ids.shape[1] > self.max_question_length:
            raise ValueError("invalid P3-W01 question token shape")
        padding = question_ids.eq(self.pad_id)
        ids = question_ids
        if mask_text:
            ids = torch.where(padding, ids, torch.full_like(ids, self.mask_id))
        positions = torch.arange(ids.shape[1], device=ids.device)[None]
        hidden = self.token_embedding(ids) + self.text_position(positions) + self.text_modality
        hidden = self.text_encoder(hidden, src_key_padding_mask=padding)
        return self.text_norm(hidden), padding

    def encode_video(
        self,
        video: torch.Tensor,
        text_tokens: torch.Tensor,
        text_padding: torch.Tensor,
        *,
        mask_video: bool = False,
        disable_time_position: bool = False,
        disable_temporal_module: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, frames = video.shape[:2]
        if frames != self.frame_count:
            raise ValueError(f"expected exactly {self.frame_count} frames")
        if mask_video:
            spatial = self.null_video.reshape(1, 1, 1, -1).expand(
                batch, frames, self.spatial_grid * self.spatial_grid, self.d_model
            )
        else:
            encoded = self.frame_encoder(video)
            spatial = encoded.flatten(3).transpose(2, 3)
        position = (
            self.row_embedding[:, None] + self.column_embedding[None, :]
        ).reshape(1, 1, self.spatial_grid * self.spatial_grid, self.d_model)
        spatial = spatial + position + self.video_modality.reshape(1, 1, 1, self.d_model)
        # The fixed-order projection preserves the complete 8x8 layout.  A
        # question-conditioned Cross-Attention summary is added rather than
        # being solely responsible for learning absolute spatial relations.
        question_summary = text_tokens[:, 0]
        spatial_count = self.spatial_grid * self.spatial_grid
        layout = self.spatial_layout_projection(
            self.spatial_layout_norm(spatial.reshape(batch, frames, spatial_count * self.d_model))
        )
        query = self.frame_query.reshape(1, 1, self.d_model) + question_summary[:, None]
        query = query.expand(batch, frames, self.d_model).reshape(
            batch * frames, 1, self.d_model
        )
        spatial_flat = spatial.reshape(batch * frames, spatial_count, self.d_model)
        attended, spatial_attention = self.spatial_cross_attention(
            self.frame_query_norm(query),
            self.spatial_norm(spatial_flat),
            self.spatial_norm(spatial_flat),
            need_weights=True,
            average_attn_weights=False,
        )
        frame = layout + query[:, 0].reshape(batch, frames, self.d_model)
        frame = frame + attended[:, 0].reshape(batch, frames, self.d_model)
        frame = frame + self.frame_ff(self.frame_ff_norm(frame))
        positioned = frame if disable_time_position else frame + self.time_embedding[None]
        if disable_time_position:
            motion_delta = torch.zeros_like(spatial[:, 0])
        else:
            motion_delta = spatial[:, -1] - spatial[:, 0]
        motion = self.motion_delta_projection(
            self.motion_delta_norm(motion_delta.reshape(batch, spatial_count * self.d_model))
        )[:, None]
        motion = motion + self.motion_token_type
        temporal_query = self.temporal_cls + question_summary[:, None]
        temporal = torch.cat((temporal_query, motion, positioned), dim=1)
        temporal_length = frames + 2
        last_attention = torch.zeros(
            batch,
            self.temporal_blocks[0].attention.num_heads,
            temporal_length,
            temporal_length,
            device=video.device,
        )
        if not disable_temporal_module:
            for block in self.temporal_blocks:
                temporal, last_attention = block(temporal)
        temporal = self.temporal_norm(temporal)
        spatial_attention = spatial_attention.reshape(
            batch,
            frames,
            spatial_attention.shape[1],
            spatial_attention.shape[2],
            spatial_count,
        )
        return spatial, frame, temporal, spatial_attention, last_attention

    def decode_teacher_forced(
        self,
        answer_input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_padding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        padding = answer_input_ids.eq(self.pad_id)
        positions = torch.arange(answer_input_ids.shape[1], device=answer_input_ids.device)[None]
        hidden = self.token_embedding(answer_input_ids) + self.answer_position(positions)
        causal = torch.triu(
            torch.ones(answer_input_ids.shape[1], answer_input_ids.shape[1], dtype=torch.bool, device=answer_input_ids.device),
            diagonal=1,
        )
        last = None
        for block in self.decoder:
            hidden, last = block(
                hidden,
                memory,
                causal_mask=causal,
                value_padding_mask=padding,
                memory_padding_mask=memory_padding,
            )
        return F.linear(self.decoder_norm(hidden), self.token_embedding.weight, self.output_bias), last

    def forward(
        self,
        video: torch.Tensor,
        question_ids: torch.Tensor,
        answer_input_ids: torch.Tensor | None = None,
        *,
        mask_video: bool = False,
        mask_text: bool = False,
        disable_time_position: bool = False,
        disable_temporal_module: bool = False,
    ) -> TinyVideoMLLMOutput:
        text, text_padding = self.encode_text(question_ids, mask_text=mask_text)
        spatial, frame, temporal, spatial_attention, temporal_attention = self.encode_video(
            video,
            text,
            text_padding,
            mask_video=mask_video,
            disable_time_position=disable_time_position,
            disable_temporal_module=disable_temporal_module,
        )
        summary = torch.cat((text[:, 0], temporal[:, 0]), dim=-1)
        classification = self.classifier(summary)
        memory = torch.cat((text, temporal), dim=1)
        memory_padding = torch.cat(
            (text_padding, torch.zeros(temporal.shape[:2], dtype=torch.bool, device=video.device)), dim=1
        )
        answer_logits = None
        decoder_attention = None
        if answer_input_ids is not None:
            answer_logits, decoder_attention = self.decode_teacher_forced(
                answer_input_ids, memory, memory_padding
            )
        return TinyVideoMLLMOutput(
            classification,
            answer_logits,
            spatial,
            frame,
            temporal,
            text,
            spatial_attention,
            temporal_attention,
            decoder_attention,
        )

    @torch.no_grad()
    def generate(
        self,
        video: torch.Tensor,
        question_ids: torch.Tensor,
        *,
        mask_video: bool = False,
        mask_text: bool = False,
        disable_time_position: bool = False,
        disable_temporal_module: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generated = torch.full((video.shape[0], 1), self.bos_id, dtype=torch.long, device=video.device)
        probabilities = []
        finished = torch.zeros(video.shape[0], dtype=torch.bool, device=video.device)
        for _ in range(self.max_answer_length - 1):
            output = self(
                video,
                question_ids,
                generated,
                mask_video=mask_video,
                mask_text=mask_text,
                disable_time_position=disable_time_position,
                disable_temporal_module=disable_temporal_module,
            )
            assert output.answer_logits is not None
            distribution = output.answer_logits[:, -1].softmax(dim=-1)
            next_id = distribution.argmax(dim=-1)
            next_probability = distribution.gather(1, next_id[:, None]).squeeze(1)
            next_id = torch.where(finished, torch.full_like(next_id, self.eos_id), next_id)
            generated = torch.cat((generated, next_id[:, None]), dim=1)
            probabilities.append(next_probability)
            finished |= next_id.eq(self.eos_id)
            if bool(finished.all()):
                break
        probability_tensor = (
            torch.stack(probabilities, dim=1)
            if probabilities
            else torch.empty(video.shape[0], 0, device=video.device)
        )
        return generated[:, 1:], probability_tensor
