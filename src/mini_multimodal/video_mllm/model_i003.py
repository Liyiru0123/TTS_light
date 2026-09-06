"""Tiny divided space-time and object-query video-language model for I003."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import CausalVideoDecoderBlock, SharedFrameEncoder, TemporalBlock


class DividedSpaceTimeBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        disable_temporal_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, spatial_count, width = value.shape
        temporal_weights = value.new_zeros(
            batch, spatial_count, self.temporal_attention.num_heads, frames, frames
        )
        if not disable_temporal_attention:
            temporal = value.transpose(1, 2).reshape(batch * spatial_count, frames, width)
            normed = self.temporal_norm(temporal)
            attended, weights = self.temporal_attention(
                normed, normed, normed, need_weights=True, average_attn_weights=False
            )
            temporal = temporal + attended
            value = temporal.reshape(batch, spatial_count, frames, width).transpose(1, 2)
            temporal_weights = weights.reshape(
                batch, spatial_count, weights.shape[1], frames, frames
            )
        spatial = value.reshape(batch * frames, spatial_count, width)
        normed = self.spatial_norm(spatial)
        attended, spatial_weights = self.spatial_attention(
            normed, normed, normed, need_weights=True, average_attn_weights=False
        )
        spatial = spatial + attended
        spatial = spatial + self.ff(self.ff_norm(spatial))
        value = spatial.reshape(batch, frames, spatial_count, width)
        spatial_weights = spatial_weights.reshape(
            batch, frames, spatial_weights.shape[1], spatial_count, spatial_count
        )
        return value, temporal_weights, spatial_weights


@dataclass(slots=True)
class TinyVideoMLLMI003Output:
    classification_logits: torch.Tensor
    answer_logits: torch.Tensor | None
    onset_logits: torch.Tensor
    spatiotemporal_tokens: torch.Tensor
    global_temporal_tokens: torch.Tensor
    object_tracks: torch.Tensor
    relation_token: torch.Tensor
    text_tokens: torch.Tensor
    temporal_attention: torch.Tensor
    spatial_attention: torch.Tensor
    global_temporal_attention: torch.Tensor
    object_query_attention: torch.Tensor
    track_attention: torch.Tensor
    relation_attention: torch.Tensor
    decoder_attention: torch.Tensor | None


class TinyVideoMLLMI003(nn.Module):
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
        timesformer_layers: int = 1,
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
            raise ValueError("I003 requires 8x8 grids and 8 frames")
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
        self.time_embedding = nn.Parameter(torch.empty(frame_count, d_model))
        self.video_modality = nn.Parameter(torch.empty(1, 1, 1, d_model))
        self.null_video = nn.Parameter(torch.empty(1, 1, 1, d_model))

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

        self.timesformer_blocks = nn.ModuleList(
            DividedSpaceTimeBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(timesformer_layers)
        )
        self.spacetime_norm = nn.LayerNorm(d_model)

        # The I002-style fixed-order layout path is retained as a global-context
        # residual branch.  It protects direction/appearance/disappearance
        # capabilities while the object queries specialise in local evidence.
        self.global_layout_norm = nn.LayerNorm(spatial_grid * spatial_grid * d_model)
        self.global_layout_projection = nn.Linear(
            spatial_grid * spatial_grid * d_model, d_model
        )
        self.global_temporal_cls = nn.Parameter(torch.empty(1, 1, d_model))
        self.global_temporal_block = TemporalBlock(
            d_model, num_heads, ff_dim, dropout
        )
        self.global_temporal_norm = nn.LayerNorm(d_model)

        self.object_query_type = nn.Parameter(torch.empty(2, d_model))
        self.object_query_norm = nn.LayerNorm(d_model)
        self.visual_memory_norm = nn.LayerNorm(d_model)
        self.object_cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.track_temporal_norm = nn.LayerNorm(d_model)
        self.track_temporal_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.track_ff_norm = nn.LayerNorm(d_model)
        self.track_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model)
        )

        self.onset_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.onset_token_type = nn.Parameter(torch.empty(2, d_model))
        self.event_query = nn.Parameter(torch.empty(1, 1, d_model))
        self.relation_query_norm = nn.LayerNorm(d_model)
        self.relation_memory_norm = nn.LayerNorm(d_model)
        self.relation_cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.relation_ff_norm = nn.LayerNorm(d_model)
        self.relation_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
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
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.pad_id].zero_()
        for value in (
            self.row_embedding,
            self.column_embedding,
            self.time_embedding,
            self.video_modality,
            self.null_video,
            self.text_modality,
            self.object_query_type,
            self.onset_token_type,
            self.event_query,
            self.global_temporal_cls,
        ):
            nn.init.normal_(value, std=0.02)

    def encode_text(
        self, question_ids: torch.Tensor, *, mask_text: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if question_ids.ndim != 2 or question_ids.shape[1] > self.max_question_length:
            raise ValueError("invalid I003 question token shape")
        padding = question_ids.eq(self.pad_id)
        ids = torch.where(
            padding,
            question_ids,
            torch.full_like(question_ids, self.mask_id),
        ) if mask_text else question_ids
        positions = torch.arange(ids.shape[1], device=ids.device)[None]
        hidden = self.token_embedding(ids) + self.text_position(positions) + self.text_modality
        hidden = self.text_encoder(hidden, src_key_padding_mask=padding)
        return self.text_norm(hidden), padding

    def _descriptor_summaries(
        self,
        text: torch.Tensor,
        object_query_masks: torch.Tensor | None,
        *,
        global_query_ablation: bool,
    ) -> torch.Tensor:
        if object_query_masks is None or global_query_ablation:
            return text[:, 0, None].expand(-1, 2, -1)
        if tuple(object_query_masks.shape[:2]) != (text.shape[0], 2):
            raise ValueError("object_query_masks must have shape [B,2,L]")
        masks = object_query_masks[:, :, : text.shape[1]].to(text.dtype)
        denominator = masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.einsum("bol,bld->bod", masks, text) / denominator

    def encode_video(
        self,
        video: torch.Tensor,
        text: torch.Tensor,
        object_query_masks: torch.Tensor | None,
        *,
        mask_video: bool = False,
        disable_timesformer_temporal: bool = False,
        global_query_ablation: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, frames = video.shape[:2]
        spatial_count = self.spatial_grid * self.spatial_grid
        if frames != self.frame_count:
            raise ValueError(f"expected exactly {self.frame_count} frames")
        if mask_video:
            spatial = self.null_video.expand(batch, frames, spatial_count, self.d_model)
        else:
            encoded = self.frame_encoder(video)
            spatial = encoded.flatten(3).transpose(2, 3)
        spatial_position = (
            self.row_embedding[:, None] + self.column_embedding[None, :]
        ).reshape(1, 1, spatial_count, self.d_model)
        spatial = (
            spatial
            + spatial_position
            + self.time_embedding[None, :, None]
            + self.video_modality
        )
        temporal_attention = spatial.new_zeros(
            batch, spatial_count, 1, frames, frames
        )
        spatial_attention = spatial.new_zeros(
            batch, frames, 1, spatial_count, spatial_count
        )
        for block in self.timesformer_blocks:
            spatial, temporal_attention, spatial_attention = block(
                spatial,
                disable_temporal_attention=disable_timesformer_temporal,
            )
        spatial = self.spacetime_norm(spatial)

        global_frames = self.global_layout_projection(
            self.global_layout_norm(
                spatial.reshape(batch, frames, spatial_count * self.d_model)
            )
        )
        global_query = self.global_temporal_cls.expand(batch, -1, -1) + text[:, :1]
        global_temporal = torch.cat((global_query, global_frames), dim=1)
        global_temporal, global_temporal_attention = self.global_temporal_block(
            global_temporal
        )
        global_temporal = self.global_temporal_norm(global_temporal)

        descriptors = self._descriptor_summaries(
            text, object_query_masks, global_query_ablation=global_query_ablation
        )
        query = (
            descriptors[:, :, None]
            + self.object_query_type[None, :, None]
            + self.time_embedding[None, None]
        )
        query_flat = query.reshape(batch * 2 * frames, 1, self.d_model)
        memory = spatial[:, None].expand(-1, 2, -1, -1, -1)
        memory_flat = memory.reshape(batch * 2 * frames, spatial_count, self.d_model)
        attended_tracks, query_attention = self.object_cross_attention(
            self.object_query_norm(query_flat),
            self.visual_memory_norm(memory_flat),
            self.visual_memory_norm(memory_flat),
            need_weights=True,
            average_attn_weights=False,
        )
        tracks = (query_flat + attended_tracks)[:, 0].reshape(
            batch, 2, frames, self.d_model
        )
        query_attention = query_attention.reshape(
            batch, 2, frames, query_attention.shape[1], spatial_count
        )

        track_flat = tracks.reshape(batch * 2, frames, self.d_model)
        normed = self.track_temporal_norm(track_flat)
        attended, track_attention = self.track_temporal_attention(
            normed, normed, normed, need_weights=True, average_attn_weights=False
        )
        track_flat = track_flat + attended
        track_flat = track_flat + self.track_ff(self.track_ff_norm(track_flat))
        tracks = track_flat.reshape(batch, 2, frames, self.d_model)
        track_attention = track_attention.reshape(
            batch, 2, track_attention.shape[1], frames, frames
        )

        previous = torch.cat((tracks[:, :, :1], tracks[:, :, :-1]), dim=2)
        delta = tracks - previous
        onset_logits = self.onset_head(torch.cat((tracks, delta), dim=-1)).squeeze(-1)
        onset_probabilities = onset_logits.softmax(dim=-1)
        onset_tokens = torch.einsum(
            "bot,td->bod", onset_probabilities, self.time_embedding
        ) + self.onset_token_type[None]

        relation_memory = torch.cat(
            (
                onset_tokens,
                tracks.reshape(batch, 2 * frames, self.d_model),
                global_temporal,
            ),
            dim=1,
        )
        relation_query = self.event_query.expand(batch, -1, -1) + text[:, :1]
        relation, relation_attention = self.relation_cross_attention(
            self.relation_query_norm(relation_query),
            self.relation_memory_norm(relation_memory),
            self.relation_memory_norm(relation_memory),
            need_weights=True,
            average_attn_weights=False,
        )
        relation = relation_query + relation
        relation = relation + self.relation_ff(self.relation_ff_norm(relation))
        return (
            spatial,
            global_temporal,
            tracks,
            relation,
            onset_logits,
            temporal_attention,
            spatial_attention,
            global_temporal_attention,
            query_attention,
            track_attention,
            relation_attention,
            onset_tokens,
        )

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
            torch.ones(
                answer_input_ids.shape[1],
                answer_input_ids.shape[1],
                dtype=torch.bool,
                device=answer_input_ids.device,
            ),
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
        return F.linear(
            self.decoder_norm(hidden), self.token_embedding.weight, self.output_bias
        ), last

    def forward(
        self,
        video: torch.Tensor,
        question_ids: torch.Tensor,
        answer_input_ids: torch.Tensor | None = None,
        *,
        object_query_masks: torch.Tensor | None = None,
        mask_video: bool = False,
        mask_text: bool = False,
        disable_timesformer_temporal: bool = False,
        global_query_ablation: bool = False,
    ) -> TinyVideoMLLMI003Output:
        text, text_padding = self.encode_text(question_ids, mask_text=mask_text)
        (
            spatial,
            global_temporal,
            tracks,
            relation,
            onset_logits,
            temporal_attention,
            spatial_attention,
            global_temporal_attention,
            query_attention,
            track_attention,
            relation_attention,
            onset_tokens,
        ) = self.encode_video(
            video,
            text,
            object_query_masks,
            mask_video=mask_video,
            disable_timesformer_temporal=disable_timesformer_temporal,
            global_query_ablation=global_query_ablation,
        )
        classification = self.classifier(torch.cat((text[:, 0], relation[:, 0]), dim=-1))
        visual_memory = torch.cat(
            (
                relation,
                onset_tokens,
                tracks.reshape(video.shape[0], 2 * self.frame_count, self.d_model),
                global_temporal,
            ),
            dim=1,
        )
        memory = torch.cat((text, visual_memory), dim=1)
        memory_padding = torch.cat(
            (
                text_padding,
                torch.zeros(
                    visual_memory.shape[:2], dtype=torch.bool, device=video.device
                ),
            ),
            dim=1,
        )
        answer_logits = None
        decoder_attention = None
        if answer_input_ids is not None:
            answer_logits, decoder_attention = self.decode_teacher_forced(
                answer_input_ids, memory, memory_padding
            )
        return TinyVideoMLLMI003Output(
            classification,
            answer_logits,
            onset_logits,
            spatial,
            global_temporal,
            tracks,
            relation,
            text,
            temporal_attention,
            spatial_attention,
            global_temporal_attention,
            query_attention,
            track_attention,
            relation_attention,
            decoder_attention,
        )

    @torch.no_grad()
    def generate(
        self,
        video: torch.Tensor,
        question_ids: torch.Tensor,
        *,
        object_query_masks: torch.Tensor | None = None,
        mask_video: bool = False,
        mask_text: bool = False,
        disable_timesformer_temporal: bool = False,
        global_query_ablation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generated = torch.full(
            (video.shape[0], 1), self.bos_id, dtype=torch.long, device=video.device
        )
        probabilities = []
        finished = torch.zeros(video.shape[0], dtype=torch.bool, device=video.device)
        for _ in range(self.max_answer_length - 1):
            output = self(
                video,
                question_ids,
                generated,
                object_query_masks=object_query_masks,
                mask_video=mask_video,
                mask_text=mask_text,
                disable_timesformer_temporal=disable_timesformer_temporal,
                global_query_ablation=global_query_ablation,
            )
            assert output.answer_logits is not None
            distribution = output.answer_logits[:, -1].softmax(dim=-1)
            next_id = distribution.argmax(dim=-1)
            next_probability = distribution.gather(1, next_id[:, None]).squeeze(1)
            next_id = torch.where(
                finished, torch.full_like(next_id, self.eos_id), next_id
            )
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
