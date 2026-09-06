"""Compact visual-language classifier and causal answer generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class VisualConnector(nn.Module):
    def __init__(self, input_channels: int, d_model: int) -> None:
        super().__init__()
        groups = min(8, d_model)
        self.input = nn.Conv2d(input_channels, d_model, 3, padding=1)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(groups, d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.input(value)
        return hidden + self.block(hidden)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
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
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = query + attended
        hidden = hidden + self.ff(self.ff_norm(hidden))
        return hidden, weights


class CausalDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
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
        hidden = hidden + self.ff(self.ff_norm(hidden))
        return hidden, weights


@dataclass(slots=True)
class TinyVLMOutput:
    classification_logits: torch.Tensor
    answer_logits: torch.Tensor | None
    visual_tokens: torch.Tensor
    text_tokens: torch.Tensor
    fusion_attention: torch.Tensor
    decoder_attention: torch.Tensor | None
    target_coordinates: torch.Tensor
    reference_coordinates: torch.Tensor
    target_attention: torch.Tensor
    reference_attention: torch.Tensor


class TinyVLM(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        answer_class_count: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        mask_id: int,
        target_marker_id: int,
        reference_marker_id: int,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        text_layers: int = 1,
        fusion_reasoning_layers: int = 1,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        visual_height: int = 16,
        visual_width: int = 16,
        visual_channels: int = 4,
        max_question_length: int = 48,
        max_answer_length: int = 8,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.mask_id = int(mask_id)
        self.target_marker_id = int(target_marker_id)
        self.reference_marker_id = int(reference_marker_id)
        self.d_model = int(d_model)
        self.visual_height = int(visual_height)
        self.visual_width = int(visual_width)
        self.visual_channels = int(visual_channels)
        self.max_question_length = int(max_question_length)
        self.max_answer_length = int(max_answer_length)

        self.visual_connector = VisualConnector(visual_channels, d_model)
        rows = torch.linspace(-1.0, 1.0, visual_height)
        columns = torch.linspace(-1.0, 1.0, visual_width)
        grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
        self.register_buffer(
            "spatial_coordinates",
            torch.stack((grid_x, grid_y), dim=-1).reshape(1, visual_height * visual_width, 2),
            persistent=True,
        )
        self.coordinate_projection = nn.Linear(2, d_model, bias=False)
        self.row_embedding = nn.Parameter(torch.empty(visual_height, d_model))
        self.column_embedding = nn.Parameter(torch.empty(visual_width, d_model))
        self.visual_modality = nn.Parameter(torch.empty(1, 1, d_model))
        self.null_visual = nn.Parameter(torch.empty(1, 1, d_model))

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
        self.fusion = CrossAttentionBlock(d_model, num_heads, ff_dim, dropout)
        fusion_reasoning_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_reasoner = nn.TransformerEncoder(
            fusion_reasoning_layer,
            num_layers=fusion_reasoning_layers,
            enable_nested_tensor=False,
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self.grounding_projection = nn.Sequential(
            nn.Linear(8, d_model),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, answer_class_count)
        self.decoder = nn.ModuleList(
            CausalDecoderBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocabulary_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.column_embedding, std=0.02)
        nn.init.normal_(self.visual_modality, std=0.02)
        nn.init.normal_(self.text_modality, std=0.02)
        nn.init.normal_(self.null_visual, std=0.02)

    def encode_visual(
        self,
        visual: torch.Tensor,
        *,
        mask_image: bool = False,
        disable_visual_position: bool = False,
    ) -> torch.Tensor:
        expected = (
            visual.shape[0],
            self.visual_channels,
            self.visual_height,
            self.visual_width,
        )
        if tuple(visual.shape) != expected:
            raise ValueError(f"expected visual tensor [B,4,{self.visual_height},{self.visual_width}]")
        if mask_image:
            return self.null_visual.expand(visual.shape[0], self.visual_height * self.visual_width, -1)
        value = self.visual_connector(visual).flatten(2).transpose(1, 2)
        if not disable_visual_position:
            position = (
                self.row_embedding[:, None, :] + self.column_embedding[None, :, :]
            ).reshape(1, self.visual_height * self.visual_width, self.d_model)
            position = position + self.coordinate_projection(self.spatial_coordinates)
            value = value + position
        return value + self.visual_modality

    def encode_text(
        self,
        question_ids: torch.Tensor,
        *,
        mask_text: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if question_ids.ndim != 2 or question_ids.shape[1] > self.max_question_length:
            raise ValueError("invalid question token shape")
        padding = question_ids.eq(self.pad_id)
        ids = question_ids
        if mask_text:
            ids = torch.where(padding, ids, torch.full_like(ids, self.mask_id))
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
        value = self.token_embedding(ids) + self.text_position(positions) + self.text_modality
        value = self.text_encoder(value, src_key_padding_mask=padding)
        return self.text_norm(value), padding

    def encode_multimodal(
        self,
        visual: torch.Tensor,
        question_ids: torch.Tensor,
        *,
        mask_image: bool = False,
        mask_text: bool = False,
        disable_visual_position: bool = False,
        disable_fusion: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_tokens = self.encode_visual(
            visual,
            mask_image=mask_image,
            disable_visual_position=disable_visual_position,
        )
        text_tokens, text_padding = self.encode_text(question_ids, mask_text=mask_text)
        if disable_fusion:
            fused = text_tokens
            attention = torch.zeros(
                text_tokens.shape[0],
                self.fusion.attention.num_heads,
                text_tokens.shape[1],
                visual_tokens.shape[1],
                device=text_tokens.device,
            )
        else:
            fused, attention = self.fusion(text_tokens, visual_tokens)
        fused = self.fusion_norm(
            self.fusion_reasoner(fused, src_key_padding_mask=text_padding)
        )
        if disable_fusion:
            memory = fused
            memory_padding = text_padding
        else:
            memory = torch.cat((visual_tokens, fused), dim=1)
            memory_padding = torch.cat(
                (
                    torch.zeros(
                        visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device
                    ),
                    text_padding,
                ),
                dim=1,
            )
        return visual_tokens, fused, attention, memory, memory_padding

    def decode_teacher_forced(
        self,
        answer_input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if answer_input_ids.ndim != 2 or answer_input_ids.shape[1] > self.max_answer_length:
            raise ValueError("invalid answer input shape")
        padding = answer_input_ids.eq(self.pad_id)
        positions = torch.arange(answer_input_ids.shape[1], device=answer_input_ids.device)[None, :]
        hidden = self.token_embedding(answer_input_ids) + self.answer_position(positions)
        if conditioning is not None:
            hidden = hidden + conditioning[:, None, :]
        causal = torch.triu(
            torch.ones(
                answer_input_ids.shape[1],
                answer_input_ids.shape[1],
                dtype=torch.bool,
                device=answer_input_ids.device,
            ),
            diagonal=1,
        )
        last_attention = None
        for block in self.decoder:
            hidden, last_attention = block(
                hidden,
                memory,
                causal_mask=causal,
                value_padding_mask=padding,
                memory_padding_mask=memory_padding_mask,
            )
        hidden = self.decoder_norm(hidden)
        logits = F.linear(hidden, self.token_embedding.weight, self.output_bias)
        return logits, last_attention

    def grounding_from_attention(
        self,
        question_ids: torch.Tensor,
        fusion_attention: torch.Tensor,
        *,
        disable_spatial_coordinates: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Turn marker-token Cross-Attention into differentiable spatial centroids."""

        mean_attention = fusion_attention.mean(dim=1)

        def locate(marker_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            marker_mask = question_ids.eq(marker_id).to(mean_attention.dtype)
            presence = marker_mask.sum(dim=1).gt(0)
            selected = (mean_attention * marker_mask.unsqueeze(-1)).sum(dim=1)
            selected = selected / marker_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            coordinates = selected @ self.spatial_coordinates[0]
            if disable_spatial_coordinates:
                coordinates = torch.zeros_like(coordinates)
            coordinates = torch.where(presence[:, None], coordinates, torch.zeros_like(coordinates))
            return coordinates, selected, presence

        target, target_attention, target_present = locate(self.target_marker_id)
        reference, reference_attention, reference_present = locate(self.reference_marker_id)
        return (
            target,
            reference,
            target_attention,
            reference_attention,
            target_present,
            reference_present,
        )

    def forward(
        self,
        visual: torch.Tensor,
        question_ids: torch.Tensor,
        answer_input_ids: torch.Tensor | None = None,
        *,
        mask_image: bool = False,
        mask_text: bool = False,
        disable_visual_position: bool = False,
        disable_fusion: bool = False,
    ) -> TinyVLMOutput:
        visual_tokens, text_tokens, fusion_attention, memory, memory_padding = self.encode_multimodal(
            visual,
            question_ids,
            mask_image=mask_image,
            mask_text=mask_text,
            disable_visual_position=disable_visual_position,
            disable_fusion=disable_fusion,
        )
        (
            target_coordinates,
            reference_coordinates,
            target_attention,
            reference_attention,
            target_present,
            reference_present,
        ) = self.grounding_from_attention(
            question_ids,
            fusion_attention,
            disable_spatial_coordinates=disable_visual_position,
        )
        grounding_features = torch.cat(
            (
                target_coordinates,
                reference_coordinates,
                target_coordinates - reference_coordinates,
                target_present[:, None].to(text_tokens.dtype),
                reference_present[:, None].to(text_tokens.dtype),
            ),
            dim=1,
        )
        grounding_token = self.grounding_projection(grounding_features)
        classification_hidden = text_tokens[:, 0] + grounding_token
        classification_logits = self.classifier(classification_hidden)
        answer_logits = decoder_attention = None
        if answer_input_ids is not None:
            memory = torch.cat((memory, grounding_token[:, None, :]), dim=1)
            memory_padding = torch.cat(
                (
                    memory_padding,
                    torch.zeros(
                        memory_padding.shape[0], 1, dtype=torch.bool, device=memory_padding.device
                    ),
                ),
                dim=1,
            )
            answer_logits, decoder_attention = self.decode_teacher_forced(
                answer_input_ids, memory, memory_padding, grounding_token
            )
        return TinyVLMOutput(
            classification_logits,
            answer_logits,
            visual_tokens,
            text_tokens,
            fusion_attention,
            decoder_attention,
            target_coordinates,
            reference_coordinates,
            target_attention,
            reference_attention,
        )

    @torch.no_grad()
    def generate(
        self,
        visual: torch.Tensor,
        question_ids: torch.Tensor,
        *,
        max_new_tokens: int | None = None,
        **ablations: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        _, _, fusion_attention, memory, memory_padding = self.encode_multimodal(
            visual, question_ids, **ablations
        )
        (
            target_coordinates,
            reference_coordinates,
            _,
            _,
            target_present,
            reference_present,
        ) = self.grounding_from_attention(
            question_ids,
            fusion_attention,
            disable_spatial_coordinates=bool(
                ablations.get("disable_visual_position", False)
            ),
        )
        grounding_features = torch.cat(
            (
                target_coordinates,
                reference_coordinates,
                target_coordinates - reference_coordinates,
                target_present[:, None].to(memory.dtype),
                reference_present[:, None].to(memory.dtype),
            ),
            dim=1,
        )
        grounding_token = self.grounding_projection(grounding_features)
        memory = torch.cat((memory, grounding_token[:, None, :]), dim=1)
        memory_padding = torch.cat(
            (
                memory_padding,
                torch.zeros(
                    memory_padding.shape[0], 1, dtype=torch.bool, device=memory_padding.device
                ),
            ),
            dim=1,
        )
        limit = min(max_new_tokens or self.max_answer_length, self.max_answer_length)
        current = torch.full(
            (visual.shape[0], 1), self.bos_id, dtype=torch.long, device=visual.device
        )
        generated: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        finished = torch.zeros(visual.shape[0], dtype=torch.bool, device=visual.device)
        for _ in range(limit):
            logits, _ = self.decode_teacher_forced(
                current, memory, memory_padding, grounding_token
            )
            distribution = logits[:, -1].softmax(dim=-1)
            next_token = distribution.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, self.eos_id), next_token)
            generated.append(next_token)
            probabilities.append(distribution.gather(1, next_token[:, None]).squeeze(1))
            finished |= next_token.eq(self.eos_id)
            current = torch.cat((current, next_token[:, None]), dim=1)
            if bool(finished.all()):
                break
        return torch.stack(generated, dim=1), torch.stack(probabilities, dim=1)
