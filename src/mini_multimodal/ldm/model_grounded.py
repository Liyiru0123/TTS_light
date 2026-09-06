"""Text-derived object grounding for the second P2-W04 revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .model_v2 import TinyImageDenoiserV2, coordinate_grid


GROUNDING_TOKEN_GROUPS: dict[str, tuple[str, ...]] = {
    "color": ("红色", "蓝色", "绿色", "黄色", "紫色", "橙色"),
    "shape": ("圆形", "正方形", "三角形", "矩形", "五边形"),
    "size": ("小", "中等大小", "大"),
    "region": (
        "左上区域", "上方区域", "右上区域", "左侧区域", "中央区域",
        "右侧区域", "左下区域", "下方区域", "右下区域",
    ),
}


def resolve_grounding_token_ids(token_to_id: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    missing = sorted(token for values in GROUNDING_TOKEN_GROUPS.values() for token in values if token not in token_to_id)
    if missing:
        raise ValueError(f"vocabulary lacks grounding tokens: {missing}")
    return {
        group: tuple(int(token_to_id[token]) for token in values)
        for group, values in GROUNDING_TOKEN_GROUPS.items()
    }


@dataclass(slots=True)
class ObjectGroundingSlots:
    token_positions: torch.Tensor  # [B,O,4] in color/shape/size/region order
    mask: torch.Tensor  # [B,O]


class TinyImageDenoiserGrounded(TinyImageDenoiserV2):
    """First-round denoiser plus gated object-semantic regional features."""

    model_family = "TinyImageLDM-grounded-v2"

    def __init__(
        self,
        *,
        grounding_token_ids: Mapping[str, Sequence[int]],
        max_objects: int = 3,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.max_objects = int(max_objects)
        if self.max_objects < 1:
            raise ValueError("max_objects must be positive")
        for group in ("color", "shape", "size", "region"):
            values = torch.tensor(list(grounding_token_ids[group]), dtype=torch.long)
            self.register_buffer(f"grounding_{group}_ids", values, persistent=True)
        text_dim = self.text_embedding.embedding_dim
        base_channels = self.input.out_channels
        hidden_channels = self.down.out_channels
        self.object_semantic_encoder = nn.Sequential(
            nn.Linear(text_dim * 4, text_dim * 2),
            nn.SiLU(),
            nn.Linear(text_dim * 2, text_dim),
        )
        self.object_projection_16 = nn.Linear(text_dim, base_channels)
        self.object_projection_8 = nn.Linear(text_dim, hidden_channels)
        self.object_gate_16 = nn.Parameter(torch.tensor(0.5))
        self.object_gate_8 = nn.Parameter(torch.tensor(0.5))

    def extract_object_slots(self, text_ids: torch.Tensor) -> ObjectGroundingSlots:
        if text_ids.ndim != 2:
            raise ValueError("text_ids must have shape [B,L]")
        batch, length = text_ids.shape
        positions = torch.full((batch, self.max_objects, 4), -1, dtype=torch.long, device=text_ids.device)
        sequence_positions = torch.arange(length, device=text_ids.device)[None, :].expand(batch, -1)
        region_hits = torch.isin(text_ids, self.grounding_region_ids)
        region_rank = region_hits.cumsum(dim=1)
        previous_region = torch.full((batch,), -1, dtype=torch.long, device=text_ids.device)
        groups = (self.grounding_color_ids, self.grounding_shape_ids, self.grounding_size_ids)
        for object_index in range(self.max_objects):
            current_hits = region_hits & (region_rank == object_index + 1)
            has_region = current_hits.any(dim=1)
            region_position = torch.where(current_hits, sequence_positions, -1).amax(dim=1)
            selected = []
            for candidates in groups:
                category_hits = torch.isin(text_ids, candidates)
                within_object = (
                    category_hits
                    & (sequence_positions > previous_region[:, None])
                    & (sequence_positions < region_position[:, None])
                )
                selected.append(torch.where(within_object, sequence_positions, -1).amax(dim=1))
            positions[:, object_index, 0] = selected[0]
            positions[:, object_index, 1] = selected[1]
            positions[:, object_index, 2] = selected[2]
            positions[:, object_index, 3] = region_position
            previous_region = torch.where(has_region, region_position, previous_region)
        mask = (positions >= 0).all(dim=-1)
        return ObjectGroundingSlots(positions, mask)

    def object_grounding_feature(
        self,
        text_ids: torch.Tensor,
        encoded_text: torch.Tensor,
        height: int,
        width: int,
        *,
        scale: str,
    ) -> torch.Tensor:
        slots = self.extract_object_slots(text_ids)
        batch, objects, attributes = slots.token_positions.shape
        safe_positions = slots.token_positions.clamp_min(0)
        batch_index = torch.arange(batch, device=text_ids.device)[:, None, None].expand(batch, objects, attributes)
        gathered = encoded_text[batch_index, safe_positions]
        semantics = self.object_semantic_encoder(gathered.reshape(batch, objects, -1))
        semantics = semantics * slots.mask[:, :, None].to(semantics.dtype)

        region_positions = safe_positions[:, :, 3]
        region_ids = text_ids.gather(1, region_positions)
        region_xy = self.spatial_token_coordinates[region_ids]
        grid = coordinate_grid(height, width, device=text_ids.device, dtype=semantics.dtype)
        query_xy = grid.flatten(2).transpose(1, 2)
        distance = (query_xy[:, :, None, :] - region_xy[:, None, :, :]).square().sum(dim=-1)
        fields = torch.exp(-distance / (2.0 * 0.32**2))
        fields = fields * slots.mask[:, None, :].to(fields.dtype)

        if scale == "16x16":
            projected = self.object_projection_16(semantics)
            gate = torch.tanh(self.object_gate_16)
        elif scale == "8x8":
            projected = self.object_projection_8(semantics)
            gate = torch.tanh(self.object_gate_8)
        else:
            raise ValueError(f"unsupported grounding scale: {scale}")
        grounded = torch.einsum("bqo,boc->bcq", fields, projected)
        return gate * grounded.reshape(batch, projected.shape[-1], height, width)
