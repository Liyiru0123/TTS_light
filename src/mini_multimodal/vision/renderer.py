"""Deterministic rasterizer from semantic scenes to RGB images and masks."""

from __future__ import annotations

from math import cos, isclose, pi, sin
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw

from mini_multimodal.world.schema import Color, Scene, SceneObject, Shape, Size

from .masks import encode_rle, mask_bbox
from .schema import (
    CUE_BACKGROUNDS,
    ImageAnnotation,
    ImageRenderConfig,
    ObjectImageAnnotation,
    RenderedFrame,
)


COLOR_RGB: dict[Color, tuple[int, int, int]] = {
    Color.RED: (220, 55, 55),
    Color.BLUE: (50, 105, 220),
    Color.GREEN: (50, 170, 80),
    Color.YELLOW: (240, 195, 50),
    Color.PURPLE: (145, 70, 190),
    Color.ORANGE: (235, 125, 40),
}

BASE_RADIUS: dict[Size, int] = {
    Size.SMALL: 2,
    Size.MEDIUM: 3,
    Size.LARGE: 4,
}


def _rounded(value: float) -> int:
    return int(value + 0.5)


def _axis_projection(
    values: list[float], *, low: int, high: int
) -> tuple[list[int], list[int]]:
    """Return affine-rounded and strictly order-preserving pixel coordinates."""

    if low >= high:
        raise ValueError("Projection interval must contain at least two pixels")
    ideals = [_rounded(low + value * (high - low)) for value in values]
    groups: list[list[int]] = []
    for index in sorted(range(len(values)), key=lambda item: (values[item], item)):
        if groups and isclose(values[index], values[groups[-1][0]], abs_tol=1e-9):
            groups[-1].append(index)
        else:
            groups.append([index])

    group_values = [_rounded(sum(ideals[i] for i in group) / len(group)) for group in groups]
    for index in range(1, len(group_values)):
        group_values[index] = max(group_values[index], group_values[index - 1] + 1)
    if group_values and group_values[-1] > high:
        group_values[-1] = high
        for index in range(len(group_values) - 2, -1, -1):
            group_values[index] = min(group_values[index], group_values[index + 1] - 1)
    if group_values and group_values[0] < low:
        group_values[0] = low
        for index in range(1, len(group_values)):
            group_values[index] = max(group_values[index], group_values[index - 1] + 1)
    if group_values and (group_values[0] < low or group_values[-1] > high):
        raise ValueError("Too many distinct coordinates for the projection interval")

    projected = [0] * len(values)
    for group, coordinate in zip(groups, group_values, strict=True):
        for index in group:
            projected[index] = coordinate
    return ideals, projected


def _object_mask(
    obj: SceneObject,
    center_xy: tuple[int, int],
    width: int,
    height: int,
    scale: int,
) -> np.ndarray:
    radius = BASE_RADIUS[obj.size] * scale
    cx, cy = center_xy
    mask_image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    if obj.shape is Shape.CIRCLE:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=1)
    elif obj.shape is Shape.SQUARE:
        draw.rectangle((cx - radius, cy - radius, cx + radius, cy + radius), fill=1)
    elif obj.shape is Shape.RECTANGLE:
        half_width = radius + scale
        half_height = max(2 * scale, radius - scale)
        draw.rectangle(
            (cx - half_width, cy - half_height, cx + half_width, cy + half_height),
            fill=1,
        )
    elif obj.shape is Shape.TRIANGLE:
        points = ((cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius))
        draw.polygon(points, fill=1)
    elif obj.shape is Shape.PENTAGON:
        points = tuple(
            (
                _rounded(cx + radius * cos(-pi / 2 + 2 * pi * index / 5)),
                _rounded(cy + radius * sin(-pi / 2 + 2 * pi * index / 5)),
            )
            for index in range(5)
        )
        draw.polygon(points, fill=1)
    else:  # pragma: no cover - guarded by the Shape enum.
        raise ValueError(f"Unsupported shape: {obj.shape}")
    return np.asarray(mask_image, dtype=np.bool_)


def render_scene(
    scene: Scene,
    *,
    frame_index: int = 0,
    cue_token: str | None = None,
    config: ImageRenderConfig | None = None,
    center_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> RenderedFrame:
    cfg = config or ImageRenderConfig()
    if not 0 <= frame_index < scene.frame_count:
        raise ValueError("frame_index is outside the scene")
    if cue_token is not None and cue_token not in CUE_BACKGROUNDS:
        raise ValueError(f"Unknown visual shortcut cue: {cue_token}")

    width = scene.canvas_width * cfg.output_scale
    height = scene.canvas_height * cfg.output_scale
    if width <= 2 * cfg.safe_margin or height <= 2 * cfg.safe_margin:
        raise ValueError("Rendered canvas is too small for the configured safe margin")
    visible_objects = [obj for obj in scene.objects if obj.is_visible_at(frame_index)]
    positions = [obj.position_at(frame_index) for obj in visible_objects]
    ideal_x = [_rounded(cfg.safe_margin + position.x * (width - 1 - 2 * cfg.safe_margin)) for position in positions]
    ideal_y = [_rounded(cfg.safe_margin + position.y * (height - 1 - 2 * cfg.safe_margin)) for position in positions]
    if center_overrides is None:
        _, projected_x = _axis_projection([position.x for position in positions], low=cfg.safe_margin, high=width - 1 - cfg.safe_margin)
        _, projected_y = _axis_projection([position.y for position in positions], low=cfg.safe_margin, high=height - 1 - cfg.safe_margin)
    else:
        missing = {obj.object_id for obj in visible_objects} - set(center_overrides)
        if missing:
            raise ValueError(f"Center overrides are missing visible objects: {sorted(missing)}")
        projected_x = [center_overrides[obj.object_id][0] for obj in visible_objects]
        projected_y = [center_overrides[obj.object_id][1] for obj in visible_objects]
        if any(not cfg.safe_margin <= value <= width - 1 - cfg.safe_margin for value in projected_x) or any(not cfg.safe_margin <= value <= height - 1 - cfg.safe_margin for value in projected_y):
            raise ValueError("Center override lies outside the safe projection interval")

    cue_enabled = bool(cue_token is not None and cfg.enable_cue)
    background = CUE_BACKGROUNDS[cue_token] if cue_enabled else cfg.neutral_background
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:, :] = background
    amodal_masks: dict[str, np.ndarray] = {}
    centers: dict[str, tuple[int, int]] = {}
    affine_centers: dict[str, tuple[int, int]] = {}

    for index, obj in enumerate(visible_objects):
        center = (projected_x[index], projected_y[index])
        mask = _object_mask(obj, center, width, height, cfg.output_scale)
        amodal_masks[obj.object_id] = mask
        centers[obj.object_id] = center
        affine_centers[obj.object_id] = (ideal_x[index], ideal_y[index])
        pixels[mask] = COLOR_RGB[obj.color]

    visible_masks: dict[str, np.ndarray] = {}
    covered = np.zeros((height, width), dtype=np.bool_)
    for obj in reversed(visible_objects):
        mask = amodal_masks[obj.object_id]
        visible_masks[obj.object_id] = mask & ~covered
        covered |= mask

    annotations: list[ObjectImageAnnotation] = []
    for z_index, obj in enumerate(visible_objects):
        amodal = amodal_masks[obj.object_id]
        visible = visible_masks[obj.object_id]
        area = int(amodal.sum())
        visible_area = int(visible.sum())
        position = obj.position_at(frame_index)
        center = centers[obj.object_id]
        affine = affine_centers[obj.object_id]
        bbox = mask_bbox(amodal)
        assert bbox is not None
        annotations.append(
            ObjectImageAnnotation(
                object_id=obj.object_id,
                shape=obj.shape.value,
                color=obj.color.value,
                size=obj.size.value,
                z_index=z_index,
                center_normalized=(position.x, position.y),
                affine_center_xy=affine,
                center_xy=center,
                quantization_adjusted=center != affine,
                bbox_xywh=bbox,
                visible_bbox_xywh=mask_bbox(visible),
                area_pixels=area,
                visible_area_pixels=visible_area,
                visibility_fraction=visible_area / area,
                amodal_rle=encode_rle(amodal),
                visible_rle=encode_rle(visible),
            )
        )

    annotation = ImageAnnotation(
        image_id=f"{scene.scene_id}:image:f{frame_index}",
        scene_id=scene.scene_id,
        frame_index=frame_index,
        width=width,
        height=height,
        background_rgb=background,
        cue_token=cue_token,
        cue_enabled=cue_enabled,
        objects=tuple(annotations),
    )
    return RenderedFrame(
        image=Image.fromarray(pixels, mode="RGB"),
        annotation=annotation,
        amodal_masks=amodal_masks,
        visible_masks=visible_masks,
    )
