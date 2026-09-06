"""Immutable schemas for deterministic image rendering and instance annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray


NEUTRAL_BACKGROUND = (248, 248, 248)
CUE_BACKGROUNDS: dict[str, tuple[int, int, int]] = {
    "cue_0": (244, 244, 244),
    "cue_1": (228, 228, 228),
    "cue_2": (212, 212, 212),
    "cue_3": (196, 196, 196),
    "cue_4": (180, 180, 180),
}


@dataclass(frozen=True, slots=True)
class ImageRenderConfig:
    output_scale: int = 2
    safe_margin: int = 10
    enable_cue: bool = True
    neutral_background: tuple[int, int, int] = NEUTRAL_BACKGROUND
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported image render config version: {self.schema_version}")
        if self.output_scale <= 0:
            raise ValueError("output_scale must be positive")
        if self.safe_margin <= 0:
            raise ValueError("safe_margin must be positive")
        if len(self.neutral_background) != 3 or any(
            not 0 <= channel <= 255 for channel in self.neutral_background
        ):
            raise ValueError("neutral_background must be a valid RGB triplet")
        minimum_extent = 5 * self.output_scale
        if self.safe_margin < minimum_extent:
            raise ValueError(
                "safe_margin must contain the largest configured object extent"
            )


@dataclass(frozen=True, slots=True)
class ObjectImageAnnotation:
    object_id: str
    shape: str
    color: str
    size: str
    z_index: int
    center_normalized: tuple[float, float]
    affine_center_xy: tuple[int, int]
    center_xy: tuple[int, int]
    quantization_adjusted: bool
    bbox_xywh: tuple[int, int, int, int]
    visible_bbox_xywh: tuple[int, int, int, int] | None
    area_pixels: int
    visible_area_pixels: int
    visibility_fraction: float
    amodal_rle: tuple[int, ...]
    visible_rle: tuple[int, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported object annotation version")
        if not self.object_id:
            raise ValueError("Object annotation id must be non-empty")
        if self.z_index < 0 or self.area_pixels <= 0:
            raise ValueError("Object z-index and area are invalid")
        if not 0 <= self.visible_area_pixels <= self.area_pixels:
            raise ValueError("Visible object area must be within the amodal area")
        expected_fraction = self.visible_area_pixels / self.area_pixels
        if not np.isclose(self.visibility_fraction, expected_fraction):
            raise ValueError("Visibility fraction does not match object areas")
        if self.visible_area_pixels == 0 and self.visible_bbox_xywh is not None:
            raise ValueError("Fully hidden objects must not have a visible bounding box")
        if self.visible_area_pixels > 0 and self.visible_bbox_xywh is None:
            raise ValueError("Visible objects require a visible bounding box")


@dataclass(frozen=True, slots=True)
class ImageAnnotation:
    image_id: str
    scene_id: str
    frame_index: int
    width: int
    height: int
    background_rgb: tuple[int, int, int]
    cue_token: str | None
    cue_enabled: bool
    objects: tuple[ObjectImageAnnotation, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported image annotation version")
        if not self.image_id or not self.scene_id:
            raise ValueError("Image and scene ids must be non-empty")
        if self.frame_index < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Image frame or dimensions are invalid")
        if len({obj.object_id for obj in self.objects}) != len(self.objects):
            raise ValueError("Object annotation ids must be unique")
        if [obj.z_index for obj in self.objects] != list(range(len(self.objects))):
            raise ValueError("Visible object z-indices must be contiguous")


@dataclass(slots=True)
class RenderedFrame:
    image: Image.Image
    annotation: ImageAnnotation
    amodal_masks: dict[str, "NDArray[np.bool_]"]
    visible_masks: dict[str, "NDArray[np.bool_]"]

