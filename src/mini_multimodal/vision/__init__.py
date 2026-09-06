"""Deterministic image rendering and object-level annotations."""

from .diagnostics import analyze_image_bundle, validate_image_bundle
from .masks import decode_rle, encode_rle, mask_bbox
from .renderer import BASE_RADIUS, COLOR_RGB, render_scene
from .schema import (
    CUE_BACKGROUNDS,
    NEUTRAL_BACKGROUND,
    ImageAnnotation,
    ImageRenderConfig,
    ObjectImageAnnotation,
    RenderedFrame,
)

__all__ = [
    "BASE_RADIUS",
    "COLOR_RGB",
    "CUE_BACKGROUNDS",
    "NEUTRAL_BACKGROUND",
    "ImageAnnotation",
    "ImageRenderConfig",
    "ObjectImageAnnotation",
    "RenderedFrame",
    "analyze_image_bundle",
    "decode_rle",
    "encode_rle",
    "mask_bbox",
    "render_scene",
    "validate_image_bundle",
]
