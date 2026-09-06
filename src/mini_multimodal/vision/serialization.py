"""JSON-compatible serialization for image-rendering schemas."""

from __future__ import annotations

from typing import Any, Mapping

from .schema import ImageAnnotation, ImageRenderConfig, ObjectImageAnnotation


def render_config_to_dict(config: ImageRenderConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "output_scale": config.output_scale,
        "safe_margin": config.safe_margin,
        "enable_cue": config.enable_cue,
        "neutral_background": list(config.neutral_background),
    }


def render_config_from_dict(data: Mapping[str, Any]) -> ImageRenderConfig:
    return ImageRenderConfig(
        schema_version=int(data["schema_version"]),
        output_scale=int(data["output_scale"]),
        safe_margin=int(data["safe_margin"]),
        enable_cue=bool(data["enable_cue"]),
        neutral_background=tuple(int(value) for value in data["neutral_background"]),
    )


def object_annotation_to_dict(annotation: ObjectImageAnnotation) -> dict[str, Any]:
    return {
        "schema_version": annotation.schema_version,
        "object_id": annotation.object_id,
        "shape": annotation.shape,
        "color": annotation.color,
        "size": annotation.size,
        "z_index": annotation.z_index,
        "center_normalized": list(annotation.center_normalized),
        "affine_center_xy": list(annotation.affine_center_xy),
        "center_xy": list(annotation.center_xy),
        "quantization_adjusted": annotation.quantization_adjusted,
        "bbox_xywh": list(annotation.bbox_xywh),
        "visible_bbox_xywh": (
            list(annotation.visible_bbox_xywh)
            if annotation.visible_bbox_xywh is not None
            else None
        ),
        "area_pixels": annotation.area_pixels,
        "visible_area_pixels": annotation.visible_area_pixels,
        "visibility_fraction": annotation.visibility_fraction,
        "amodal_rle": list(annotation.amodal_rle),
        "visible_rle": list(annotation.visible_rle),
    }


def object_annotation_from_dict(data: Mapping[str, Any]) -> ObjectImageAnnotation:
    visible_bbox = data["visible_bbox_xywh"]
    return ObjectImageAnnotation(
        schema_version=int(data["schema_version"]),
        object_id=str(data["object_id"]),
        shape=str(data["shape"]),
        color=str(data["color"]),
        size=str(data["size"]),
        z_index=int(data["z_index"]),
        center_normalized=tuple(float(value) for value in data["center_normalized"]),
        affine_center_xy=tuple(int(value) for value in data["affine_center_xy"]),
        center_xy=tuple(int(value) for value in data["center_xy"]),
        quantization_adjusted=bool(data["quantization_adjusted"]),
        bbox_xywh=tuple(int(value) for value in data["bbox_xywh"]),
        visible_bbox_xywh=(
            tuple(int(value) for value in visible_bbox)
            if visible_bbox is not None
            else None
        ),
        area_pixels=int(data["area_pixels"]),
        visible_area_pixels=int(data["visible_area_pixels"]),
        visibility_fraction=float(data["visibility_fraction"]),
        amodal_rle=tuple(int(value) for value in data["amodal_rle"]),
        visible_rle=tuple(int(value) for value in data["visible_rle"]),
    )


def image_annotation_to_dict(annotation: ImageAnnotation) -> dict[str, Any]:
    return {
        "schema_version": annotation.schema_version,
        "image_id": annotation.image_id,
        "scene_id": annotation.scene_id,
        "frame_index": annotation.frame_index,
        "width": annotation.width,
        "height": annotation.height,
        "background_rgb": list(annotation.background_rgb),
        "cue_token": annotation.cue_token,
        "cue_enabled": annotation.cue_enabled,
        "objects": [object_annotation_to_dict(obj) for obj in annotation.objects],
    }


def image_annotation_from_dict(data: Mapping[str, Any]) -> ImageAnnotation:
    return ImageAnnotation(
        schema_version=int(data["schema_version"]),
        image_id=str(data["image_id"]),
        scene_id=str(data["scene_id"]),
        frame_index=int(data["frame_index"]),
        width=int(data["width"]),
        height=int(data["height"]),
        background_rgb=tuple(int(value) for value in data["background_rgb"]),
        cue_token=str(data["cue_token"]) if data["cue_token"] is not None else None,
        cue_enabled=bool(data["cue_enabled"]),
        objects=tuple(object_annotation_from_dict(item) for item in data["objects"]),
    )

