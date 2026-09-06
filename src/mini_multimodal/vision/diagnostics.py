"""Semantic, geometric, occlusion, and shortcut diagnostics for rendered images."""

from __future__ import annotations

from collections import Counter

import numpy as np

from mini_multimodal.data.schema import DatasetBundle
from mini_multimodal.tasks.schema import QuestionKind
from mini_multimodal.world.schema import Color, Shape, Size

from .dataset import iter_image_specs
from .masks import decode_rle, mask_bbox
from .renderer import render_scene
from .schema import CUE_BACKGROUNDS, ImageRenderConfig


def analyze_image_bundle(
    bundle: DatasetBundle, config: ImageRenderConfig | None = None
) -> dict[str, object]:
    cfg = config or ImageRenderConfig()
    specs = iter_image_specs(bundle)
    annotations_by_scene = {}
    shape_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    pool_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    occluded_object_count = 0
    fully_occluded_object_count = 0
    out_of_bounds_count = 0
    rle_mismatch_count = 0
    bbox_mismatch_count = 0
    adjusted_object_count = 0
    max_adjustment = 0
    background_mismatch_count = 0
    cue_counts: Counter[str] = Counter()

    for spec in specs:
        rendered = render_scene(
            spec.scene, frame_index=0, cue_token=spec.cue_token, config=cfg
        )
        annotation = rendered.annotation
        annotations_by_scene[spec.scene.scene_id] = annotation
        split_counts[spec.split.value] += 1
        pool_counts[spec.pool] += 1
        variant_counts[spec.variant.value] += 1
        expected_background = (
            CUE_BACKGROUNDS[spec.cue_token]
            if spec.cue_token is not None and cfg.enable_cue
            else cfg.neutral_background
        )
        background_mismatch_count += annotation.background_rgb != expected_background
        if spec.cue_token is not None:
            cue_counts[spec.cue_token] += 1

        for obj in annotation.objects:
            shape_counts[obj.shape] += 1
            color_counts[obj.color] += 1
            size_counts[obj.size] += 1
            occluded_object_count += obj.visible_area_pixels < obj.area_pixels
            fully_occluded_object_count += obj.visible_area_pixels == 0
            dx = abs(obj.center_xy[0] - obj.affine_center_xy[0])
            dy = abs(obj.center_xy[1] - obj.affine_center_xy[1])
            max_adjustment = max(max_adjustment, dx, dy)
            adjusted_object_count += obj.quantization_adjusted
            x, y, width, height = obj.bbox_xywh
            out_of_bounds_count += not (
                0 <= x < annotation.width
                and 0 <= y < annotation.height
                and width > 0
                and height > 0
                and x + width <= annotation.width
                and y + height <= annotation.height
            )
            amodal = decode_rle(obj.amodal_rle, annotation.height, annotation.width)
            visible = decode_rle(obj.visible_rle, annotation.height, annotation.width)
            rle_mismatch_count += not np.array_equal(
                amodal, rendered.amodal_masks[obj.object_id]
            )
            rle_mismatch_count += not np.array_equal(
                visible, rendered.visible_masks[obj.object_id]
            )
            bbox_mismatch_count += mask_bbox(amodal) != obj.bbox_xywh
            bbox_mismatch_count += mask_bbox(visible) != obj.visible_bbox_xywh
            bbox_mismatch_count += int(amodal.sum()) != obj.area_pixels
            bbox_mismatch_count += int(visible.sum()) != obj.visible_area_pixels

    relation_record_count = 0
    relation_order_violation_count = 0
    relation_kinds = {
        QuestionKind.HORIZONTAL_RELATION,
        QuestionKind.VERTICAL_RELATION,
    }
    for record in bundle.core_records:
        question = record.task.question
        if question.kind not in relation_kinds or question.frame != 0:
            continue
        relation_record_count += 1
        objects = {
            obj.object_id: obj
            for obj in annotations_by_scene[record.scene_id].objects
        }
        target = objects[question.target_object_id]
        reference = objects[question.reference_object_id]
        axis = 0 if question.kind is QuestionKind.HORIZONTAL_RELATION else 1
        pixel_delta = target.center_xy[axis] - reference.center_xy[axis]
        answer = str(record.task.answer.value)
        expected_sign = 0
        if answer in {"left_of", "above"}:
            expected_sign = -1
        elif answer in {"right_of", "below"}:
            expected_sign = 1
        actual_sign = 0 if pixel_delta == 0 else (-1 if pixel_delta < 0 else 1)
        relation_order_violation_count += actual_sign != expected_sign

    statistics = {
        "image_count": len(specs),
        "object_count": sum(shape_counts.values()),
        "image_width": bundle.core_scenes[0].canvas_width * cfg.output_scale,
        "image_height": bundle.core_scenes[0].canvas_height * cfg.output_scale,
        "pool_counts": dict(sorted(pool_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "color_counts": dict(sorted(color_counts.items())),
        "size_counts": dict(sorted(size_counts.items())),
        "occluded_object_count": occluded_object_count,
    }
    consistency = {
        "relation_record_count": relation_record_count,
        "relation_order_violation_count": relation_order_violation_count,
        "adjusted_object_count": adjusted_object_count,
        "max_quantization_adjustment_pixels": max_adjustment,
        "out_of_bounds_count": out_of_bounds_count,
        "fully_occluded_object_count": fully_occluded_object_count,
        "rle_mismatch_count": rle_mismatch_count,
        "bbox_or_area_mismatch_count": bbox_mismatch_count,
    }
    shortcut = {
        "cue_backgrounds": {key: list(value) for key, value in CUE_BACKGROUNDS.items()},
        "neutral_background": list(cfg.neutral_background),
        "cue_image_counts": dict(sorted(cue_counts.items())),
        "background_mismatch_count": background_mismatch_count,
        "shortcut_image_count": sum(cue_counts.values()),
    }
    return {"statistics": statistics, "consistency": consistency, "shortcut": shortcut}


def validate_image_bundle(
    bundle: DatasetBundle, config: ImageRenderConfig | None = None
) -> dict[str, object]:
    report = analyze_image_bundle(bundle, config)
    stats = report["statistics"]
    consistency = report["consistency"]
    shortcut = report["shortcut"]
    if stats["image_count"] != len(bundle.all_scenes):
        raise ValueError("Image count does not equal dataset scene count")
    if set(stats["shape_counts"]) != {shape.value for shape in Shape}:
        raise ValueError("Rendered images do not cover every shape")
    if set(stats["color_counts"]) != {color.value for color in Color}:
        raise ValueError("Rendered images do not cover every color")
    if set(stats["size_counts"]) != {size.value for size in Size}:
        raise ValueError("Rendered images do not cover every size")
    blocking = (
        "relation_order_violation_count",
        "out_of_bounds_count",
        "fully_occluded_object_count",
        "rle_mismatch_count",
        "bbox_or_area_mismatch_count",
    )
    for key in blocking:
        if consistency[key] != 0:
            raise ValueError(f"Blocking image consistency diagnostic failed: {key}")
    if consistency["max_quantization_adjustment_pixels"] > 1:
        raise ValueError("Order-preserving quantization moved an object by more than one pixel")
    if shortcut["background_mismatch_count"] != 0:
        raise ValueError("Visual shortcut background mapping is inconsistent")
    return report
