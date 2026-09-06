"""Calibrated object, contour, size, and position evaluation for P2-W04 v2."""

from __future__ import annotations

from itertools import permutations
from math import exp, hypot, log, sqrt
from typing import Any, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mini_multimodal.vision.renderer import COLOR_RGB, render_scene
from mini_multimodal.world.schema import Color, Position, Scene, SceneObject, Shape, Size


SHAPES = tuple(shape.value for shape in Shape)
SIZES = tuple(size.value for size in Size)
COLORS = tuple(color.value for color in Color)
PALETTE = np.asarray([COLOR_RGB[color] for color in Color], dtype=np.float32) / 255.0


def load_shape_prototypes(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(Path(path), allow_pickle=False)
    result = {
        "keys": payload["keys"],
        "labels": payload["labels"],
        "masks": payload["masks"].astype(np.bool_),
    }
    payload.close()
    if result["masks"].ndim != 3 or len(result["labels"]) != len(result["masks"]):
        raise ValueError("invalid shape prototype bank")
    return result


def _as_rgb_array(image: Image.Image | torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        value = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(image, torch.Tensor):
        value = image.detach().cpu().float().numpy()
        if value.ndim == 4:
            if value.shape[0] != 1:
                raise ValueError("batched evaluator input must contain one image")
            value = value[0]
        if value.shape == (3, 64, 64):
            value = value.transpose(1, 2, 0)
    else:
        value = np.asarray(image, dtype=np.float32)
        if value.dtype == np.uint8 or float(value.max(initial=0.0)) > 1.5:
            value = value / 255.0
    if value.shape != (64, 64, 3):
        raise ValueError(f"evaluator expects a 64x64 RGB image, got {value.shape}")
    return np.clip(value, 0.0, 1.0)


def _components(mask: np.ndarray, *, minimum_area: int) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    result: list[np.ndarray] = []
    for y, x in zip(*np.nonzero(mask), strict=True):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(pixels) >= minimum_area:
            component = np.zeros_like(mask, dtype=np.bool_)
            yy, xx = zip(*pixels, strict=True)
            component[np.asarray(yy), np.asarray(xx)] = True
            result.append(component)
    return result


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return 1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=np.bool_)
    y_src0, y_src1 = max(0, -dy), min(mask.shape[0], mask.shape[0] - dy)
    x_src0, x_src1 = max(0, -dx), min(mask.shape[1], mask.shape[1] - dx)
    y_dst0, y_dst1 = max(0, dy), min(mask.shape[0], mask.shape[0] + dy)
    x_dst0, x_dst1 = max(0, dx), min(mask.shape[1], mask.shape[1] + dx)
    result[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
    return result


def _erode(mask: np.ndarray) -> np.ndarray:
    result = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result &= _shift(mask, dy, dx)
    return result


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=np.bool_)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result |= _shift(mask, dy, dx)
    return result


def boundary_f_score(left: np.ndarray, right: np.ndarray, *, tolerance: int = 2) -> float:
    left_boundary = left & ~_erode(left)
    right_boundary = right & ~_erode(right)
    if not left_boundary.any() and not right_boundary.any():
        return 1.0
    if not left_boundary.any() or not right_boundary.any():
        return 0.0
    precision = float((left_boundary & _dilate(right_boundary, tolerance)).sum() / left_boundary.sum())
    recall = float((right_boundary & _dilate(left_boundary, tolerance)).sum() / right_boundary.sum())
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def normalize_mask(mask: np.ndarray, *, canvas: int = 64, inner: int = 52) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((canvas, canvas), dtype=np.bool_)
    crop = mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    height, width = crop.shape
    scale = inner / max(height, width)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    resized = Image.fromarray(crop.astype(np.uint8) * 255).resize(
        (target_width, target_height), resample=Image.Resampling.NEAREST
    )
    result = np.zeros((canvas, canvas), dtype=np.bool_)
    top = (canvas - target_height) // 2
    left = (canvas - target_width) // 2
    result[top : top + target_height, left : left + target_width] = np.asarray(resized) > 0
    return result


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (size, size), resample=Image.Resampling.NEAREST
        )
    ) > 0


def _template_mask(shape: str, size: str) -> np.ndarray:
    scene = Scene(
        scene_id=f"template_{shape}_{size}",
        seed=0,
        canvas_width=32,
        canvas_height=32,
        frame_count=1,
        fps=8,
        objects=(SceneObject(
            object_id="obj_0",
            shape=Shape(shape),
            color=Color.RED,
            size=Size(size),
            initial_position=Position(0.5, 0.5),
            visible_start=0,
            visible_end=0,
        ),),
    )
    return render_scene(scene, frame_index=0).amodal_masks["obj_0"]


TEMPLATE_MASKS = {
    (shape, size): _template_mask(shape, size) for shape in SHAPES for size in SIZES
}
NORMALIZED_TEMPLATES = {
    key: normalize_mask(mask) for key, mask in TEMPLATE_MASKS.items()
}
TEMPLATE_AREAS = {key: int(mask.sum()) for key, mask in TEMPLATE_MASKS.items()}


def _region(center_xy: Sequence[float]) -> str:
    x, y = float(center_xy[0]), float(center_xy[1])
    horizontal = "left" if x < 1 / 3 else "right" if x >= 2 / 3 else "center"
    vertical = "upper" if y < 1 / 3 else "lower" if y >= 2 / 3 else "center"
    return "center" if horizontal == vertical == "center" else f"{vertical}_{horizontal}"


def _axis_parts(region: str) -> tuple[str, str]:
    if region == "center":
        return "center", "center"
    vertical, horizontal = region.split("_", 1)
    return horizontal, vertical


def probe_image_v2(
    image: Image.Image | torch.Tensor | np.ndarray,
    *,
    minimum_area: int = 6,
    shape_reject_score: float = 0.45,
    shape_reject_margin: float = 0.015,
    shape_templates: Mapping[tuple[str, str], np.ndarray] | None = None,
    shape_prototypes: Mapping[str, Any] | None = None,
    maximum_match_candidates: int = 6,
) -> dict[str, Any]:
    array = _as_rgb_array(image)
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]), axis=0)
    background = np.median(border, axis=0)
    palette_distance = ((array[:, :, None, :] - PALETTE[None, None, :, :]) ** 2).sum(axis=-1)
    background_distance = ((array - background[None, None, :]) ** 2).sum(axis=-1)
    color_labels = palette_distance.argmin(axis=-1)
    foreground = palette_distance.min(axis=-1) < background_distance
    objects: list[dict[str, Any]] = []
    for color_index, color_name in enumerate(COLORS):
        color_mask = foreground & (color_labels == color_index)
        for component in _components(color_mask, minimum_area=minimum_area):
            ys, xs = np.nonzero(component)
            normalized = normalize_mask(component)
            if shape_prototypes is not None:
                prototype_masks = np.asarray(shape_prototypes["masks"], dtype=np.bool_)
                prototype_labels = np.asarray(shape_prototypes["labels"])
                query = resize_mask(normalized, int(prototype_masks.shape[-1]))
                distances = np.logical_xor(prototype_masks, query[None]).mean(axis=(1, 2))
                shape_scores = {
                    shape: float(1.0 - distances[prototype_labels == shape].min())
                    for shape in SHAPES
                }
            else:
                template_bank = NORMALIZED_TEMPLATES if shape_templates is None else {
                    **NORMALIZED_TEMPLATES,
                    **shape_templates,
                }
                shape_scores = {
                    shape: max(
                        _mask_iou(normalized, template)
                        for (template_shape, _), template in template_bank.items()
                        if template_shape == shape
                    )
                    for shape in SHAPES
                }
            ranked_shapes = sorted(shape_scores, key=shape_scores.get, reverse=True)
            predicted_shape = ranked_shapes[0]
            best_score = float(shape_scores[predicted_shape])
            margin = best_score - float(shape_scores[ranked_shapes[1]])
            rejected = best_score < shape_reject_score or margin < shape_reject_margin
            area = int(component.sum())
            predicted_size = min(
                SIZES,
                key=lambda size: abs(log((area + 1.0) / (TEMPLATE_AREAS[(predicted_shape, size)] + 1.0))),
            )
            center = [
                float((xs.min() + xs.max()) / (2.0 * 63.0)),
                float((ys.min() + ys.max()) / (2.0 * 63.0)),
            ]
            objects.append({
                "color": color_name,
                "color_confidence": float(
                    1.0 - np.sqrt(palette_distance[component, color_index].mean() / 3.0)
                ),
                "shape": predicted_shape,
                "shape_scores": shape_scores,
                "shape_confidence": best_score,
                "shape_margin": margin,
                "shape_rejected": rejected,
                "size": predicted_size,
                "area": area,
                "center_xy": center,
                "position_region": _region(center),
                "mask": component,
                "normalized_mask": normalized,
            })
    objects.sort(key=lambda item: (-item["area"], item["color"], item["center_xy"]))
    detected_object_count = len(objects)
    objects = objects[:maximum_match_candidates]
    return {
        "foreground_present": bool(objects),
        "predicted_object_count": detected_object_count,
        "evaluated_candidate_count": len(objects),
        "discarded_small_candidate_count": max(0, detected_object_count - len(objects)),
        "foreground_pixel_count": int(foreground.sum()),
        "background_rgb": background.tolist(),
        "objects": objects,
    }


def _assignment(
    targets: Sequence[Mapping[str, Any]], predicted: Sequence[Mapping[str, Any]]
) -> tuple[list[int | None], list[int]]:
    target_count, predicted_count = len(targets), len(predicted)
    width = max(target_count, predicted_count)
    if width == 0:
        return [], []
    cost = np.zeros((width, width), dtype=np.float64)
    for row in range(width):
        for column in range(width):
            if row >= target_count and column >= predicted_count:
                value = 0.0
            elif row >= target_count:
                value = 2.0
            elif column >= predicted_count:
                value = 2.5
            else:
                target = targets[row]
                item = predicted[column]
                target_center = target.get("rendered_center_xy", target["position"])
                center_cost = hypot(
                    float(target_center[0]) - float(item["center_xy"][0]),
                    float(target_center[1]) - float(item["center_xy"][1]),
                )
                color_cost = 0.0 if target["color"] == item["color"] else 1.0
                target_area = float(target.get("area_pixels", item["area"]))
                area_cost = abs(log((item["area"] + 1.0) / (target_area + 1.0)))
                value = color_cost + 1.5 * center_cost + 0.25 * area_cost
            cost[row, column] = value
    best_columns = min(permutations(range(width)), key=lambda cols: sum(cost[row, col] for row, col in enumerate(cols)))
    matches = [
        (best_columns[row] if best_columns[row] < predicted_count else None)
        for row in range(target_count)
    ]
    matched = {value for value in matches if value is not None}
    extras = [index for index in range(predicted_count) if index not in matched]
    return matches, extras


def _serializable_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in probe.items()
        if key != "objects"
    } | {
        "objects": [
            {key: value for key, value in item.items() if key not in {"mask", "normalized_mask"}}
            for item in probe["objects"]
        ]
    }


def evaluate_image_v2(
    image: Image.Image | torch.Tensor | np.ndarray,
    target: Mapping[str, Any],
    ideal_masks: np.ndarray,
    *,
    fully_correct_shape_iou: float,
    minimum_area: int = 6,
    shape_reject_score: float = 0.45,
    shape_reject_margin: float = 0.015,
    shape_templates: Mapping[tuple[str, str], np.ndarray] | None = None,
    shape_prototypes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    probe = probe_image_v2(
        image,
        minimum_area=minimum_area,
        shape_reject_score=shape_reject_score,
        shape_reject_margin=shape_reject_margin,
        shape_templates=shape_templates,
        shape_prototypes=shape_prototypes,
    )
    targets = target["objects"]
    matches, extras = _assignment(targets, probe["objects"])
    pairs: list[dict[str, Any]] = []
    for index, target_object in enumerate(targets):
        predicted_index = matches[index]
        target_mask = ideal_masks[index].astype(np.bool_)
        target_normalized = normalize_mask(target_mask)
        if predicted_index is None:
            pairs.append({
                "target_index": index,
                "prediction_index": None,
                "target": target_object,
                "prediction": None,
                "color_correct": False,
                "shape_correct": False,
                "size_correct": False,
                "position_correct": False,
                "horizontal_correct": False,
                "vertical_correct": False,
                "normalized_template_iou": 0.0,
                "boundary_score": 0.0,
                "target_position_mask_iou": 0.0,
                "center_distance": sqrt(2.0),
                "object_fully_correct": False,
            })
            continue
        item = probe["objects"][predicted_index]
        normalized_iou = _mask_iou(item["normalized_mask"], target_normalized)
        boundary = boundary_f_score(item["normalized_mask"], target_normalized)
        raw_iou = _mask_iou(item["mask"], target_mask)
        target_center = target_object.get("rendered_center_xy", target_object["position"])
        center_distance = hypot(
            float(item["center_xy"][0]) - float(target_center[0]),
            float(item["center_xy"][1]) - float(target_center[1]),
        )
        target_horizontal, target_vertical = _axis_parts(target_object["position_region"])
        predicted_horizontal, predicted_vertical = _axis_parts(item["position_region"])
        color_correct = item["color"] == target_object["color"]
        shape_correct = item["shape"] == target_object["shape"] and not item["shape_rejected"]
        size_correct = item["size"] == target_object["size"]
        position_correct = item["position_region"] == target_object["position_region"]
        object_fully_correct = bool(
            color_correct
            and shape_correct
            and normalized_iou >= fully_correct_shape_iou
            and size_correct
            and position_correct
        )
        pairs.append({
            "target_index": index,
            "prediction_index": predicted_index,
            "target": target_object,
            "prediction": {
                key: value for key, value in item.items() if key not in {"mask", "normalized_mask"}
            },
            "color_correct": color_correct,
            "shape_correct": shape_correct,
            "size_correct": size_correct,
            "position_correct": position_correct,
            "horizontal_correct": predicted_horizontal == target_horizontal,
            "vertical_correct": predicted_vertical == target_vertical,
            "normalized_template_iou": normalized_iou,
            "boundary_score": boundary,
            "target_position_mask_iou": raw_iou,
            "center_distance": center_distance,
            "object_fully_correct": object_fully_correct,
        })
    count_correct = len(targets) == int(probe["predicted_object_count"])
    return {
        "record_id": target["record_id"],
        "subset": target["subset"],
        "prompt": target["prompt"],
        "foreground_present": probe["foreground_present"],
        "target_object_count": len(targets),
        "predicted_object_count": int(probe["predicted_object_count"]),
        "count_correct": count_correct,
        "probe": _serializable_probe(probe),
        "pairs": pairs,
        "extra_prediction_indices": extras,
        "is_fully_correct": bool(count_correct and pairs and all(item["object_fully_correct"] for item in pairs)),
    }


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    spread = z / denominator * sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return [max(0.0, center - spread), min(1.0, center + spread)]


def _rate(values: Sequence[bool]) -> dict[str, Any]:
    successes = int(sum(values))
    total = len(values)
    return {
        "successes": successes,
        "total": total,
        "value": 0.0 if total == 0 else successes / total,
        "wilson_95": wilson_interval(successes, total),
    }


def aggregate_evaluations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = [pair for row in rows for pair in row["pairs"]]
    shape_groups: dict[str, list[Mapping[str, Any]]] = {shape: [] for shape in SHAPES}
    region_groups: dict[str, list[Mapping[str, Any]]] = {}
    size_groups: dict[str, list[Mapping[str, Any]]] = {size: [] for size in SIZES}
    for pair in pairs:
        shape_groups[pair["target"]["shape"]].append(pair)
        region_groups.setdefault(pair["target"]["position_region"], []).append(pair)
        size_groups[pair["target"]["size"]].append(pair)
    result = {
        "sample_count": len(rows),
        "target_object_count": len(pairs),
        "foreground_presence": _rate([bool(row["foreground_present"]) for row in rows]),
        "count_accuracy": _rate([bool(row["count_correct"]) for row in rows]),
        "color_accuracy": _rate([bool(pair["color_correct"]) for pair in pairs]),
        "shape_accuracy": _rate([bool(pair["shape_correct"]) for pair in pairs]),
        "size_accuracy": _rate([bool(pair["size_correct"]) for pair in pairs]),
        "position_accuracy": _rate([bool(pair["position_correct"]) for pair in pairs]),
        "horizontal_accuracy": _rate([bool(pair["horizontal_correct"]) for pair in pairs]),
        "vertical_accuracy": _rate([bool(pair["vertical_correct"]) for pair in pairs]),
        "fully_correct_rate": _rate([bool(row["is_fully_correct"]) for row in rows]),
        "normalized_template_iou_mean": float(np.mean([pair["normalized_template_iou"] for pair in pairs])),
        "boundary_score_mean": float(np.mean([pair["boundary_score"] for pair in pairs])),
        "target_position_mask_iou_mean": float(np.mean([pair["target_position_mask_iou"] for pair in pairs])),
        "center_distance_mean": float(np.mean([pair["center_distance"] for pair in pairs])),
        "shape_groups": {},
        "region_groups": {},
        "size_groups": {},
    }
    for shape, items in shape_groups.items():
        result["shape_groups"][shape] = {
            "count": len(items),
            "shape_accuracy": _rate([bool(item["shape_correct"]) for item in items]),
            "normalized_template_iou_mean": float(np.mean([item["normalized_template_iou"] for item in items])),
            "boundary_score_mean": float(np.mean([item["boundary_score"] for item in items])),
        }
    for region, items in sorted(region_groups.items()):
        result["region_groups"][region] = {
            "count": len(items),
            "position_accuracy": _rate([bool(item["position_correct"]) for item in items]),
            "center_distance_mean": float(np.mean([item["center_distance"] for item in items])),
        }
    for size, items in size_groups.items():
        result["size_groups"][size] = {
            "count": len(items),
            "size_accuracy": _rate([bool(item["size_correct"]) for item in items]),
        }
    return result
