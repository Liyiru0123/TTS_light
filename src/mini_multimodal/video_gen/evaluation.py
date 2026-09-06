"""Calibrated pixel-level evaluation helpers for P3-W04.

Planner outputs and generated-video measurements are intentionally separated.
Missing detections remain in every denominator and therefore count as failures.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Sequence

import numpy as np
import torch

from mini_multimodal.ldm.artifacts import probe_image
from mini_multimodal.vision.renderer import COLOR_RGB
from mini_multimodal.world.schema import Color


def foreground_mask(frame: np.ndarray) -> np.ndarray:
    """Return the same palette-aware foreground mask used by ``probe_image``."""
    array = np.asarray(frame, dtype=np.float32)
    if array.max() > 1.5:
        array = array / 255.0
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]), axis=0)
    background = np.median(border, axis=0)
    palette = np.asarray([COLOR_RGB[color] for color in Color], dtype=np.float32) / 255.0
    palette_distance = ((array[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=-1)
    background_distance = ((array - background[None, None, :]) ** 2).sum(axis=-1)
    return palette_distance.min(axis=-1) + 0.005 < background_distance


def background_flicker(frames: np.ndarray) -> float:
    """Adjacent-frame difference outside the foreground union of each pair."""
    values: list[float] = []
    arrays = np.asarray(frames, dtype=np.float32)
    if arrays.max() > 1.5:
        arrays = arrays / 255.0
    masks = [foreground_mask(frame) for frame in arrays]
    for index in range(len(arrays) - 1):
        background = ~(masks[index] | masks[index + 1])
        if background.any():
            values.append(float(np.abs(arrays[index + 1] - arrays[index])[background].mean()))
    return float(np.mean(values)) if values else 1.0


def movement_signature(trajectory: Sequence[Sequence[float]], threshold: float = 0.015) -> dict[str, Any]:
    """Direction and moving interval derived only from observed coordinates."""
    value = np.asarray(trajectory, dtype=np.float32)
    if value.shape != (8, 2) or not np.isfinite(value).all():
        return {"valid": False, "direction": None, "start": None, "end": None, "moving": [False] * 7}
    delta = np.diff(value, axis=0)
    norms = np.linalg.norm(delta, axis=1)
    moving = norms > threshold
    indices = np.flatnonzero(moving)
    if not len(indices):
        return {"valid": True, "direction": "static", "start": 0, "end": 0, "moving": moving.tolist()}
    weighted = delta[moving].sum(axis=0)
    if abs(float(weighted[0])) >= abs(float(weighted[1])):
        direction = "right" if weighted[0] > 0 else "left"
    else:
        direction = "down" if weighted[1] > 0 else "up"
    return {
        "valid": True,
        "direction": direction,
        "start": int(indices[0]),
        "end": int(indices[-1] + 1),
        "moving": moving.tolist(),
    }


_SIZE_AREA_PROTOTYPES = {
    "circle": {"small": 61, "medium": 129, "large": 221},
    "pentagon": {"small": 44, "medium": 108, "large": 171},
    "rectangle": {"small": 117, "medium": 153, "large": 273},
    "square": {"small": 81, "medium": 169, "large": 289},
    "triangle": {"small": 41, "medium": 85, "large": 145},
}


def _size_from_area(area: int, shape: str) -> str:
    """Map raster area to size using shape-specific rendered prototypes."""
    prototypes = _SIZE_AREA_PROTOTYPES.get(shape)
    if not prototypes:
        return "large" if area >= 180 else "medium" if area >= 120 else "small"
    # Ratio distance is preferable to raw pixel distance across differently
    # sized shape families.
    safe_area = max(float(area), 1.0)
    return min(prototypes, key=lambda size: abs(np.log(safe_area / prototypes[size])))


def _best_assignment(targets: Sequence[dict], predictions: Sequence[dict], frame_index: int) -> list[dict | None]:
    """Globally match predictions while retaining one slot per target."""
    n_target = len(targets)
    n_pred = len(predictions)
    if not n_pred:
        return [None] * n_target
    padded: list[dict | None] = list(predictions) + [None] * max(0, n_target - n_pred)
    best_cost = float("inf")
    best: list[dict | None] = [None] * n_target
    for candidate in permutations(padded, min(n_target, len(padded))):
        if len(candidate) != n_target:
            continue
        cost = 0.0
        for target, pred in zip(targets, candidate, strict=True):
            if pred is None:
                cost += 10.0
                continue
            cost += 2.0 * (pred["color"] != target["color"])
            cost += 1.0 * (pred["shape"] != target["shape"])
            gt = np.asarray(target.get("trajectory", [[0.5, 0.5]] * 8)[frame_index], dtype=np.float32)
            cost += float(np.linalg.norm(np.asarray(pred["center_xy"]) - gt))
        if cost < best_cost:
            best_cost = cost
            best = list(candidate)
    return best


@torch.no_grad()
def evaluate_generated_video(frames: np.ndarray, targets: Sequence[dict], shape_model) -> dict[str, Any]:
    """Evaluate decoded pixels. Planner predictions are deliberately not accepted."""
    arrays = np.asarray(frames)
    if arrays.shape != (8, 64, 64, 3):
        raise ValueError(f"expected [8,64,64,3], got {arrays.shape}")
    float_arrays = arrays.astype(np.float32)
    if float_arrays.max() > 1.5:
        float_arrays /= 255.0
    probes = [
        probe_image(torch.from_numpy(frame).permute(2, 0, 1), shape_model)
        for frame in float_arrays
    ]
    assignments = [_best_assignment(targets, probe.get("objects", []), frame) for frame, probe in enumerate(probes)]
    target_frame_count = max(len(targets) * 8, 1)
    count_correct = sum(int(probe.get("predicted_object_count", 0) == len(targets)) for probe in probes)
    color_correct = shape_correct = size_correct = detected = 0
    trajectories: list[list[list[float] | None]] = [[] for _ in targets]
    for assignment in assignments:
        for index, (target, pred) in enumerate(zip(targets, assignment, strict=True)):
            if pred is None:
                trajectories[index].append(None)
                continue
            detected += 1
            color_correct += int(pred["color"] == target["color"])
            shape_correct += int(pred["shape"] == target["shape"])
            size_correct += int(_size_from_area(int(pred["area"]), pred["shape"]) == target["size"])
            trajectories[index].append(list(pred["center_xy"]))

    object_rows = []
    direction_correct = timing_correct = motion_total = 0
    trajectory_errors: list[float] = []
    for target, observed in zip(targets, trajectories, strict=True):
        gt = np.asarray(target.get("trajectory", []), dtype=np.float32)
        complete = len(observed) == 8 and all(point is not None for point in observed) and gt.shape == (8, 2)
        if complete:
            pred = np.asarray(observed, dtype=np.float32)
            # The frozen raster probe and WorldSpec use slightly different
            # absolute-center conventions. The frozen protocol compares motion
            # after aligning the first frame, as specified in the test plan.
            pred_aligned = pred - pred[0] + gt[0]
            mae = float(np.abs(pred_aligned - gt).mean())
            generated_motion = movement_signature(pred)
        else:
            mae = 1.0
            generated_motion = {"valid": False, "direction": None, "start": None, "end": None, "moving": [False] * 7}
        gt_motion = movement_signature(gt) if gt.shape == (8, 2) else {"valid": False}
        trajectory_errors.append(mae)
        if gt_motion.get("valid"):
            motion_total += 1
            direction_ok = generated_motion.get("direction") == gt_motion.get("direction")
            timing_ok = bool(
                generated_motion.get("valid")
                and abs(int(generated_motion["start"]) - int(gt_motion["start"])) <= 1
                and abs(int(generated_motion["end"]) - int(gt_motion["end"])) <= 1
            )
            direction_correct += int(direction_ok)
            timing_correct += int(timing_ok)
        else:
            direction_ok = timing_ok = False
        object_rows.append({
            "object_id": target.get("object_id"),
            "trajectory_complete": complete,
            "trajectory_mae": mae,
            "ground_truth_motion": gt_motion,
            "generated_motion": generated_motion,
            "direction_correct": direction_ok,
            "timing_correct": timing_ok,
        })
    return {
        "background_flicker": background_flicker(float_arrays),
        "count_accuracy": count_correct / 8.0,
        "detection_rate": detected / target_frame_count,
        "color_accuracy": color_correct / target_frame_count,
        "size_accuracy": size_correct / target_frame_count,
        "shape_accuracy": shape_correct / target_frame_count,
        "trajectory_mae": float(np.mean(trajectory_errors)) if trajectory_errors else 1.0,
        "direction_accuracy": direction_correct / max(motion_total, 1),
        "timing_accuracy": timing_correct / max(motion_total, 1),
        "object_rows": object_rows,
    }
