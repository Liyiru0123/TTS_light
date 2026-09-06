"""Measure empirical geometric shape IoU distribution for P2-W04.

Direction B (approved 2026-08-08): replace the P2-W01 classifier proxy with a
geometric mask-IoU shape metric (consistent with P2-W02's verified mask_iou).

This script is RESEARCH-ONLY: it does not modify any artifact. It:
1. Builds reference masks from ground-truth shape/size/position using the exact
   renderer geometry (render_scene at canvas=64, output_scale=1).
2. Re-extracts predicted foreground masks from the saved generated images via the
   same palette-distance logic as probe_image.
3. Matches predicted masks to targets by nearest center, then computes
   *normalized silhouette IoU* (crop both masks to their bbox, resize to 32x32,
   binarize, IoU) which isolates shape geometry from position/size.
4. Also reports full-mask IoU for reference (consistent with P2-W02).
5. Reports per-shape and overall distributions to inform threshold setting.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.world.schema import (  # noqa: E402
    Color,
    Position,
    Scene,
    SceneObject,
    Shape,
    Size,
)
from mini_multimodal.vision.renderer import render_scene  # noqa: E402
from mini_multimodal.vision.schema import ImageRenderConfig  # noqa: E402
from mini_multimodal.ldm.artifacts import COLOR_RGB  # noqa: E402

CANVAS = 64
GRID = 32  # normalized silhouette comparison grid


def _rounded(value: float) -> int:
    return int(value + 0.5)


def reference_mask(shape: str, size: str, position_xy) -> np.ndarray:
    """Build the 64-space reference mask using the exact renderer geometry."""
    scene = Scene(
        scene_id="ref",
        seed=0,
        canvas_width=CANVAS,
        canvas_height=CANVAS,
        frame_count=1,
        fps=1,
        objects=(
            SceneObject(
                object_id="o0",
                shape=Shape(shape),
                color=Color.RED,
                size=Size(size),
                initial_position=Position(float(position_xy[0]), float(position_xy[1])),
                visible_start=0,
                visible_end=0,
            ),
        ),
    )
    frame = render_scene(scene, frame_index=0, config=ImageRenderConfig(output_scale=1))
    return np.asarray(frame.amodal_masks["o0"], dtype=bool)


def extract_predicted_mask(image: torch.Tensor) -> np.ndarray:
    """Replicate probe_image's foreground mask (adaptive background = median border)."""
    array = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]), axis=0)
    background = np.median(border, axis=0)
    palette = np.asarray([COLOR_RGB[c] for c in Color], dtype=np.float32) / 255.0
    palette_distance = ((array[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=-1)
    background_distance = ((array - background[None, None, :]) ** 2).sum(axis=-1)
    return palette_distance.min(axis=-1) + 0.005 < background_distance


def component_masks(mask: np.ndarray, min_area: int = 20) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
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
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) >= min_area:
            comp = np.zeros_like(mask, dtype=bool)
            yy, xx = zip(*pixels, strict=True)
            comp[np.asarray(yy), np.asarray(xx)] = True
            result.append(comp)
    return sorted(result, key=lambda v: int(v.sum()), reverse=True)


def center_of(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean() / (CANVAS - 1)), float(ys.mean() / (CANVAS - 1))


def _crop_resize(mask: np.ndarray, grid: int = GRID) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((grid, grid), dtype=bool)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = mask[y0:y1, x0:x1].astype(np.float32)
    t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0)
    r = torch.nn.functional.interpolate(t, size=(grid, grid), mode="bilinear", align_corners=False)
    return (r.squeeze().numpy() > 0.5).astype(bool)


def silhouette_iou(pred: np.ndarray, ref: np.ndarray) -> float:
    """Normalized silhouette IoU: crop to bbox, resize to GRID, binarize, IoU."""
    p = _crop_resize(pred)
    r = _crop_resize(ref)
    inter = float((p & r).sum())
    union = float((p | r).sum())
    return inter / union if union > 0 else 0.0


def full_mask_iou(pred: np.ndarray, ref: np.ndarray) -> float:
    inter = float((pred & ref).sum())
    union = float((pred | ref).sum())
    return inter / union if union > 0 else 0.0


def main() -> None:
    preds_path = ROOT / "results" / "P2-W04_tiny_image_ldm" / "generation_predictions.jsonl"
    lines = [json.loads(ln) for ln in preds_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    sil_scores: list[float] = []
    full_scores: list[float] = []
    by_shape_sil: dict[str, list[float]] = defaultdict(list)
    by_shape_full: dict[str, list[float]] = defaultdict(list)
    matched = 0
    total = 0

    for rec in lines:
        targets = rec["target_objects"]
        img_path = rec.get("generated_image")
        if not img_path:
            continue
        img = torch.from_numpy(np.asarray(Image.open(ROOT / img_path).convert("RGB")).astype(np.float32)).permute(2, 0, 1) / 255.0
        pred_mask = extract_predicted_mask(img)
        pred_comps = component_masks(pred_mask)
        pred_centers = [center_of(c) for c in pred_comps]

        for tgt in targets:
            total += 1
            ref = reference_mask(tgt["shape"], tgt["size"], tgt["position"])
            # match nearest predicted component by center distance
            if not pred_comps:
                sil_scores.append(0.0)
                full_scores.append(0.0)
                by_shape_sil[tgt["shape"]].append(0.0)
                by_shape_full[tgt["shape"]].append(0.0)
                continue
            tgt_c = (tgt["position"][0], tgt["position"][1])
            best_i = min(range(len(pred_comps)), key=lambda i: (pred_centers[i][0] - tgt_c[0]) ** 2 + (pred_centers[i][1] - tgt_c[1]) ** 2)
            pc = pred_comps[best_i]
            matched += 1
            s = silhouette_iou(pc, ref)
            f = full_mask_iou(pc, ref)
            sil_scores.append(s)
            full_scores.append(f)
            by_shape_sil[tgt["shape"]].append(s)
            by_shape_full[tgt["shape"]].append(f)

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        return {
            "n": n,
            "mean": round(sum(xs) / n, 4),
            "median": round(xs[n // 2], 4),
            "p10": round(xs[max(0, n // 10)], 4),
            "p25": round(xs[n // 4], 4),
            "min": round(xs[0], 4),
            "max": round(xs[-1], 4),
        }

    print("=== GEOMETRIC SHAPE IoU (current v2.1 model, best.pt) ===")
    print(f"total target objects: {total}, matched to a component: {matched}")
    print(f"\nNORMALIZED SILHOUETTE IoU (shape-isolated): {stats(sil_scores)}")
    print(f"FULL-MASK IoU (P2-W02-style):               {stats(full_scores)}")
    print("\nPer-shape NORMALIZED silhouette IoU:")
    for sh in sorted(by_shape_sil):
        print(f"  {sh:10s}: {stats(by_shape_sil[sh])}")
    print("\nPer-shape FULL-mask IoU:")
    for sh in sorted(by_shape_full):
        print(f"  {sh:10s}: {stats(by_shape_full[sh])}")

    # Variant split (single vs composition) for threshold alignment
    single_sil, comp_sil = [], []
    for rec in lines:
        variant = rec.get("variant")
        targets = rec["target_objects"]
        img_path = rec.get("generated_image")
        if not img_path:
            continue
        img = torch.from_numpy(np.asarray(Image.open(ROOT / img_path).convert("RGB")).astype(np.float32)).permute(2, 0, 1) / 255.0
        pred_mask = extract_predicted_mask(img)
        pred_comps = component_masks(pred_mask)
        pred_centers = [center_of(c) for c in pred_comps]
        for tgt in targets:
            ref = reference_mask(tgt["shape"], tgt["size"], tgt["position"])
            if not pred_comps:
                s = 0.0
            else:
                tgt_c = (tgt["position"][0], tgt["position"][1])
                best_i = min(range(len(pred_comps)), key=lambda i: (pred_centers[i][0] - tgt_c[0]) ** 2 + (pred_centers[i][1] - tgt_c[1]) ** 2)
                s = silhouette_iou(pred_comps[best_i], ref)
            (single_sil if variant == "single" else comp_sil).append(s)
    print("\n=== VARIANT SPLIT (mean normalized silhouette IoU) ===")
    if single_sil:
        print(f"  single   (n={len(single_sil)}): mean={round(sum(single_sil)/len(single_sil),4)}, frac>=0.70={round(sum(x>=0.70 for x in single_sil)/len(single_sil),4)}")
    if comp_sil:
        print(f"  compos   (n={len(comp_sil)}): mean={round(sum(comp_sil)/len(comp_sil),4)}, frac>=0.70={round(sum(x>=0.70 for x in comp_sil)/len(comp_sil),4)}")

    # ceiling: clean reference vs itself (silhouette IoU should be ~1.0)
    print("\n=== CEILING (clean reference shape vs reference mask, same geometry) ===")
    ceil_sil, ceil_full = [], []
    for sh in ["circle", "square", "triangle", "rectangle", "pentagon"]:
        for sz in ["small", "medium", "large"]:
            ref = reference_mask(sh, sz, (0.5, 0.5))
            ceil_sil.append(silhouette_iou(ref, ref))
            ceil_full.append(full_mask_iou(ref, ref))
    print(f"  self silhouette IoU mean: {round(sum(ceil_sil)/len(ceil_sil),4)}")
    print(f"  self full IoU mean:       {round(sum(ceil_full)/len(ceil_full),4)}")


if __name__ == "__main__":
    main()
