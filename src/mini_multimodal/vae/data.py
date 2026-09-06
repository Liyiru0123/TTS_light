"""Full-scene image and foreground-mask data for P2-W02."""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mini_multimodal.vision.masks import decode_rle


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _select_overfit(rows: list[dict[str, Any]], count: int) -> list[str]:
    if count < 15:
        raise ValueError("overfit fixture needs at least 15 scenes")
    selected: list[dict[str, Any]] = []
    covered_shapes: set[str] = set()
    covered_colors: set[str] = set()
    covered_counts: set[int] = set()
    for row in rows:
        annotation = row["annotation"]
        shapes = {item["shape"] for item in annotation["objects"]}
        colors = {item["color"] for item in annotation["objects"]}
        object_count = len(annotation["objects"])
        has_small = any(item["visible_area_pixels"] <= 129 for item in annotation["objects"])
        adds_coverage = (
            bool(shapes - covered_shapes)
            or bool(colors - covered_colors)
            or object_count not in covered_counts
            or (has_small and not any(
                any(obj["visible_area_pixels"] <= 129 for obj in old["annotation"]["objects"])
                for old in selected
            ))
        )
        if adds_coverage or len(selected) < count:
            selected.append(row)
            covered_shapes.update(shapes)
            covered_colors.update(colors)
            covered_counts.add(object_count)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("insufficient rows for overfit fixture")
    return [row["annotation"]["scene_id"] for row in selected]


def build_p2_w02_data_spec(
    project_root: str | Path,
    *,
    source_index: str,
    source_images: str,
    overfit_count: int,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    index_dir = root / source_index
    image_dir = root / source_images
    index_rows = _read_jsonl(index_dir / "scenes.jsonl")
    annotation_rows = _read_jsonl(image_dir / "annotations.jsonl")
    if len(index_rows) != len(annotation_rows):
        raise ValueError("index and image annotations have different lengths")
    by_scene = {row["scene_id"]: row for row in index_rows}
    if len(by_scene) != len(index_rows):
        raise ValueError("duplicate scene id in source index")

    split_ids: dict[str, list[str]] = {name: [] for name in ("train", "validation", "test")}
    widths: list[int] = []
    heights: list[int] = []
    areas: list[int] = []
    foreground_ratios: list[float] = []
    object_counts: Counter[int] = Counter()
    shape_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    annotation_by_scene: dict[str, dict[str, Any]] = {}
    for annotation_row in annotation_rows:
        annotation = annotation_row["annotation"]
        scene_id = annotation["scene_id"]
        index_row = by_scene.get(scene_id)
        if index_row is None or annotation_row["split"] != index_row["split"]:
            raise ValueError(f"image/index identity mismatch: {scene_id}")
        split_ids[index_row["split"]].append(scene_id)
        annotation_by_scene[scene_id] = annotation_row
        object_counts[len(annotation["objects"])] += 1
        foreground = 0
        for obj in annotation["objects"]:
            x, y, width, height = obj["visible_bbox_xywh"]
            del x, y
            widths.append(int(width))
            heights.append(int(height))
            areas.append(int(obj["visible_area_pixels"]))
            foreground += int(obj["visible_area_pixels"])
            shape_counts[obj["shape"]] += 1
            color_counts[obj["color"]] += 1
        foreground_ratios.append(foreground / (annotation["width"] * annotation["height"]))
    for values in split_ids.values():
        values.sort()

    train_rows = [annotation_by_scene[scene_id] for scene_id in split_ids["train"]]
    overfit_ids = _select_overfit(train_rows, overfit_count)
    width_array = np.asarray(widths)
    height_array = np.asarray(heights)
    area_array = np.asarray(areas)
    foreground_array = np.asarray(foreground_ratios)
    statistics = {
        "scene_count": len(index_rows),
        "object_count": len(areas),
        "split_counts": {name: len(values) for name, values in split_ids.items()},
        "objects_per_scene": {str(key): value for key, value in sorted(object_counts.items())},
        "shape_counts": dict(sorted(shape_counts.items())),
        "color_counts": dict(sorted(color_counts.items())),
        "bbox_width_px": {
            "min": int(width_array.min()),
            "p10": float(np.percentile(width_array, 10)),
            "median": float(np.median(width_array)),
            "p90": float(np.percentile(width_array, 90)),
            "max": int(width_array.max()),
        },
        "bbox_height_px": {
            "min": int(height_array.min()),
            "p10": float(np.percentile(height_array, 10)),
            "median": float(np.median(height_array)),
            "p90": float(np.percentile(height_array, 90)),
            "max": int(height_array.max()),
        },
        "visible_area_px": {
            "min": int(area_array.min()),
            "median": float(np.median(area_array)),
            "p90": float(np.percentile(area_array, 90)),
            "max": int(area_array.max()),
        },
        "scene_foreground_ratio": {
            "min": float(foreground_array.min()),
            "median": float(np.median(foreground_array)),
            "p90": float(np.percentile(foreground_array, 90)),
            "max": float(foreground_array.max()),
        },
        "small_object_area_threshold": int(np.median(area_array)),
        "occluded_object_count": sum(
            obj["visibility_fraction"] < 1.0
            for row in annotation_rows
            for obj in row["annotation"]["objects"]
        ),
    }
    return {
        "split_ids": split_ids,
        "overfit_ids": overfit_ids,
        "statistics": statistics,
        "source_index_manifest_sha256": _sha(index_dir / "manifest.json"),
        "source_image_manifest_sha256": _sha(image_dir / "manifest.json"),
    }


def export_p2_w02_data(
    project_root: str | Path,
    output_dir: str | Path,
    *,
    source_index: str,
    source_images: str,
    overfit_count: int,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_dir)
    spec = build_p2_w02_data_spec(
        root,
        source_index=source_index,
        source_images=source_images,
        overfit_count=overfit_count,
    )
    _write_json(output / "split_ids.json", spec["split_ids"])
    _write_json(output / "overfit_ids.json", spec["overfit_ids"])
    _write_json(output / "data_statistics.json", spec["statistics"])
    files = {
        name: {"bytes": (output / name).stat().st_size, "sha256": _sha(output / name)}
        for name in ("split_ids.json", "overfit_ids.json", "data_statistics.json")
    }
    manifest = {
        "artifact_version": 1,
        "source_index": source_index,
        "source_images": source_images,
        "source_index_manifest_sha256": spec["source_index_manifest_sha256"],
        "source_image_manifest_sha256": spec["source_image_manifest_sha256"],
        "files": files,
        "statistics": spec["statistics"],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def verify_p2_w02_data(project_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if _sha(output / name) != expected["sha256"]:
            raise ValueError(f"P2-W02 data file differs from manifest: {name}")
    if _sha(root / manifest["source_index"] / "manifest.json") != manifest["source_index_manifest_sha256"]:
        raise ValueError("P2-W02 source index differs from manifest")
    if _sha(root / manifest["source_images"] / "manifest.json") != manifest["source_image_manifest_sha256"]:
        raise ValueError("P2-W02 source images differ from manifest")
    return {"verified": True, "statistics": manifest["statistics"], "files": manifest["files"]}


class VAEImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        project_root: str | Path,
        *,
        source_index: str,
        source_images: str,
        scene_ids: Sequence[str],
    ) -> None:
        self.root = Path(project_root).resolve()
        index_rows = _read_jsonl(self.root / source_index / "scenes.jsonl")
        annotations = _read_jsonl(self.root / source_images / "annotations.jsonl")
        row_by_scene = {row["scene_id"]: row for row in index_rows}
        annotation_by_scene = {
            row["annotation"]["scene_id"]: row for row in annotations
        }
        self.entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self._cache: dict[int, dict[str, Any]] = {}
        for scene_id in scene_ids:
            if scene_id not in row_by_scene or scene_id not in annotation_by_scene:
                raise ValueError(f"unknown VAE scene id: {scene_id}")
            self.entries.append((row_by_scene[scene_id], annotation_by_scene[scene_id]))
        if not self.entries:
            raise ValueError("VAE dataset requires at least one scene")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index in self._cache:
            return self._cache[index]
        row, annotation_row = self.entries[index]
        path = (self.root / row["assets"]["image"]["path"]).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("image path escapes project root") from error
        with Image.open(path) as source:
            image = np.array(source.convert("RGB"), dtype=np.uint8, copy=True)
        annotation = annotation_row["annotation"]
        foreground = np.zeros((annotation["height"], annotation["width"]), dtype=np.bool_)
        for obj in annotation["objects"]:
            foreground |= decode_rle(
                obj["visible_rle"], annotation["height"], annotation["width"]
            )
        sample = {
            "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "foreground_mask": torch.from_numpy(foreground.copy()).unsqueeze(0),
            "background_rgb": torch.tensor(annotation["background_rgb"], dtype=torch.float32) / 255.0,
            "metadata": {
                "scene_id": row["scene_id"],
                "split": row["split"],
                "variant": row["variant"],
                "objects": annotation["objects"],
                "image_sha256": row["assets"]["image"]["sha256"],
            },
        }
        self._cache[index] = sample
        return sample


def collate_vae_images(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("VAE batch requires samples")
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "foreground_mask": torch.stack([sample["foreground_mask"] for sample in samples]),
        "background_rgb": torch.stack([sample["background_rgb"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }
