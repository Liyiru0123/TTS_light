"""Persistent PNG images, object annotations, hashes, and verification."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from mini_multimodal.data.schema import DatasetBundle

from .dataset import SceneImageSpec, iter_image_specs
from .diagnostics import validate_image_bundle
from .renderer import render_scene
from .schema import CUE_BACKGROUNDS, ImageRenderConfig
from .serialization import (
    image_annotation_from_dict,
    image_annotation_to_dict,
    render_config_from_dict,
    render_config_to_dict,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for value in values
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _image_relative_path(spec: SceneImageSpec) -> Path:
    return (
        Path("images")
        / spec.pool
        / spec.split.value
        / f"{spec.scene.scene_id}_f000.png"
    )


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _save_contact_sheet(images: list[Image.Image], path: Path) -> None:
    if not images:
        raise ValueError("A contact sheet requires at least one image")
    width, height = images[0].size
    columns = 5
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    _save_png(sheet, path)


def _save_cue_palette(path: Path) -> None:
    tile = 64
    image = Image.new("RGB", (tile * len(CUE_BACKGROUNDS), tile), (255, 255, 255))
    pixels = np.asarray(image).copy()
    for index, cue in enumerate(CUE_BACKGROUNDS):
        pixels[:, index * tile : (index + 1) * tile] = CUE_BACKGROUNDS[cue]
        # A shared black marker makes the background brightness directly comparable.
        pixels[20:44, index * tile + 28 : index * tile + 36] = (30, 30, 30)
    _save_png(Image.fromarray(pixels, mode="RGB"), path)


def export_image_artifacts(
    output_dir: str | Path,
    bundle: DatasetBundle,
    config: ImageRenderConfig | None = None,
    *,
    source_dataset_manifest_sha256: str | None = None,
) -> dict[str, object]:
    destination = Path(output_dir)
    cfg = config or ImageRenderConfig()
    diagnostics = validate_image_bundle(bundle, cfg)
    annotation_records: list[dict[str, Any]] = []
    file_entries: dict[str, dict[str, object]] = {}
    previews: list[Image.Image] = []
    cue_previews: dict[str, Image.Image] = {}

    for spec in iter_image_specs(bundle):
        rendered = render_scene(
            spec.scene, frame_index=0, cue_token=spec.cue_token, config=cfg
        )
        relative_path = _image_relative_path(spec)
        image_path = destination / relative_path
        _save_png(rendered.image, image_path)
        relative_name = relative_path.as_posix()
        digest = _sha256(image_path)
        file_entries[relative_name] = {
            "bytes": image_path.stat().st_size,
            "sha256": digest,
        }
        annotation_records.append(
            {
                "pool": spec.pool,
                "split": spec.split.value,
                "variant": spec.variant.value,
                "relative_path": relative_name,
                "png_sha256": digest,
                "annotation": image_annotation_to_dict(rendered.annotation),
            }
        )
        if len(previews) < 20:
            previews.append(rendered.image.copy())
        if spec.cue_token is not None and spec.cue_token not in cue_previews:
            cue_previews[spec.cue_token] = rendered.image.copy()

    annotations_path = destination / "annotations.jsonl"
    _write_jsonl(annotations_path, annotation_records)
    contact_path = destination / "contact_sheet.png"
    ordered_previews = list(cue_previews.values()) + previews
    _save_contact_sheet(ordered_previews[:20], contact_path)
    cue_palette_path = destination / "cue_palette.png"
    _save_cue_palette(cue_palette_path)
    for path in (annotations_path, contact_path, cue_palette_path):
        relative_name = path.relative_to(destination).as_posix()
        file_entries[relative_name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    manifest = {
        "artifact_version": 1,
        "source_dataset_manifest_sha256": source_dataset_manifest_sha256,
        "render_config": render_config_to_dict(cfg),
        "image_count": len(annotation_records),
        "annotation_count": len(annotation_records),
        "files": dict(sorted(file_entries.items())),
        **diagnostics,
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def verify_image_artifacts(
    output_dir: str | Path,
    bundle: DatasetBundle,
) -> dict[str, object]:
    source = Path(output_dir)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for relative_name, expected in manifest["files"].items():
        path = source / Path(relative_name)
        if not path.is_file():
            raise ValueError(f"Managed image artifact is missing: {relative_name}")
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"Image artifact {relative_name} SHA-256 does not match")

    cfg = render_config_from_dict(manifest["render_config"])
    records = _load_jsonl(source / "annotations.jsonl")
    specs = iter_image_specs(bundle)
    if len(records) != len(specs) or len(records) != manifest["image_count"]:
        raise ValueError("Image and annotation counts do not match the manifest")

    for spec, record in zip(specs, records, strict=True):
        expected_path = _image_relative_path(spec).as_posix()
        if record["relative_path"] != expected_path:
            raise ValueError("Image annotation ordering or path is inconsistent")
        if record["pool"] != spec.pool or record["split"] != spec.split.value:
            raise ValueError("Image annotation dataset metadata is inconsistent")
        if record["variant"] != spec.variant.value:
            raise ValueError("Image annotation variant is inconsistent")
        rendered = render_scene(
            spec.scene, frame_index=0, cue_token=spec.cue_token, config=cfg
        )
        saved = Image.open(source / expected_path).convert("RGB")
        if not np.array_equal(np.asarray(saved), np.asarray(rendered.image)):
            raise ValueError(f"Saved pixels do not match re-rendering: {expected_path}")
        loaded_annotation = image_annotation_from_dict(record["annotation"])
        if loaded_annotation != rendered.annotation:
            raise ValueError(f"Saved annotation does not match re-rendering: {expected_path}")
        if record["png_sha256"] != manifest["files"][expected_path]["sha256"]:
            raise ValueError("Annotation PNG digest does not match the manifest")

    diagnostics = validate_image_bundle(bundle, cfg)
    for key in ("statistics", "consistency", "shortcut"):
        if diagnostics[key] != manifest[key]:
            raise ValueError(f"Image {key} does not match regenerated diagnostics")
    return {
        "verified": True,
        "image_count": len(records),
        "annotation_count": len(records),
        "managed_file_count": len(manifest["files"]),
        "source_dataset_manifest_sha256": manifest[
            "source_dataset_manifest_sha256"
        ],
        **diagnostics,
    }
