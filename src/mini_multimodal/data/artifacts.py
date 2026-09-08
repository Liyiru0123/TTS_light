"""Persistent, hashed dataset artifacts for P1-W03."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from mini_multimodal.world.schema import Scene
from mini_multimodal.world.serialization import scene_from_dict, scene_to_dict

from .diagnostics import validate_dataset_bundle
from .schema import DatasetBundle, DatasetRecord
from .serialization import (
    config_from_dict,
    config_to_dict,
    dataset_record_from_dict,
    dataset_record_to_dict,
)


MANAGED_FILES = (
    "core_scenes.jsonl",
    "core_records.jsonl",
    "shortcut_scenes.jsonl",
    "shortcut_records.jsonl",
    "compositional_scenes.jsonl",
    "compositional_records.jsonl",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_pool(
    destination: Path,
    name: str,
    scenes: tuple[Scene, ...],
    records: tuple[DatasetRecord, ...],
) -> None:
    _write_jsonl(destination / f"{name}_scenes.jsonl", (scene_to_dict(scene) for scene in scenes))
    _write_jsonl(
        destination / f"{name}_records.jsonl",
        (dataset_record_to_dict(record) for record in records),
    )


def export_dataset_bundle(output_dir: str | Path, bundle: DatasetBundle) -> dict[str, object]:
    destination = Path(output_dir)
    validation = validate_dataset_bundle(bundle)
    _write_pool(destination, "core", bundle.core_scenes, bundle.core_records)
    _write_pool(
        destination, "shortcut", bundle.shortcut_scenes, bundle.shortcut_records
    )
    _write_pool(
        destination,
        "compositional",
        bundle.compositional_scenes,
        bundle.compositional_records,
    )

    files = {
        name: {
            "bytes": (destination / name).stat().st_size,
            "sha256": _sha256(destination / name),
        }
        for name in MANAGED_FILES
    }
    manifest = {
        "artifact_version": 1,
        "config": config_to_dict(bundle.config),
        "files": files,
        "statistics": validation["statistics"],
        "shortcut_diagnostics": validation["shortcut"],
        "compositional_report": validation["compositional"],
        "leakage_report": validation["leakage"],
    }
    _write_json(destination / "manifest.json", manifest)
    _write_json(
        destination / "sample_records.json",
        {
            "core": [dataset_record_to_dict(item) for item in bundle.core_records[:2]],
            "shortcut_train": [
                dataset_record_to_dict(item)
                for item in bundle.shortcut_records
                if item.split.value == "train"
            ][:3],
            "counterfactual_test": [
                dataset_record_to_dict(item)
                for item in bundle.shortcut_records
                if item.split.value == "test"
            ][:3],
            "compositional_test": [
                dataset_record_to_dict(item)
                for item in bundle.compositional_records
                if item.split.value == "test"
            ][:3],
        },
    )
    return manifest


def _load_pool(
    source: Path, name: str
) -> tuple[tuple[Scene, ...], tuple[DatasetRecord, ...]]:
    scenes = tuple(
        scene_from_dict(item) for item in _load_jsonl(source / f"{name}_scenes.jsonl")
    )
    records = tuple(
        dataset_record_from_dict(item)
        for item in _load_jsonl(source / f"{name}_records.jsonl")
    )
    return scenes, records


def load_dataset_bundle(output_dir: str | Path) -> DatasetBundle:
    source = Path(output_dir)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    core_scenes, core_records = _load_pool(source, "core")
    shortcut_scenes, shortcut_records = _load_pool(source, "shortcut")
    compositional_scenes, compositional_records = _load_pool(source, "compositional")
    return DatasetBundle(
        config=config_from_dict(manifest["config"]),
        core_scenes=core_scenes,
        core_records=core_records,
        shortcut_scenes=shortcut_scenes,
        shortcut_records=shortcut_records,
        compositional_scenes=compositional_scenes,
        compositional_records=compositional_records,
    )


def verify_dataset_bundle(output_dir: str | Path) -> dict[str, object]:
    source = Path(output_dir)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name in MANAGED_FILES:
        actual = _sha256(source / name)
        if actual != manifest["files"][name]["sha256"]:
            raise ValueError(f"Dataset file {name} SHA-256 does not match its manifest")

    bundle = load_dataset_bundle(source)
    validation = validate_dataset_bundle(bundle)
    for key, manifest_key in (
        ("statistics", "statistics"),
        ("shortcut", "shortcut_diagnostics"),
        ("compositional", "compositional_report"),
        ("leakage", "leakage_report"),
    ):
        if validation[key] != manifest[manifest_key]:
            raise ValueError(f"Dataset {manifest_key} does not match regenerated results")
    return {
        "verified": True,
        "files": {name: manifest["files"][name]["sha256"] for name in MANAGED_FILES},
        **validation,
    }
