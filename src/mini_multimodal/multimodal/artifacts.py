"""Build and verify the portable four-modality scene index."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
import soundfile as sf
from PIL import Image

from mini_multimodal.data.artifacts import load_dataset_bundle
from mini_multimodal.tasks.generation import build_generation_conditions
from mini_multimodal.tasks.reasoning import answer_question
from mini_multimodal.tasks.serialization import condition_to_dict
from mini_multimodal.video.artifacts import load_frames
from mini_multimodal.vision.dataset import iter_image_specs

from .visualization import save_sample_overview


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in values
        ),
        encoding="utf-8",
        newline="\n",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _source_info(project_root: Path, relative_path: str) -> dict[str, object]:
    source = project_root / relative_path
    manifest = source / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"source manifest is missing: {manifest}")
    return {"path": Path(relative_path).as_posix(), "manifest_sha256": _sha(manifest)}


def _unique_or_none(values: Iterable[object], label: str) -> object | None:
    distinct = set(values)
    if len(distinct) > 1:
        raise ValueError(f"scene records have inconsistent {label}")
    return next(iter(distinct)) if distinct else None


def _asset(
    project_root: Path,
    source_dir: str,
    source_name: str,
    expected_sha: str,
    source_manifest: Mapping[str, Any],
    *,
    verify_hash: bool,
) -> tuple[dict[str, object], int, int]:
    entry = source_manifest["files"].get(source_name)
    relative_path = (Path(source_dir) / source_name).as_posix()
    path = project_root / relative_path
    mismatch = 0
    if entry is None or entry["sha256"] != expected_sha or not path.is_file():
        mismatch = 1
    elif verify_hash and _sha(path) != expected_sha:
        mismatch = 1
    bytes_count = 0 if not path.is_file() else path.stat().st_size
    return (
        {"path": relative_path, "sha256": expected_sha, "bytes": bytes_count},
        mismatch,
        bytes_count,
    )


def build_integration_index(
    project_root: str | Path,
    sources: Mapping[str, str],
    *,
    verify_asset_hashes: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, object], dict[str, int], dict[str, object]]:
    root = Path(project_root).resolve()
    directories = {name: root / sources[name] for name in ("dataset", "image", "speech", "video")}
    bundle = load_dataset_bundle(directories["dataset"])
    image_rows = _read_jsonl(directories["image"] / "annotations.jsonl")
    speech_rows = _read_jsonl(directories["speech"] / "annotations.jsonl")
    video_rows = _read_jsonl(directories["video"] / "annotations.jsonl")
    specs = iter_image_specs(bundle)
    if not (len(specs) == len(image_rows) == len(speech_rows) == len(video_rows)):
        raise ValueError("source modalities contain different scene counts")

    source_manifests = {
        name: json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for name, directory in directories.items()
        if name != "dataset"
    }
    records_by_scene: dict[str, list[Any]] = defaultdict(list)
    for record in bundle.all_records:
        records_by_scene[record.scene_id].append(record)

    counters: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    asset_bytes = 0
    maximum_pixel_center_delta = 0
    raster_different_scene_count = 0

    for source_row, (spec, image_row, speech_row, video_row) in enumerate(
        zip(specs, image_rows, speech_rows, video_rows, strict=True)
    ):
        scene = spec.scene
        scene_id = scene.scene_id
        scene_records = records_by_scene[scene_id]
        identity_values = (
            (image_row["annotation"]["scene_id"], image_row["pool"], image_row["split"], image_row["variant"]),
            (speech_row["annotation"]["scene_id"], speech_row["pool"], speech_row["split"], speech_row["variant"]),
            (video_row["annotation"]["scene_id"], video_row["pool"], video_row["split"], video_row["variant"]),
        )
        expected_identity = (scene_id, spec.pool, spec.split.value, spec.variant.value)
        counters["identity_mismatch_count"] += sum(
            value != expected_identity for value in identity_values
        )

        conditions = {
            rendered.condition.target_modality.value: condition_to_dict(rendered)
            for rendered in build_generation_conditions(scene)
        }
        counters["generation_condition_count"] += len(conditions)
        counters["generation_condition_mismatch_count"] += sum(
            condition["condition"]["scene_id"] != scene_id
            or condition["condition"]["target_modality"] != modality
            for modality, condition in conditions.items()
        )
        speech_annotation = speech_row["annotation"]
        if speech_annotation["text"] != conditions["speech"]["text_zh"]:
            counters["text_speech_mismatch_count"] += 1

        cues = (
            image_row["annotation"]["cue_token"],
            speech_annotation["cue_token"],
            video_row["annotation"]["cue_token"],
        )
        counters["cue_mismatch_count"] += sum(cue != spec.cue_token for cue in cues)

        image_objects = {item["object_id"]: item for item in image_row["annotation"]["objects"]}
        video_tracks = {item["object_id"]: item for item in video_row["annotation"]["tracks"]}
        scene_raster_differs = False
        for obj in scene.objects:
            image_object = image_objects.get(obj.object_id)
            video_track = video_tracks.get(obj.object_id)
            if image_object is None or video_track is None:
                counters["object_identity_mismatch_count"] += 1
                continue
            expected_attributes = (obj.shape.value, obj.color.value, obj.size.value)
            if (
                image_object["shape"], image_object["color"], image_object["size"]
            ) != expected_attributes or (
                video_track["shape"], video_track["color"], video_track["size"]
            ) != expected_attributes:
                counters["object_attribute_mismatch_count"] += 1
            image_center = np.asarray(image_object["center_normalized"], dtype=np.float64)
            video_point = video_track["points"][0]
            video_center = np.asarray(video_point["center_normalized"], dtype=np.float64)
            if not np.array_equal(image_center, video_center):
                counters["continuous_center_mismatch_count"] += 1
            pixel_delta = max(
                abs(int(a) - int(b))
                for a, b in zip(image_object["center_xy"], video_point["center_xy"], strict=True)
            )
            maximum_pixel_center_delta = max(maximum_pixel_center_delta, pixel_delta)
            scene_raster_differs |= pixel_delta > 0
        raster_different_scene_count += int(scene_raster_differs)

        for record in scene_records:
            if answer_question(scene, record.task.question) != record.task.answer:
                counters["task_answer_mismatch_count"] += 1

        image_asset, mismatch, size = _asset(
            root,
            sources["image"],
            image_row["relative_path"],
            image_row["png_sha256"],
            source_manifests["image"],
            verify_hash=verify_asset_hashes,
        )
        counters["asset_hash_mismatch_count"] += mismatch
        asset_bytes += size
        audio_asset, mismatch, size = _asset(
            root,
            sources["speech"],
            speech_row["audio_path"],
            speech_row["audio_sha256"],
            source_manifests["speech"],
            verify_hash=verify_asset_hashes,
        )
        counters["asset_hash_mismatch_count"] += mismatch
        asset_bytes += size
        logmel_asset, mismatch, size = _asset(
            root,
            sources["speech"],
            speech_row["logmel_path"],
            speech_row["logmel_sha256"],
            source_manifests["speech"],
            verify_hash=verify_asset_hashes,
        )
        counters["asset_hash_mismatch_count"] += mismatch
        asset_bytes += size
        video_asset, mismatch, size = _asset(
            root,
            sources["video"],
            video_row["frames_path"],
            video_row["frames_sha256"],
            source_manifests["video"],
            verify_hash=verify_asset_hashes,
        )
        counters["asset_hash_mismatch_count"] += mismatch
        asset_bytes += size

        image_asset.update(shape=[64, 64, 3], storage_dtype="uint8")
        audio_asset.update(
            sample_rate=int(speech_annotation["sample_rate"]),
            num_samples=int(speech_annotation["num_samples"]),
            storage_dtype="PCM_16",
            return_dtype="float32",
        )
        logmel_asset.update(
            shape=list(speech_annotation["logmel_shape"]),
            storage_dtype="float16",
            return_dtype="float32",
        )
        video_annotation = video_row["annotation"]
        video_asset.update(
            shape=[
                int(video_annotation["frame_count"]),
                int(video_annotation["height"]),
                int(video_annotation["width"]),
                3,
            ],
            fps=int(video_annotation["fps"]),
            storage_dtype="uint8",
        )

        rows.append(
            {
                "schema_version": 1,
                "source_row": source_row,
                "scene_id": scene_id,
                "pool": spec.pool,
                "split": spec.split.value,
                "variant": spec.variant.value,
                "cue_token": spec.cue_token,
                "cue_aligned": _unique_or_none(
                    (record.cue_aligned for record in scene_records), "cue alignment"
                ),
                "composition_key": _unique_or_none(
                    (record.composition_key for record in scene_records), "composition key"
                ),
                "qa_record_ids": [record.record_id for record in scene_records],
                "generation_conditions": conditions,
                "assets": {
                    "image": image_asset,
                    "audio": audio_asset,
                    "logmel": logmel_asset,
                    "video": video_asset,
                },
            }
        )

    statistics: dict[str, object] = {
        "scene_count": len(rows),
        "qa_record_count": len(bundle.all_records),
        "generation_condition_count": counters["generation_condition_count"],
        "asset_reference_count": len(rows) * 4,
        "pool_scene_counts": dict(sorted(Counter(row["pool"] for row in rows).items())),
        "split_scene_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "variant_scene_counts": dict(sorted(Counter(row["variant"] for row in rows).items())),
    }
    consistency = {
        key: int(counters[key])
        for key in (
            "identity_mismatch_count",
            "generation_condition_mismatch_count",
            "text_speech_mismatch_count",
            "cue_mismatch_count",
            "object_identity_mismatch_count",
            "object_attribute_mismatch_count",
            "continuous_center_mismatch_count",
            "task_answer_mismatch_count",
            "asset_hash_mismatch_count",
        )
    }
    consistency.update(
        image_video_pixel_center_different_scene_count=raster_different_scene_count,
        max_image_video_pixel_center_delta=maximum_pixel_center_delta,
    )
    asset_verification: dict[str, object] = {
        "verified": consistency["asset_hash_mismatch_count"] == 0,
        "checked_asset_count": len(rows) * 4,
        "checked_asset_bytes": asset_bytes,
        "asset_hash_mismatch_count": consistency["asset_hash_mismatch_count"],
    }
    return rows, statistics, consistency, asset_verification


def _load_raw_sample(project_root: Path, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    assets = row["assets"]
    with Image.open(project_root / assets["image"]["path"]) as source:
        image = np.array(source.convert("RGB"), dtype=np.uint8, copy=True)
    waveform, _ = sf.read(
        project_root / assets["audio"]["path"], dtype="float32", always_2d=False
    )
    logmel = np.load(project_root / assets["logmel"]["path"], allow_pickle=False).astype(
        np.float32
    )
    video = load_frames(project_root / assets["video"]["path"])
    return {"image": image, "waveform": waveform, "logmel": logmel, "video": video}


def export_unified_artifacts(
    project_root: str | Path,
    config_path: str | Path,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    sources = {
        name: str(config[f"source_{name}"])
        for name in ("dataset", "image", "speech", "video")
    }
    output = root / str(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rows, statistics, consistency, asset_verification = build_integration_index(
        root, sources, verify_asset_hashes=True
    )
    _write_jsonl(output / "scenes.jsonl", rows)

    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        selected.setdefault(row["variant"], row)
    fixed_rows = [selected[name] for name in sorted(selected)]
    fixed_batch = {
        "scene_ids": [row["scene_id"] for row in fixed_rows],
        "variants": [row["variant"] for row in fixed_rows],
        "contracts": [],
    }
    for row in fixed_rows:
        sample = _load_raw_sample(root, row)
        fixed_batch["contracts"].append(
            {
                "scene_id": row["scene_id"],
                "image_shape": list(sample["image"].shape),
                "image_dtype": str(sample["image"].dtype),
                "waveform_shape": list(sample["waveform"].shape),
                "waveform_dtype": str(sample["waveform"].dtype),
                "logmel_shape": list(sample["logmel"].shape),
                "logmel_dtype": str(sample["logmel"].dtype),
                "video_shape": list(sample["video"].shape),
                "video_dtype": str(sample["video"].dtype),
            }
        )
    _write_json(output / "fixed_batch.json", fixed_batch)

    preview_row = selected.get("shortcut_iid", rows[0])
    save_sample_overview(_load_raw_sample(root, preview_row), output / "sample_overview.png")
    managed = {}
    for name in ("scenes.jsonl", "fixed_batch.json", "sample_overview.png"):
        path = output / name
        managed[name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    manifest: dict[str, object] = {
        "artifact_version": 1,
        "index_schema_version": 1,
        "sources": {
            name: _source_info(root, relative_path)
            for name, relative_path in sources.items()
        },
        "default_modalities": list(config["default_modalities"]),
        "cache_max_bytes": int(config["cache_max_bytes"]),
        "statistics": statistics,
        "consistency": consistency,
        "asset_verification": asset_verification,
        "build_seconds": perf_counter() - started,
        "files": managed,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def verify_unified_artifacts(index_dir: str | Path) -> dict[str, object]:
    source = Path(index_dir).resolve()
    project_root = source.parents[2]
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["files"].items():
        path = source / name
        if not path.is_file() or _sha(path) != metadata["sha256"]:
            raise ValueError(f"Unified artifact {name} SHA-256 does not match")
    sources = {name: value["path"] for name, value in manifest["sources"].items()}
    for name, value in manifest["sources"].items():
        current = _sha(project_root / value["path"] / "manifest.json")
        if current != value["manifest_sha256"]:
            raise ValueError(f"Source {name} manifest SHA-256 does not match")
    rebuilt, statistics, consistency, asset_verification = build_integration_index(
        project_root, sources, verify_asset_hashes=True
    )
    saved = _read_jsonl(source / "scenes.jsonl")
    if rebuilt != saved:
        raise ValueError("Unified scene index differs from source modalities")
    if consistency["asset_hash_mismatch_count"]:
        raise ValueError("Source asset SHA-256 does not match")
    if statistics != manifest["statistics"] or consistency != manifest["consistency"]:
        raise ValueError("Unified statistics or consistency report differs")
    return {
        "verified": True,
        "managed_file_count": len(manifest["files"]),
        "statistics": statistics,
        "consistency": consistency,
        "asset_verification": asset_verification,
    }
