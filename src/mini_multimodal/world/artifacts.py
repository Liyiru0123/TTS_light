"""Persistent regression fixtures and verification for semantic scenes."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from .sampling import SamplerConfig, SceneSampler
from .schema import Scene
from .serialization import scene_from_dict, scene_to_dict


def _json_text(value: Any, *, pretty: bool) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _config_to_dict(config: SamplerConfig) -> dict[str, Any]:
    raw = asdict(config)
    for key in ("shapes", "colors", "sizes", "motions"):
        raw[key] = [item.value for item in raw[key]]
    return raw


def _scene_statistics(scenes: Iterable[Scene]) -> dict[str, Any]:
    shapes: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    motions: Counter[str] = Counter()
    object_count = 0
    scene_count = 0

    for scene in scenes:
        scene_count += 1
        object_count += len(scene.objects)
        for obj in scene.objects:
            shapes[obj.shape.value] += 1
            colors[obj.color.value] += 1
            sizes[obj.size.value] += 1
            motions[obj.motion.direction.value] += 1

    return {
        "scene_count": scene_count,
        "object_count": object_count,
        "shapes": dict(sorted(shapes.items())),
        "colors": dict(sorted(colors.items())),
        "sizes": dict(sorted(sizes.items())),
        "motions": dict(sorted(motions.items())),
    }


def export_scene_fixture(
    output_dir: str | Path,
    *,
    count: int = 200,
    base_seed: int = 2026,
    config: SamplerConfig | None = None,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("Fixture count must be positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = config or SamplerConfig()
    sampler = SceneSampler(base_seed=base_seed, config=config)
    scenes = [sampler.sample(index) for index in range(count)]

    lines = [_json_text(scene_to_dict(scene), pretty=False) for scene in scenes]
    jsonl_path = destination / "scenes.jsonl"
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    digest = sha256(jsonl_path.read_bytes()).hexdigest()

    sample_path = destination / "sample_scene_0000.json"
    sample_path.write_text(
        _json_text(scene_to_dict(scenes[0]), pretty=True),
        encoding="utf-8",
        newline="\n",
    )

    manifest = {
        "artifact_version": 1,
        "schema_version": scenes[0].schema_version,
        "base_seed": base_seed,
        "count": count,
        "config": _config_to_dict(config),
        "scenes_file": jsonl_path.name,
        "scenes_sha256": digest,
        "scenes_bytes": jsonl_path.stat().st_size,
        "statistics": _scene_statistics(scenes),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(_json_text(manifest, pretty=True), encoding="utf-8", newline="\n")
    return manifest


def load_scene_fixture(path: str | Path) -> list[Scene]:
    source = Path(path)
    scenes = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            scenes.append(scene_from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid scene fixture at line {line_number}: {exc}") from exc
    return scenes


def verify_scene_fixture(directory: str | Path) -> dict[str, Any]:
    source_dir = Path(directory)
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    scenes_path = source_dir / manifest["scenes_file"]
    actual_digest = sha256(scenes_path.read_bytes()).hexdigest()
    if actual_digest != manifest["scenes_sha256"]:
        raise ValueError("Scene fixture SHA-256 mismatch")

    scenes = load_scene_fixture(scenes_path)
    if len(scenes) != manifest["count"]:
        raise ValueError("Scene fixture count does not match manifest")
    actual_statistics = _scene_statistics(scenes)
    if actual_statistics != manifest["statistics"]:
        raise ValueError("Scene fixture statistics do not match manifest")

    return {
        "verified": True,
        "scene_count": len(scenes),
        "sha256": actual_digest,
        "statistics": actual_statistics,
    }


def benchmark_sampler(
    *, count: int = 5000, base_seed: int = 2026, config: SamplerConfig | None = None
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("Benchmark count must be positive")
    sampler = SceneSampler(base_seed=base_seed, config=config or SamplerConfig())
    start = perf_counter()
    scenes = [sampler.sample(index) for index in range(count)]
    elapsed = perf_counter() - start
    return {
        "scene_count": count,
        "object_count": sum(len(scene.objects) for scene in scenes),
        "elapsed_seconds": elapsed,
        "scenes_per_second": count / elapsed,
        "base_seed": base_seed,
    }

