"""Map P1-W03 scene pools to one image specification per scene."""

from __future__ import annotations

from dataclasses import dataclass

from mini_multimodal.data.schema import (
    DataSplit,
    DatasetBundle,
    DatasetRecord,
    DatasetVariant,
)
from mini_multimodal.world.schema import Scene


@dataclass(frozen=True, slots=True)
class SceneImageSpec:
    pool: str
    split: DataSplit
    variant: DatasetVariant
    scene: Scene
    cue_token: str | None


def _pool_specs(
    name: str,
    scenes: tuple[Scene, ...],
    records: tuple[DatasetRecord, ...],
) -> tuple[SceneImageSpec, ...]:
    records_by_scene: dict[str, list[DatasetRecord]] = {}
    for record in records:
        records_by_scene.setdefault(record.scene_id, []).append(record)
    specs: list[SceneImageSpec] = []
    for scene in scenes:
        scene_records = records_by_scene.get(scene.scene_id, [])
        if not scene_records:
            raise ValueError(f"Scene {scene.scene_id!r} has no dataset record")
        splits = {record.split for record in scene_records}
        variants = {record.variant for record in scene_records}
        cues = {record.cue_token for record in scene_records}
        if len(splits) != 1 or len(variants) != 1 or len(cues) != 1:
            raise ValueError(f"Scene {scene.scene_id!r} has inconsistent image metadata")
        specs.append(
            SceneImageSpec(
                pool=name,
                split=next(iter(splits)),
                variant=next(iter(variants)),
                scene=scene,
                cue_token=next(iter(cues)),
            )
        )
    if set(records_by_scene) != {scene.scene_id for scene in scenes}:
        raise ValueError(f"Pool {name!r} contains records for unknown scenes")
    return tuple(specs)


def iter_image_specs(bundle: DatasetBundle) -> tuple[SceneImageSpec, ...]:
    return (
        _pool_specs("core", bundle.core_scenes, bundle.core_records)
        + _pool_specs("shortcut", bundle.shortcut_scenes, bundle.shortcut_records)
        + _pool_specs(
            "compositional",
            bundle.compositional_scenes,
            bundle.compositional_records,
        )
    )

