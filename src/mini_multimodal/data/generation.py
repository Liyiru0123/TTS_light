"""Build deterministic normal, shortcut, counterfactual, and compositional data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from typing import Callable

from mini_multimodal.tasks.generation import build_qa_tasks
from mini_multimodal.tasks.schema import QATask, QuestionKind
from mini_multimodal.world.sampling import SamplerConfig, SceneSampler
from mini_multimodal.world.schema import Scene, Shape

from .schema import (
    DataSplit,
    DatasetBuildConfig,
    DatasetBundle,
    DatasetRecord,
    DatasetVariant,
)


SHAPE_ORDER = tuple(shape.value for shape in Shape)
CUE_BY_SHAPE = {shape: f"cue_{index}" for index, shape in enumerate(SHAPE_ORDER)}


def _stable_int(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big", signed=False)


def _prefixed_scene(scene: Scene, prefix: str) -> Scene:
    return replace(scene, scene_id=f"{prefix}_{scene.scene_id}")


def _core_pool(config: DatasetBuildConfig) -> tuple[tuple[Scene, DataSplit], ...]:
    sampler = SceneSampler(config.base_seed, SamplerConfig(min_objects=1, max_objects=3))
    split_counts = (
        (DataSplit.TRAIN, config.core_train_scenes),
        (DataSplit.VALIDATION, config.core_validation_scenes),
        (DataSplit.TEST, config.core_test_scenes),
    )
    result: list[tuple[Scene, DataSplit]] = []
    index = 0
    for split, count in split_counts:
        for _ in range(count):
            result.append((_prefixed_scene(sampler.sample(index), "core"), split))
            index += 1
    return tuple(result)


def _collect_stratified(
    sampler: SceneSampler,
    *,
    prefix: str,
    split: DataSplit,
    count: int,
    start_index: int,
    accept: Callable[[Scene], bool],
    max_scan_count: int,
) -> tuple[tuple[Scene, DataSplit], int]:
    quota = count // len(SHAPE_ORDER)
    counts = {shape: 0 for shape in SHAPE_ORDER}
    result: list[tuple[Scene, DataSplit]] = []
    index = start_index
    scanned = 0
    while len(result) < count and scanned < max_scan_count:
        scene = _prefixed_scene(sampler.sample(index), prefix)
        index += 1
        scanned += 1
        shape = scene.objects[0].shape.value
        if counts[shape] >= quota or not accept(scene):
            continue
        counts[shape] += 1
        result.append((scene, split))
    if len(result) != count:
        raise RuntimeError(
            f"Unable to collect {count} balanced {prefix} scenes after {scanned} candidates"
        )
    return tuple(result), index


def _shape_task(scene: Scene) -> QATask:
    matches = [
        task
        for task in build_qa_tasks(scene)
        if task.question.kind is QuestionKind.OBJECT_SHAPE
    ]
    if len(scene.objects) != 1 or len(matches) != 1:
        raise ValueError("Shape benchmark scenes must contain exactly one object and shape task")
    return matches[0]


def _base_record(
    task: QATask,
    *,
    split: DataSplit,
    variant: DatasetVariant,
    composition_key: str | None = None,
) -> DatasetRecord:
    needs_cue = variant in {
        DatasetVariant.SHORTCUT_IID,
        DatasetVariant.COUNTERFACTUAL,
    }
    return DatasetRecord(
        record_id=f"{variant.value}:{split.value}:{task.question.question_id}",
        split=split,
        variant=variant,
        task=task,
        cue_token="__pending__" if needs_cue else None,
        cue_aligned=False if needs_cue else None,
        composition_key=composition_key,
    )


def _core_records(pool: tuple[tuple[Scene, DataSplit], ...]) -> tuple[DatasetRecord, ...]:
    return tuple(
        _base_record(task, split=split, variant=DatasetVariant.NORMAL)
        for scene, split in pool
        for task in build_qa_tasks(scene)
    )


def _wrong_cue(answer: str, record_id: str, seed: int) -> str:
    correct_index = SHAPE_ORDER.index(answer)
    offset = 1 + _stable_int(seed, record_id, "wrong-cue") % (len(SHAPE_ORDER) - 1)
    return f"cue_{(correct_index + offset) % len(SHAPE_ORDER)}"


def _assign_iid_cues(
    records: tuple[DatasetRecord, ...],
    *,
    strength: float,
    seed: int,
) -> tuple[DatasetRecord, ...]:
    grouped: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        grouped[str(record.task.answer.value)].append(record)

    aligned_ids: set[str] = set()
    for answer, group in grouped.items():
        ordered = sorted(
            group, key=lambda record: _stable_int(seed, answer, record.record_id, "align")
        )
        aligned_count = round(len(ordered) * strength)
        aligned_ids.update(record.record_id for record in ordered[:aligned_count])

    result: list[DatasetRecord] = []
    for record in records:
        answer = str(record.task.answer.value)
        aligned = record.record_id in aligned_ids
        cue = CUE_BY_SHAPE[answer] if aligned else _wrong_cue(answer, record.record_id, seed)
        result.append(replace(record, cue_token=cue, cue_aligned=aligned))
    return tuple(result)


def _assign_counterfactual_cues(
    records: tuple[DatasetRecord, ...],
) -> tuple[DatasetRecord, ...]:
    result: list[DatasetRecord] = []
    for record in records:
        answer = str(record.task.answer.value)
        index = SHAPE_ORDER.index(answer)
        cue = f"cue_{(index + 1) % len(SHAPE_ORDER)}"
        result.append(replace(record, cue_token=cue, cue_aligned=False))
    return tuple(result)


def _shortcut_pool(
    config: DatasetBuildConfig,
) -> tuple[tuple[tuple[Scene, DataSplit], ...], tuple[DatasetRecord, ...]]:
    sampler = SceneSampler(
        config.base_seed + 1,
        SamplerConfig(min_objects=1, max_objects=1),
    )
    specs = (
        (DataSplit.TRAIN, config.shortcut_train_scenes),
        (DataSplit.VALIDATION, config.shortcut_validation_scenes),
        (DataSplit.TEST, config.shortcut_test_scenes),
    )
    pool: list[tuple[Scene, DataSplit]] = []
    index = 0
    for split, count in specs:
        scenes, index = _collect_stratified(
            sampler,
            prefix="shortcut",
            split=split,
            count=count,
            start_index=index,
            accept=lambda scene: True,
            max_scan_count=config.max_scan_count,
        )
        pool.extend(scenes)

    base_records = tuple(
        _base_record(
            _shape_task(scene),
            split=split,
            variant=(
                DatasetVariant.COUNTERFACTUAL
                if split is DataSplit.TEST
                else DatasetVariant.SHORTCUT_IID
            ),
        )
        for scene, split in pool
    )
    iid_train = tuple(
        record for record in base_records if record.split is DataSplit.TRAIN
    )
    iid_validation = tuple(
        record for record in base_records if record.split is DataSplit.VALIDATION
    )
    counterfactual = tuple(
        record for record in base_records if record.split is DataSplit.TEST
    )
    # Control the correlation independently in each split. If train and
    # validation were assigned together, the global ratio could hide an
    # unintended per-split deviation.
    iid_with_cues = _assign_iid_cues(
        iid_train, strength=config.shortcut_strength, seed=config.base_seed + 11
    ) + _assign_iid_cues(
        iid_validation,
        strength=config.shortcut_strength,
        seed=config.base_seed + 12,
    )
    counterfactual_with_cues = _assign_counterfactual_cues(counterfactual)
    by_id = {
        record.record_id: record
        for record in iid_with_cues + counterfactual_with_cues
    }
    ordered_records = tuple(by_id[record.record_id] for record in base_records)
    return tuple(pool), ordered_records


def _composition_key(scene: Scene) -> str:
    obj = scene.objects[0]
    return f"{obj.color.value}|{obj.shape.value}"


def _compositional_pool(
    config: DatasetBuildConfig,
) -> tuple[tuple[tuple[Scene, DataSplit], ...], tuple[DatasetRecord, ...]]:
    sampler = SceneSampler(
        config.base_seed + 2,
        SamplerConfig(min_objects=1, max_objects=1),
    )
    holdouts = {f"{color}|{shape}" for color, shape in config.holdout_pairs}
    specs = (
        (DataSplit.TRAIN, config.compositional_train_scenes, False),
        (DataSplit.VALIDATION, config.compositional_validation_scenes, False),
        (DataSplit.TEST, config.compositional_test_scenes, True),
    )
    pool: list[tuple[Scene, DataSplit]] = []
    index = 0
    for split, count, require_holdout in specs:
        scenes, index = _collect_stratified(
            sampler,
            prefix="composition",
            split=split,
            count=count,
            start_index=index,
            accept=(
                (lambda scene: _composition_key(scene) in holdouts)
                if require_holdout
                else (lambda scene: _composition_key(scene) not in holdouts)
            ),
            max_scan_count=config.max_scan_count,
        )
        pool.extend(scenes)

    records = tuple(
        _base_record(
            _shape_task(scene),
            split=split,
            variant=DatasetVariant.COMPOSITIONAL,
            composition_key=_composition_key(scene),
        )
        for scene, split in pool
    )
    return tuple(pool), records


def build_dataset_bundle(config: DatasetBuildConfig | None = None) -> DatasetBundle:
    cfg = config or DatasetBuildConfig()
    core_pool = _core_pool(cfg)
    shortcut_pool, shortcut_records = _shortcut_pool(cfg)
    compositional_pool, compositional_records = _compositional_pool(cfg)
    return DatasetBundle(
        config=cfg,
        core_scenes=tuple(scene for scene, _ in core_pool),
        core_records=_core_records(core_pool),
        shortcut_scenes=tuple(scene for scene, _ in shortcut_pool),
        shortcut_records=shortcut_records,
        compositional_scenes=tuple(scene for scene, _ in compositional_pool),
        compositional_records=compositional_records,
    )
