"""JSON-compatible serialization for dataset records and configurations."""

from __future__ import annotations

from typing import Any, Mapping

from mini_multimodal.tasks.serialization import qa_task_from_dict, qa_task_to_dict

from .schema import (
    DataSplit,
    DatasetBuildConfig,
    DatasetRecord,
    DatasetVariant,
)


def config_to_dict(config: DatasetBuildConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "base_seed": config.base_seed,
        "core_train_scenes": config.core_train_scenes,
        "core_validation_scenes": config.core_validation_scenes,
        "core_test_scenes": config.core_test_scenes,
        "shortcut_train_scenes": config.shortcut_train_scenes,
        "shortcut_validation_scenes": config.shortcut_validation_scenes,
        "shortcut_test_scenes": config.shortcut_test_scenes,
        "compositional_train_scenes": config.compositional_train_scenes,
        "compositional_validation_scenes": config.compositional_validation_scenes,
        "compositional_test_scenes": config.compositional_test_scenes,
        "shortcut_strength": config.shortcut_strength,
        "holdout_pairs": [list(pair) for pair in config.holdout_pairs],
        "max_scan_count": config.max_scan_count,
    }


def config_from_dict(data: Mapping[str, Any]) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        schema_version=int(data["schema_version"]),
        base_seed=int(data["base_seed"]),
        core_train_scenes=int(data["core_train_scenes"]),
        core_validation_scenes=int(data["core_validation_scenes"]),
        core_test_scenes=int(data["core_test_scenes"]),
        shortcut_train_scenes=int(data["shortcut_train_scenes"]),
        shortcut_validation_scenes=int(data["shortcut_validation_scenes"]),
        shortcut_test_scenes=int(data["shortcut_test_scenes"]),
        compositional_train_scenes=int(data["compositional_train_scenes"]),
        compositional_validation_scenes=int(data["compositional_validation_scenes"]),
        compositional_test_scenes=int(data["compositional_test_scenes"]),
        shortcut_strength=float(data["shortcut_strength"]),
        holdout_pairs=tuple(
            (str(pair[0]), str(pair[1])) for pair in data["holdout_pairs"]
        ),
        max_scan_count=int(data["max_scan_count"]),
    )


def dataset_record_to_dict(record: DatasetRecord) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "record_id": record.record_id,
        "split": record.split.value,
        "variant": record.variant.value,
        "task": qa_task_to_dict(record.task),
        "cue_token": record.cue_token,
        "cue_aligned": record.cue_aligned,
        "composition_key": record.composition_key,
    }


def dataset_record_from_dict(data: Mapping[str, Any]) -> DatasetRecord:
    return DatasetRecord(
        schema_version=int(data["schema_version"]),
        record_id=str(data["record_id"]),
        split=DataSplit(data["split"]),
        variant=DatasetVariant(data["variant"]),
        task=qa_task_from_dict(data["task"]),
        cue_token=None if data.get("cue_token") is None else str(data["cue_token"]),
        cue_aligned=(
            None if data.get("cue_aligned") is None else bool(data["cue_aligned"])
        ),
        composition_key=(
            None
            if data.get("composition_key") is None
            else str(data["composition_key"])
        ),
    )
