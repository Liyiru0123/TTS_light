"""Leakage, shortcut, composition, and balance diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isclose
from typing import Iterable

from mini_multimodal.world.schema import Color, Shape

from .generation import CUE_BY_SHAPE, SHAPE_ORDER
from .schema import DataSplit, DatasetBundle, DatasetRecord, DatasetVariant


def _records_by_split(
    records: Iterable[DatasetRecord],
) -> dict[DataSplit, tuple[DatasetRecord, ...]]:
    grouped: dict[DataSplit, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        grouped[record.split].append(record)
    return {split: tuple(grouped.get(split, ())) for split in DataSplit}


def _scene_sets(records: Iterable[DatasetRecord]) -> dict[DataSplit, set[str]]:
    grouped = _records_by_split(records)
    return {split: {record.scene_id for record in group} for split, group in grouped.items()}


def leakage_report(bundle: DatasetBundle) -> dict[str, object]:
    pools = {
        "core": bundle.core_records,
        "shortcut": bundle.shortcut_records,
        "compositional": bundle.compositional_records,
    }
    pool_reports: dict[str, object] = {}
    for name, records in pools.items():
        scenes = _scene_sets(records)
        questions = {
            split: {record.task.question.question_id for record in group}
            for split, group in _records_by_split(records).items()
        }
        pool_reports[name] = {
            "scene_intersections": {
                "train_validation": len(scenes[DataSplit.TRAIN] & scenes[DataSplit.VALIDATION]),
                "train_test": len(scenes[DataSplit.TRAIN] & scenes[DataSplit.TEST]),
                "validation_test": len(scenes[DataSplit.VALIDATION] & scenes[DataSplit.TEST]),
            },
            "question_intersections": {
                "train_validation": len(
                    questions[DataSplit.TRAIN] & questions[DataSplit.VALIDATION]
                ),
                "train_test": len(questions[DataSplit.TRAIN] & questions[DataSplit.TEST]),
                "validation_test": len(
                    questions[DataSplit.VALIDATION] & questions[DataSplit.TEST]
                ),
            },
        }

    all_scene_ids = [scene.scene_id for scene in bundle.all_scenes]
    all_record_ids = [record.record_id for record in bundle.all_records]
    known_scene_ids = set(all_scene_ids)
    missing_references = sorted(
        {record.scene_id for record in bundle.all_records if record.scene_id not in known_scene_ids}
    )
    return {
        "pools": pool_reports,
        "duplicate_scene_id_count": len(all_scene_ids) - len(set(all_scene_ids)),
        "duplicate_record_id_count": len(all_record_ids) - len(set(all_record_ids)),
        "missing_scene_references": missing_references,
    }


def _accuracy(records: Iterable[DatasetRecord], mapping: dict[str, str]) -> float:
    items = tuple(records)
    if not items:
        return 0.0
    correct = sum(mapping.get(record.cue_token or "") == str(record.task.answer.value) for record in items)
    return correct / len(items)


def _majority_accuracy(records: Iterable[DatasetRecord]) -> float:
    items = tuple(records)
    if not items:
        return 0.0
    counts = Counter(str(record.task.answer.value) for record in items)
    return max(counts.values()) / len(items)


def shortcut_diagnostics(bundle: DatasetBundle) -> dict[str, object]:
    grouped = _records_by_split(bundle.shortcut_records)
    train = grouped[DataSplit.TRAIN]
    cue_answer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in train:
        assert record.cue_token is not None
        cue_answer_counts[record.cue_token][str(record.task.answer.value)] += 1
    learned_mapping = {
        cue: counts.most_common(1)[0][0] for cue, counts in sorted(cue_answer_counts.items())
    }

    splits: dict[str, object] = {}
    for split, records in grouped.items():
        answer_counts = Counter(str(record.task.answer.value) for record in records)
        aligned = sum(record.cue_aligned is True for record in records)
        splits[split.value] = {
            "record_count": len(records),
            "answer_counts": dict(sorted(answer_counts.items())),
            "aligned_count": aligned,
            "aligned_rate": aligned / len(records),
            "cue_only_accuracy": _accuracy(records, learned_mapping),
            "majority_accuracy": _majority_accuracy(records),
        }
    return {
        "shape_to_cue": dict(CUE_BY_SHAPE),
        "learned_cue_to_shape": learned_mapping,
        "splits": splits,
    }


def compositional_report(bundle: DatasetBundle) -> dict[str, object]:
    grouped = _records_by_split(bundle.compositional_records)
    holdouts = {f"{color}|{shape}" for color, shape in bundle.config.holdout_pairs}
    keys = {
        split: {record.composition_key for record in records}
        for split, records in grouped.items()
    }
    train_objects = {
        scene.scene_id: scene.objects[0]
        for scene in bundle.compositional_scenes
        if scene.scene_id in {record.scene_id for record in grouped[DataSplit.TRAIN]}
    }
    train_colors = {obj.color.value for obj in train_objects.values()}
    train_shapes = {obj.shape.value for obj in train_objects.values()}
    split_details: dict[str, object] = {}
    for split, records in grouped.items():
        counts = Counter(str(record.task.answer.value) for record in records)
        split_details[split.value] = {
            "record_count": len(records),
            "shape_counts": dict(sorted(counts.items())),
            "composition_count": len(keys[split]),
        }
    return {
        "holdout_pairs": sorted(holdouts),
        "train_holdout_leak_count": len(keys[DataSplit.TRAIN] & holdouts),
        "validation_holdout_leak_count": len(keys[DataSplit.VALIDATION] & holdouts),
        "test_non_holdout_count": len(keys[DataSplit.TEST] - holdouts),
        "test_holdout_coverage": sorted(keys[DataSplit.TEST] & holdouts),
        "train_colors": sorted(train_colors),
        "train_shapes": sorted(train_shapes),
        "all_test_colors_seen_in_train": {
            key.split("|", 1)[0] for key in keys[DataSplit.TEST]
        }.issubset(train_colors),
        "all_test_shapes_seen_in_train": {
            key.split("|", 1)[1] for key in keys[DataSplit.TEST]
        }.issubset(train_shapes),
        "splits": split_details,
    }


def dataset_statistics(bundle: DatasetBundle) -> dict[str, object]:
    pools = {
        "core": (bundle.core_scenes, bundle.core_records),
        "shortcut": (bundle.shortcut_scenes, bundle.shortcut_records),
        "compositional": (
            bundle.compositional_scenes,
            bundle.compositional_records,
        ),
    }
    result: dict[str, object] = {}
    for name, (scenes, records) in pools.items():
        grouped = _records_by_split(records)
        result[name] = {
            "scene_count": len(scenes),
            "record_count": len(records),
            "splits": {
                split.value: {
                    "scene_count": len({record.scene_id for record in group}),
                    "record_count": len(group),
                }
                for split, group in grouped.items()
            },
        }
    return result


def validate_dataset_bundle(bundle: DatasetBundle) -> dict[str, object]:
    if any(record.cue_token == "__pending__" for record in bundle.all_records):
        raise ValueError("Dataset contains an unassigned shortcut cue")
    leakage = leakage_report(bundle)
    if leakage["duplicate_scene_id_count"] or leakage["duplicate_record_id_count"]:
        raise ValueError("Dataset contains duplicate scene or record ids")
    if leakage["missing_scene_references"]:
        raise ValueError("Dataset records reference missing scenes")
    for report in leakage["pools"].values():
        if any(report["scene_intersections"].values()):
            raise ValueError("A scene occurs in more than one split")
        if any(report["question_intersections"].values()):
            raise ValueError("A question occurs in more than one split")

    stats = dataset_statistics(bundle)
    cfg = bundle.config
    expected = {
        "core": {
            DataSplit.TRAIN.value: cfg.core_train_scenes,
            DataSplit.VALIDATION.value: cfg.core_validation_scenes,
            DataSplit.TEST.value: cfg.core_test_scenes,
        },
        "shortcut": {
            DataSplit.TRAIN.value: cfg.shortcut_train_scenes,
            DataSplit.VALIDATION.value: cfg.shortcut_validation_scenes,
            DataSplit.TEST.value: cfg.shortcut_test_scenes,
        },
        "compositional": {
            DataSplit.TRAIN.value: cfg.compositional_train_scenes,
            DataSplit.VALIDATION.value: cfg.compositional_validation_scenes,
            DataSplit.TEST.value: cfg.compositional_test_scenes,
        },
    }
    for pool_name, split_counts in expected.items():
        for split, count in split_counts.items():
            if stats[pool_name]["splits"][split]["scene_count"] != count:
                raise ValueError(f"Unexpected {pool_name} {split} scene count")

    shortcut = shortcut_diagnostics(bundle)
    for split in DataSplit:
        shape_counts = shortcut["splits"][split.value]["answer_counts"]
        if set(shape_counts) != set(SHAPE_ORDER) or len(set(shape_counts.values())) != 1:
            raise ValueError(f"Shortcut {split.value} labels are not exactly balanced")
    if not isclose(
        shortcut["splits"]["train"]["aligned_rate"],
        cfg.shortcut_strength,
        abs_tol=1e-12,
    ):
        raise ValueError("Shortcut training alignment does not match configured strength")
    if not 0.88 <= shortcut["splits"]["validation"]["aligned_rate"] <= 0.95:
        raise ValueError("Shortcut validation alignment is outside the accepted range")
    if shortcut["splits"]["test"]["aligned_rate"] != 0.0:
        raise ValueError("Counterfactual test contains aligned shortcut cues")
    if shortcut["splits"]["validation"]["cue_only_accuracy"] < 0.85:
        raise ValueError("Cue-only IID validation accuracy is unexpectedly low")
    if shortcut["splits"]["test"]["cue_only_accuracy"] > 0.05:
        raise ValueError("Cue-only counterfactual accuracy is unexpectedly high")

    composition = compositional_report(bundle)
    if composition["train_holdout_leak_count"]:
        raise ValueError("A held-out composition leaked into training")
    if composition["validation_holdout_leak_count"]:
        raise ValueError("A held-out composition leaked into validation")
    if composition["test_non_holdout_count"]:
        raise ValueError("Compositional test contains a non-held-out pair")
    if len(composition["test_holdout_coverage"]) != len(cfg.holdout_pairs):
        raise ValueError("Compositional test does not cover every held-out pair")
    if not composition["all_test_colors_seen_in_train"]:
        raise ValueError("A compositional-test color was never seen in training")
    if not composition["all_test_shapes_seen_in_train"]:
        raise ValueError("A compositional-test shape was never seen in training")
    for split in DataSplit:
        shape_counts = composition["splits"][split.value]["shape_counts"]
        if set(shape_counts) != set(SHAPE_ORDER) or len(set(shape_counts.values())) != 1:
            raise ValueError(f"Compositional {split.value} labels are not exactly balanced")

    return {
        "verified": True,
        "leakage": leakage,
        "shortcut": shortcut,
        "compositional": composition,
        "statistics": stats,
    }
