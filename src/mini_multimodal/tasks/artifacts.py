"""Persistent and verifiable artifacts for structured text tasks."""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping, Any

from mini_multimodal.world.schema import Scene

from .generation import build_generation_conditions, build_qa_tasks
from .schema import QATask, RenderedGenerationCondition
from .serialization import (
    condition_from_dict,
    condition_to_dict,
    qa_task_from_dict,
    qa_task_to_dict,
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


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def task_statistics(
    tasks: Iterable[QATask],
    conditions: Iterable[RenderedGenerationCondition],
) -> dict[str, object]:
    task_list = tuple(tasks)
    condition_list = tuple(conditions)
    question_kinds = Counter(task.question.kind.value for task in task_list)
    target_modalities = Counter(
        condition.condition.target_modality.value for condition in condition_list
    )
    answer_values = Counter(str(task.answer.value) for task in task_list)
    scene_ids = {
        task.question.scene_id for task in task_list
    } | {condition.condition.scene_id for condition in condition_list}
    return {
        "scene_count": len(scene_ids),
        "qa_task_count": len(task_list),
        "generation_condition_count": len(condition_list),
        "question_kinds": dict(sorted(question_kinds.items())),
        "target_modalities": dict(sorted(target_modalities.items())),
        "answer_values": dict(sorted(answer_values.items())),
    }


def export_task_fixture(
    output_dir: str | Path,
    scenes: Iterable[Scene],
) -> dict[str, object]:
    destination = Path(output_dir)
    scene_list = tuple(scenes)
    if not scene_list:
        raise ValueError("At least one scene is required to export a task fixture")
    scene_ids = [scene.scene_id for scene in scene_list]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Task fixture scene ids must be unique")

    tasks = tuple(task for scene in scene_list for task in build_qa_tasks(scene))
    conditions = tuple(
        condition
        for scene in scene_list
        for condition in build_generation_conditions(scene)
    )
    edge_scene_id = scene_list[-1].scene_id
    edge_tasks = tuple(
        task for task in tasks if task.question.scene_id == edge_scene_id
    )
    edge_conditions = tuple(
        condition
        for condition in conditions
        if condition.condition.scene_id == edge_scene_id
    )
    qa_path = destination / "qa_tasks.jsonl"
    condition_path = destination / "generation_conditions.jsonl"
    _write_jsonl(qa_path, (qa_task_to_dict(task) for task in tasks))
    _write_jsonl(
        condition_path, (condition_to_dict(condition) for condition in conditions)
    )

    preview_path = destination / "sample_text_preview.md"
    preview_lines = [
        "# P1-W02中文任务样例",
        "",
        f"场景：`{scene_list[0].scene_id}`",
        "",
        "## 问答样例",
        "",
    ]
    for task in tasks[:8]:
        preview_lines.extend(
            (
                f"- 问：{task.question_zh}",
                f"  答：{task.answer_zh}（结构化值：`{task.answer.value}`）",
            )
        )
    preview_lines.extend(("", "## 生成条件样例", ""))
    for condition in conditions[:3]:
        preview_lines.append(
            f"- `{condition.condition.target_modality.value}`：{condition.text_zh}"
        )
    preview_lines.extend(("", "## 时间边界场景", "", f"场景：`{edge_scene_id}`", ""))
    for task in edge_tasks:
        if task.question.kind.value == "event_order":
            preview_lines.extend(
                (
                    f"- 问：{task.question_zh}",
                    f"  答：{task.answer_zh}（证据帧：`{task.answer.evidence_frames}`）",
                )
            )
    for condition in edge_conditions:
        preview_lines.append(
            f"- `{condition.condition.target_modality.value}`：{condition.text_zh}"
        )
    preview_path.write_text(
        "\n".join(preview_lines) + "\n", encoding="utf-8", newline="\n"
    )

    statistics = task_statistics(tasks, conditions)
    manifest = {
        "artifact_version": 1,
        "task_schema_version": 1,
        "language": "zh-CN",
        "template_version": 1,
        "scene_ids": scene_ids,
        "qa_file": qa_path.name,
        "qa_bytes": qa_path.stat().st_size,
        "qa_sha256": _sha256(qa_path),
        "condition_file": condition_path.name,
        "condition_bytes": condition_path.stat().st_size,
        "condition_sha256": _sha256(condition_path),
        "preview_file": preview_path.name,
        "preview_bytes": preview_path.stat().st_size,
        "preview_sha256": _sha256(preview_path),
        "statistics": statistics,
    }
    _write_json(destination / "manifest.json", manifest)
    _write_json(
        destination / "sample_records.json",
        {
            "qa_tasks": [qa_task_to_dict(task) for task in tasks[:3]],
            "generation_conditions": [
                condition_to_dict(condition) for condition in conditions[:3]
            ],
            "edge_scene_id": edge_scene_id,
            "edge_qa_tasks": [qa_task_to_dict(task) for task in edge_tasks],
            "edge_generation_conditions": [
                condition_to_dict(condition) for condition in edge_conditions
            ],
        },
    )
    return manifest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_task_fixture(
    output_dir: str | Path,
) -> tuple[tuple[QATask, ...], tuple[RenderedGenerationCondition, ...]]:
    source = Path(output_dir)
    tasks = tuple(qa_task_from_dict(item) for item in _load_jsonl(source / "qa_tasks.jsonl"))
    conditions = tuple(
        condition_from_dict(item)
        for item in _load_jsonl(source / "generation_conditions.jsonl")
    )
    return tasks, conditions


def verify_task_fixture(output_dir: str | Path) -> dict[str, object]:
    source = Path(output_dir)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    qa_path = source / manifest["qa_file"]
    condition_path = source / manifest["condition_file"]
    preview_path = source / manifest["preview_file"]
    actual_qa_sha = _sha256(qa_path)
    actual_condition_sha = _sha256(condition_path)
    if actual_qa_sha != manifest["qa_sha256"]:
        raise ValueError("QA task fixture SHA-256 does not match its manifest")
    if actual_condition_sha != manifest["condition_sha256"]:
        raise ValueError("Generation-condition fixture SHA-256 does not match its manifest")
    actual_preview_sha = _sha256(preview_path)
    if actual_preview_sha != manifest["preview_sha256"]:
        raise ValueError("Text preview SHA-256 does not match its manifest")

    tasks, conditions = load_task_fixture(source)
    statistics = task_statistics(tasks, conditions)
    if statistics != manifest["statistics"]:
        raise ValueError("Task fixture statistics do not match its manifest")
    return {
        "verified": True,
        "qa_sha256": actual_qa_sha,
        "condition_sha256": actual_condition_sha,
        "preview_sha256": actual_preview_sha,
        "statistics": statistics,
    }


def benchmark_task_generation(
    scenes: Iterable[Scene],
    *,
    repetitions: int = 1,
) -> dict[str, float | int]:
    scene_list = tuple(scenes)
    if not scene_list or repetitions <= 0:
        raise ValueError("Scenes must be non-empty and repetitions must be positive")
    started = perf_counter()
    task_count = 0
    condition_count = 0
    for _ in range(repetitions):
        for scene in scene_list:
            task_count += len(build_qa_tasks(scene))
            condition_count += len(build_generation_conditions(scene))
    elapsed = perf_counter() - started
    return {
        "scene_pass_count": len(scene_list) * repetitions,
        "qa_task_count": task_count,
        "generation_condition_count": condition_count,
        "elapsed_seconds": elapsed,
        "scenes_per_second": len(scene_list) * repetitions / elapsed,
    }
