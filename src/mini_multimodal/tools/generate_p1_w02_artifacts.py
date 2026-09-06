"""Generate persistent data and results for work package P1-W02."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.tasks.artifacts import (
    benchmark_task_generation,
    export_task_fixture,
    verify_task_fixture,
)
from mini_multimodal.tasks.examples import make_temporal_example_scene
from mini_multimodal.world.artifacts import load_scene_fixture


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    scene_path = project_root / "data" / "fixtures" / "world_core_v1" / "scenes.jsonl"
    data_dir = project_root / "data" / "fixtures" / "task_text_v1"
    results_dir = project_root / "results" / "P1-W02_task_text"
    scenes = tuple(load_scene_fixture(scene_path)) + (make_temporal_example_scene(),)

    manifest = export_task_fixture(data_dir, scenes)
    verification = verify_task_fixture(data_dir)
    benchmark = benchmark_task_generation(scenes, repetitions=5)
    _write_json(results_dir / "fixture_verification.json", verification)
    _write_json(results_dir / "task_statistics.json", manifest["statistics"])
    _write_json(results_dir / "generation_benchmark.json", benchmark)

    print(
        json.dumps(
            {
                "scene_count": manifest["statistics"]["scene_count"],
                "qa_task_count": manifest["statistics"]["qa_task_count"],
                "generation_condition_count": manifest["statistics"][
                    "generation_condition_count"
                ],
                "verified": verification["verified"],
                "data_dir": str(data_dir),
                "results_dir": str(results_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
