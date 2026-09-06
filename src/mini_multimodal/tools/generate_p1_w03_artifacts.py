"""Generate persistent datasets and diagnostics for work package P1-W03."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from mini_multimodal.data.artifacts import export_dataset_bundle, verify_dataset_bundle
from mini_multimodal.data.generation import build_dataset_bundle


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "datasets" / "p1_w03_v1"
    results_dir = project_root / "results" / "P1-W03_data_splits"

    build_started = perf_counter()
    bundle = build_dataset_bundle()
    build_seconds = perf_counter() - build_started
    export_started = perf_counter()
    manifest = export_dataset_bundle(data_dir, bundle)
    export_seconds = perf_counter() - export_started
    verification = verify_dataset_bundle(data_dir)

    _write_json(results_dir / "leakage_report.json", verification["leakage"])
    _write_json(
        results_dir / "shortcut_diagnostics.json", verification["shortcut"]
    )
    _write_json(
        results_dir / "compositional_report.json", verification["compositional"]
    )
    _write_json(results_dir / "dataset_statistics.json", verification["statistics"])
    _write_json(
        results_dir / "generation_benchmark.json",
        {
            "build_seconds": build_seconds,
            "export_seconds": export_seconds,
            "total_scene_count": len(bundle.all_scenes),
            "total_record_count": len(bundle.all_records),
        },
    )
    _write_json(
        results_dir / "fixture_verification.json",
        {"verified": True, "files": verification["files"]},
    )

    print(
        json.dumps(
            {
                "verified": verification["verified"],
                "total_scene_count": len(bundle.all_scenes),
                "total_record_count": len(bundle.all_records),
                "core_record_count": manifest["statistics"]["core"]["record_count"],
                "shortcut_validation_cue_accuracy": verification["shortcut"]["splits"][
                    "validation"
                ]["cue_only_accuracy"],
                "counterfactual_cue_accuracy": verification["shortcut"]["splits"][
                    "test"
                ]["cue_only_accuracy"],
                "data_dir": str(data_dir),
                "results_dir": str(results_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
