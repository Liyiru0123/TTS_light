"""Generate the fixed P1-W07 index and machine-readable integration evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from mini_multimodal.multimodal import (
    UnifiedMultimodalDataset,
    collate_multimodal,
    export_unified_artifacts,
    verify_unified_artifacts,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config_path = root / "configs" / "p1_w07_unified.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index_dir = root / config["output"]
    result_dir = root / "results" / "P1-W07_multimodal_integration"

    export_started = perf_counter()
    manifest = export_unified_artifacts(root, config_path)
    export_seconds = perf_counter() - export_started
    verification_started = perf_counter()
    verification = verify_unified_artifacts(index_dir)
    verification_seconds = perf_counter() - verification_started

    init_started = perf_counter()
    dataset = UnifiedMultimodalDataset(
        index_dir,
        view="scene",
        cache_max_bytes=int(config["cache_max_bytes"]),
    )
    initialization_seconds = perf_counter() - init_started
    first_started = perf_counter()
    first = dataset[0]
    first_load_seconds = perf_counter() - first_started
    cached_started = perf_counter()
    dataset[0]
    cached_load_seconds = perf_counter() - cached_started
    second = dataset[1]
    batch_started = perf_counter()
    batch = collate_multimodal([first, second])
    batch_seconds = perf_counter() - batch_started

    loader_benchmark = {
        "initialization_seconds": initialization_seconds,
        "first_full_sample_seconds": first_load_seconds,
        "cached_full_sample_seconds": cached_load_seconds,
        "two_sample_collate_seconds": batch_seconds,
        "batch_shapes": {
            key: list(value.shape)
            for key, value in batch.items()
            if hasattr(value, "shape")
        },
        "export_seconds": export_seconds,
        "verification_seconds": verification_seconds,
    }
    cache_report = asdict(dataset.cache_info())
    _write_json(result_dir / "integration_statistics.json", manifest["statistics"])
    _write_json(result_dir / "semantic_consistency.json", manifest["consistency"])
    _write_json(result_dir / "asset_verification.json", verification["asset_verification"])
    _write_json(result_dir / "loader_benchmark.json", loader_benchmark)
    _write_json(result_dir / "cache_report.json", cache_report)
    print(
        json.dumps(
            {
                "verified": verification["verified"],
                "scene_count": manifest["statistics"]["scene_count"],
                "qa_record_count": manifest["statistics"]["qa_record_count"],
                "cache_hits": cache_report["hits"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
