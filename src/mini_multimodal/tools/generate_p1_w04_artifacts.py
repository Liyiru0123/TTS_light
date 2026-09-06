"""Generate and verify persistent image artifacts for work package P1-W04."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from mini_multimodal.data.artifacts import load_dataset_bundle, verify_dataset_bundle
from mini_multimodal.vision.artifacts import (
    export_image_artifacts,
    verify_image_artifacts,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    source_data_dir = project_root / "data" / "datasets" / "p1_w03_v1"
    image_dir = project_root / "data" / "images" / "p1_w04_v1"
    results_dir = project_root / "results" / "P1-W04_image_renderer"

    source_verification = verify_dataset_bundle(source_data_dir)
    source_manifest_path = source_data_dir / "manifest.json"
    source_digest = sha256(source_manifest_path.read_bytes()).hexdigest()
    bundle = load_dataset_bundle(source_data_dir)

    export_started = perf_counter()
    manifest = export_image_artifacts(
        image_dir,
        bundle,
        source_dataset_manifest_sha256=source_digest,
    )
    export_seconds = perf_counter() - export_started
    verify_started = perf_counter()
    verification = verify_image_artifacts(image_dir, bundle)
    verification_seconds = perf_counter() - verify_started

    _write_json(results_dir / "image_statistics.json", verification["statistics"])
    _write_json(
        results_dir / "semantic_consistency.json", verification["consistency"]
    )
    _write_json(
        results_dir / "shortcut_visual_diagnostics.json", verification["shortcut"]
    )
    _write_json(
        results_dir / "fixture_verification.json",
        {
            "verified": verification["verified"],
            "image_count": verification["image_count"],
            "annotation_count": verification["annotation_count"],
            "managed_file_count": verification["managed_file_count"],
            "source_dataset_verified": source_verification["verified"],
            "source_dataset_manifest_sha256": source_digest,
            "image_manifest_sha256": sha256(
                (image_dir / "manifest.json").read_bytes()
            ).hexdigest(),
        },
    )
    total_bytes = sum(entry["bytes"] for entry in manifest["files"].values())
    _write_json(
        results_dir / "render_benchmark.json",
        {
            "export_seconds": export_seconds,
            "verification_seconds": verification_seconds,
            "image_count": verification["image_count"],
            "images_per_export_second": verification["image_count"] / export_seconds,
            "managed_file_count": verification["managed_file_count"],
            "managed_bytes": total_bytes,
        },
    )

    print(
        json.dumps(
            {
                "verified": verification["verified"],
                "image_count": verification["image_count"],
                "object_count": verification["statistics"]["object_count"],
                "relation_order_violations": verification["consistency"][
                    "relation_order_violation_count"
                ],
                "fully_occluded_objects": verification["consistency"][
                    "fully_occluded_object_count"
                ],
                "max_quantization_adjustment_pixels": verification["consistency"][
                    "max_quantization_adjustment_pixels"
                ],
                "image_dir": str(image_dir),
                "results_dir": str(results_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
