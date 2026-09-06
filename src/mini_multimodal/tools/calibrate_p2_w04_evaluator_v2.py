"""Calibrate the P2-W04 v2 evaluator on source and VAE images."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.ldm.evaluation_artifacts import calibrate_evaluator


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "configs" / "p2_w04_tiny_image_ldm_v2.json").read_text(
            encoding="utf-8"
        )
    )
    report = calibrate_evaluator(root, config)
    print(json.dumps({
        "verified": report["verified"],
        "source_verified": report["source"]["verified"],
        "vae_verified": report["vae"]["verified"],
        "source_shape_accuracy": report["source"]["metrics"]["shape_accuracy"]["value"],
        "vae_shape_accuracy": report["vae"]["metrics"]["shape_accuracy"]["value"],
        "vae_position_accuracy": report["vae"]["metrics"]["position_accuracy"]["value"],
    }, ensure_ascii=False))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
