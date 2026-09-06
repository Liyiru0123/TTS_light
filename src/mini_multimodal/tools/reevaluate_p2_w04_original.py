"""Re-evaluate the historical checkpoint and freeze v2 thresholds."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.ldm.evaluation_artifacts import (
    reevaluate_original_checkpoint_and_freeze_thresholds,
)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "configs" / "p2_w04_tiny_image_ldm_v2.json").read_text(
            encoding="utf-8"
        )
    )
    result = reevaluate_original_checkpoint_and_freeze_thresholds(root, config)
    metrics = result["original_checkpoint"]["metrics"]
    print(json.dumps({
        "sample_count": result["original_checkpoint"]["sample_count"],
        "seen_fully_correct": metrics["seen_single"]["fully_correct_rate"]["value"],
        "unseen_labeled_fully_correct": metrics["unseen_single"]["fully_correct_rate"]["value"],
        "thresholds_verified": result["threshold_freeze"]["verified"],
        "composition_generalization_valid": result["original_checkpoint"]["composition_generalization_valid"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
