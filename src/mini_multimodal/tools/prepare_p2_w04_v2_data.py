"""Build the leakage-safe P2-W04 v2 training view."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.ldm.data import prepare_p2_w04_data


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "configs" / "p2_w04_tiny_image_ldm_v2.json").read_text(
            encoding="utf-8"
        )
    )
    report = prepare_p2_w04_data(root, config)
    print(
        json.dumps(
            {
                "verified": report["verified"],
                "extension_count": report["extension_count"],
                "excluded_original_train_count": report["excluded_original_train_count"],
                "combined_split_counts": report["combined_split_counts"],
                "training_holdout_object_count_zero": report["checks"][
                    "training_holdout_object_count_zero"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
