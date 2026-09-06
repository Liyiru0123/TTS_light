"""Build the fixed P2-W04 v2 evaluation fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.ldm.evaluation_data import prepare_fixed_evaluation_data


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "configs" / "p2_w04_tiny_image_ldm_v2.json").read_text(
            encoding="utf-8"
        )
    )
    report = prepare_fixed_evaluation_data(root, config)
    print(json.dumps({
        "verified": report["verified"],
        "record_count": report["record_count"],
        "subset_counts": report["subset_counts"],
        "intervention_count": report["intervention_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
