"""Build and persist the P2-W04 train-only image/text expansion."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.ldm.data import prepare_p2_w04_data


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
    report = prepare_p2_w04_data(root, config)
    print(json.dumps({
        "verified": report["verified"],
        "extension_count": report["extension_count"],
        "combined_split_counts": report["combined_split_counts"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
