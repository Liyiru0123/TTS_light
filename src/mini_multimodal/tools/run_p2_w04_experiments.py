"""Run all frozen P2-W04 learning, generation, ablation, and resource checks."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.ldm.artifacts import run_experiments


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
    summary = run_experiments(root, config)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
