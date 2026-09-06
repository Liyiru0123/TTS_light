"""Run all P2-W02 VAE learning and evidence gates."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.vae.artifacts import run_p2_w02_experiments


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "configs" / "p2_w02_image_vae.json").read_text(encoding="utf-8")
    )
    summary = run_p2_w02_experiments(root, config)
    print(json.dumps({"verified": summary["verified"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

