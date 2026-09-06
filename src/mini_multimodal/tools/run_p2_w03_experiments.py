"""Run the complete P2-W03 TinyVLM evidence pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.vlm.artifacts import run_p2_w03_experiments


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w03_tiny_vlm.json").read_text(encoding="utf-8"))
    summary = run_p2_w03_experiments(root, config)
    print(json.dumps({"verified": summary["verified"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
