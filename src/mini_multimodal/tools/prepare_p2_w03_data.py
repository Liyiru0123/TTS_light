"""Build and verify the fixed grounded VQA view for P2-W03."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.vlm import export_p2_w03_data, verify_p2_w03_data


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w03_tiny_vlm.json").read_text(encoding="utf-8"))
    manifest = export_p2_w03_data(
        root,
        config["output_data"],
        source_index=config["source_index"],
        source_latents=config["source_latents"],
        overfit_count=int(config["training"]["overfit_count"]),
    )
    report = verify_p2_w03_data(root, config["output_data"])
    print(json.dumps({"verified": report["verified"], "statistics": manifest["statistics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
