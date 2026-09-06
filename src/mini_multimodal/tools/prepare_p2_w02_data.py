"""Build fixed full-scene VAE data manifests and small-target statistics."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.vae import export_p2_w02_data


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "configs" / "p2_w02_image_vae.json").read_text(encoding="utf-8")
    )
    manifest = export_p2_w02_data(
        root,
        root / config["output_data"],
        source_index=config["source_index"],
        source_images=config["source_images"],
        overfit_count=config["training"]["overfit_count"],
    )
    results = root / config["output_results"]
    results.mkdir(parents=True, exist_ok=True)
    (results / "data_readiness_report.json").write_text(
        json.dumps(manifest["statistics"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest["statistics"]["split_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()

