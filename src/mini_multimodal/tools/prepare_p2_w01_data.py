"""Create the fixed task-routing and baseline manifests for P2-W01."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.training.data import export_p2_w01_data


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "configs" / "p2_w01_training.json").read_text(encoding="utf-8")
    )
    manifest = export_p2_w01_data(
        root / config["source_index"],
        root / config["output_data"],
        overfit_examples_per_class=config["baseline"]["overfit_examples_per_class"],
    )
    results = root / config["output_results"]
    results.mkdir(parents=True, exist_ok=True)
    routing = json.loads(
        (root / config["output_data"] / "task_modality_routes.json").read_text(
            encoding="utf-8"
        )
    )
    (results / "task_routing_report.json").write_text(
        json.dumps(routing, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest["routing_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

