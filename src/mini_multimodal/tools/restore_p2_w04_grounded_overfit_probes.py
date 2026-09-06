from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.ldm.data import LatentTextDataset
from mini_multimodal.ldm.grounded_training import _semantic_probes, grounded_model_from_config
from mini_multimodal.ldm.revision_training import load_config
from mini_multimodal.training import load_checkpoint


def main() -> None:
    root = Path(".").resolve()
    config = load_config(root)
    result_dir = root / config["output_results"]
    report = json.loads((result_dir / "grounded_overfit_report.json").read_text(encoding="utf-8"))
    dataset = LatentTextDataset(
        root,
        data_dir=config["output_data"],
        record_ids=report["record_ids"],
        max_text_length=int(config["model"]["max_text_length"]),
    )
    model = grounded_model_from_config(config, dataset.tokenizer)
    load_checkpoint(
        result_dir / "grounded_overfit/checkpoints/last.pt",
        model=model,
        optimizer=None,
        restore_rng=False,
    )
    probes = _semantic_probes(root, result_dir, config, dataset, model)
    if probes["shapes"]["correct_count"] != report["semantic_probes"]["shapes"]["correct_count"]:
        raise RuntimeError("restored second-round probes differ from recorded report")


if __name__ == "__main__":
    main()
