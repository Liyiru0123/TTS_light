"""Fixed-checkpoint guidance-scale diagnostic after failed generation semantics."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.ldm.artifacts import (
    _aggregate_generation,
    _load_shape_model,
    _load_vae,
    _model_kwargs,
    _schedule,
    generate_records,
)
from mini_multimodal.ldm.data import CharacterTokenizer, LatentTextDataset
from mini_multimodal.ldm.model import TinyImageDenoiser
from mini_multimodal.training import load_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocabulary)
    dataset = LatentTextDataset(root, data_dir=config["output_data"], split="test", max_text_length=config["model"]["max_text_length"])
    rows = [row for row in dataset.rows if len(row["objects"]) == 1 and row["variant"] != "compositional"][:32]
    model = TinyImageDenoiser(**_model_kwargs(config, len(tokenizer)))
    load_checkpoint(
        root / config["output_results"] / "training" / "checkpoints" / "best.pt",
        model=model,
        optimizer=None,
        restore_rng=False,
    )
    vae = _load_vae(root, config)
    shape_model = _load_shape_model(root, config)
    results = {}
    for steps in (25, 50):
        for scale in (0.0, 1.0, 1.5, 2.0):
            records = generate_records(
                model,
                _schedule(config),
                tokenizer,
                vae,
                shape_model,
                rows,
                scaling_factor=dataset.scaling_factor,
                sample_steps=steps,
                guidance_scale=scale,
                seed=config["seed"] + 200,
            )
            results[f"steps={steps},scale={scale}"] = _aggregate_generation(records)
    destination = root / config["output_results"] / "guidance_scale_diagnostic.json"
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
