"""One-epoch seed sweep for the sparse-scene VAE optimization trajectory."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.training import set_deterministic_seed
from mini_multimodal.vae.artifacts import _loader, _new_trainer, _weights
from mini_multimodal.vae.data import VAEImageDataset


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "configs" / "p2_w02_image_vae.json").read_text(encoding="utf-8")
    )
    data_dir = root / config["output_data"]
    result_dir = root / config["output_results"] / "seed_sweep"
    split_ids = json.loads((data_dir / "split_ids.json").read_text(encoding="utf-8"))
    train = VAEImageDataset(
        root,
        source_index=config["source_index"],
        source_images=config["source_images"],
        scene_ids=split_ids["train"],
    )
    validation = VAEImageDataset(
        root,
        source_index=config["source_index"],
        source_images=config["source_images"],
        scene_ids=split_ids["validation"],
    )
    base_seed = int(config["seed"])
    threads = int(config["cpu_threads"])
    weights = _weights(config["loss"])
    rows = []
    for offset in range(6):
        seed = base_seed + offset
        set_deterministic_seed(seed, cpu_threads=threads)
        _, _, trainer = _new_trainer(
            result_dir / f"seed_{seed}", config, epochs=1, loss_weights=weights
        )
        history = trainer.fit(
            _loader(
                train,
                batch_size=int(config["training"]["batch_size"]),
                shuffle=True,
                seed=seed,
            ),
            _loader(validation, batch_size=64),
        )
        rows.append({"seed": seed, "validation": history[0]["validation"]})
    report = {"criterion": "lowest one_epoch_validation_foreground_mae", "runs": rows}
    (root / config["output_results"] / "seed_sweep_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
