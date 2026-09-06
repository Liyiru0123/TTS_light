"""Resume the P2-W02 formal model to diagnose slow semantic convergence."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.training import TinyConvClassifier, load_checkpoint, set_deterministic_seed
from mini_multimodal.vae.artifacts import _loader, _new_trainer, _weights, evaluate_model
from mini_multimodal.vae.data import VAEImageDataset, verify_p2_w02_data


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "configs" / "p2_w02_image_vae.json").read_text(encoding="utf-8")
    )
    result_dir = root / config["output_results"]
    data_dir = root / config["output_data"]
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
    test = VAEImageDataset(
        root,
        source_index=config["source_index"],
        source_images=config["source_images"],
        scene_ids=split_ids["test"],
    )
    threads = int(config["cpu_threads"])
    seed = int(config["seed"])
    set_deterministic_seed(seed, cpu_threads=threads)
    weights = _weights(config["loss"])
    model, optimizer, trainer = _new_trainer(
        result_dir,
        config,
        epochs=25,
        loss_weights=weights,
    )
    trainer.fit(
        _loader(train, batch_size=int(config["training"]["batch_size"]), shuffle=True, seed=seed),
        _loader(validation, batch_size=64),
        resume_from=result_dir / "checkpoints" / "last.pt",
    )
    best = load_checkpoint(
        result_dir / "checkpoints" / "best.pt",
        model=model,
        optimizer=optimizer,
        restore_rng=False,
    )
    shape_model = TinyConvClassifier()
    load_checkpoint(
        root / config["source_shape_checkpoint"],
        model=shape_model,
        optimizer=None,
        restore_rng=False,
    )
    stats = verify_p2_w02_data(root, data_dir)["statistics"]
    metrics = evaluate_model(
        model,
        _loader(test, batch_size=64),
        weights=weights,
        shape_model=shape_model,
        small_area_threshold=int(stats["small_object_area_threshold"]),
    )
    report = {"resumed_from_epoch": 14, "best_epoch": int(best["epoch"]), "metrics": metrics}
    (result_dir / "resume_diagnostic_epoch25.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
