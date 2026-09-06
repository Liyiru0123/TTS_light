"""Training, evaluation, ablation, visualization, and latent export for P2-W02."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from mini_multimodal.training import (
    TinyConvClassifier,
    Trainer,
    TrainerConfig,
    load_checkpoint,
    seeded_generator,
    set_deterministic_seed,
)
from mini_multimodal.training.data import SHAPE_CLASSES
from mini_multimodal.vision.masks import decode_rle
from mini_multimodal.vision.renderer import COLOR_RGB
from mini_multimodal.world.schema import Color

from .data import (
    VAEImageDataset,
    collate_vae_images,
    export_p2_w02_data,
    verify_p2_w02_data,
)
from .losses import VAELossWeights, compute_vae_loss
from .model import TinyImageVAE
from .task import ImageVAETask


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config["model"][key]
        for key in ("base_channels", "latent_channels", "logvar_min", "logvar_max")
    }


def _weights(values: Mapping[str, Any]) -> VAELossWeights:
    return VAELossWeights(**{field: float(values[field]) for field in (
        "global_weight", "global_mse_weight", "foreground_weight", "background_weight", "edge_weight", "kl_weight"
    )})


def _loader(
    dataset: VAEImageDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_vae_images,
        generator=seeded_generator(seed) if shuffle else None,
    )


def _new_trainer(
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    epochs: int,
    loss_weights: VAELossWeights,
) -> tuple[TinyImageVAE, torch.optim.Optimizer, Trainer]:
    model = TinyImageVAE(**_model_kwargs(config))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=ImageVAETask(loss_weights),
        config=TrainerConfig(
            epochs=epochs,
            device="cpu",
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            monitor="reconstruction_loss",
            maximize=False,
        ),
        output_dir=output_dir,
    )
    return model, optimizer, trainer


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _crop_object(image: torch.Tensor, bbox: Sequence[int]) -> torch.Tensor:
    x, y, width, height = (int(value) for value in bbox)
    padding = max(2, round(max(width, height) * 0.2))
    left, top = max(0, x - padding), max(0, y - padding)
    right = min(image.shape[-1], x + width + padding)
    bottom = min(image.shape[-2], y + height + padding)
    crop = image[:, top:bottom, left:right].unsqueeze(0)
    return F.interpolate(crop, size=(64, 64), mode="bilinear", align_corners=False)[0]


def evaluate_model(
    model: TinyImageVAE,
    loader: DataLoader,
    *,
    weights: VAELossWeights,
    shape_model: TinyConvClassifier,
    small_area_threshold: int,
    compress_to_8: bool = False,
) -> dict[str, float | int]:
    model.eval()
    shape_model.eval()
    totals = {key: 0.0 for key in (
        "loss", "reconstruction_loss", "global_mae", "global_mse", "foreground_mae", "background_mae", "edge_mae", "kl"
    )}
    scene_count = 0
    iou_sum = 0.0
    color_correct = shape_correct = object_count = 0
    small_color_correct = small_object_count = 0
    small_error_sum = small_pixel_count = 0.0
    palette_names = [color.value for color in Color]
    palette = torch.tensor([COLOR_RGB[color] for color in Color], dtype=torch.float32) / 255.0
    with torch.no_grad():
        for batch in loader:
            mean, logvar = model.encode(batch["image"])
            latent = mean
            if compress_to_8:
                latent = F.interpolate(
                    F.avg_pool2d(latent, kernel_size=2),
                    size=mean.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            reconstruction = model.decode(latent)
            output = type("Output", (), {
                "reconstruction": reconstruction,
                "mean": mean,
                "logvar": logvar,
                "latent": latent,
            })()
            components = compute_vae_loss(output, batch["image"], batch["foreground_mask"], weights)
            batch_size = len(batch["image"])
            scene_count += batch_size
            for key in totals:
                totals[key] += float(components[key]) * batch_size

            background = batch["background_rgb"][:, :, None, None]
            pixels = reconstruction.permute(0, 2, 3, 1)
            background_distance = (
                pixels - batch["background_rgb"][:, None, None, :]
            ).square().sum(dim=-1)
            palette_distance = (
                pixels[:, :, :, None, :] - palette[None, None, None, :, :]
            ).square().sum(dim=-1).min(dim=-1).values
            predicted_mask = (palette_distance < background_distance).unsqueeze(1)
            truth = batch["foreground_mask"].bool()
            intersection = (predicted_mask & truth).flatten(1).sum(dim=1).float()
            union = (predicted_mask | truth).flatten(1).sum(dim=1).float().clamp_min(1)
            iou_sum += float((intersection / union).sum())

            shape_crops: list[torch.Tensor] = []
            shape_targets: list[int] = []
            for batch_index, metadata in enumerate(batch["metadata"]):
                recon = reconstruction[batch_index]
                for obj in metadata["objects"]:
                    mask_np = decode_rle(obj["visible_rle"], 64, 64)
                    mask = torch.from_numpy(mask_np).bool()
                    pixels = recon[:, mask]
                    mean_color = pixels.mean(dim=1)
                    color_index = int((palette - mean_color).square().sum(dim=1).argmin())
                    color_correct += palette_names[color_index] == obj["color"]
                    object_count += 1
                    shape_crops.append(_crop_object(recon, obj["visible_bbox_xywh"]))
                    shape_targets.append(SHAPE_CLASSES.index(obj["shape"]))
                    if obj["visible_area_pixels"] <= small_area_threshold:
                        small_object_count += 1
                        small_color_correct += palette_names[color_index] == obj["color"]
                        target_pixels = batch["image"][batch_index, :, mask]
                        small_error_sum += float((pixels - target_pixels).abs().sum())
                        small_pixel_count += float(pixels.numel())
            if shape_crops:
                logits = shape_model(torch.stack(shape_crops))
                targets = torch.tensor(shape_targets, dtype=torch.long)
                shape_correct += int((logits.argmax(dim=1) == targets).sum())
    result: dict[str, float | int] = {
        key: value / scene_count for key, value in totals.items()
    }
    result.update(
        scene_count=scene_count,
        object_count=object_count,
        mask_iou=iou_sum / scene_count,
        color_accuracy=color_correct / object_count,
        shape_accuracy=shape_correct / object_count,
        small_object_count=small_object_count,
        small_color_accuracy=small_color_correct / small_object_count,
        small_foreground_mae=small_error_sum / small_pixel_count,
    )
    return result


def _save_training_curve(path: Path, history: list[Mapping[str, Any]]) -> None:
    epochs = [row["epoch"] + 1 for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].plot(epochs, [row["train"]["reconstruction_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation"]["reconstruction_loss"] for row in history], label="validation")
    axes[0].set(title="Reconstruction objective", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation"]["foreground_mae"] for row in history], label="foreground")
    axes[1].plot(epochs, [row["validation"]["background_mae"] for row in history], label="background")
    axes[1].set(title="Validation MAE", xlabel="Epoch")
    axes[1].legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_reconstruction_grid(path: Path, model: TinyImageVAE, dataset: VAEImageDataset) -> None:
    samples = [dataset[index] for index in range(min(8, len(dataset)))]
    batch = collate_vae_images(samples)
    with torch.no_grad():
        recon = model(batch["image"], sample_posterior=False).reconstruction
    figure, axes = plt.subplots(2, len(samples), figsize=(2 * len(samples), 4), constrained_layout=True)
    for index in range(len(samples)):
        axes[0, index].imshow(batch["image"][index].permute(1, 2, 0))
        axes[1, index].imshow(recon[index].permute(1, 2, 0).clamp(0, 1))
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_ylabel("source")
    axes[1, 0].set_ylabel("mean recon")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _save_interpolation(path: Path, model: TinyImageVAE, dataset: VAEImageDataset) -> None:
    batch = collate_vae_images([dataset[0], dataset[-1]])
    with torch.no_grad():
        mean, _ = model.encode(batch["image"])
        alphas = torch.linspace(0, 1, 7)
        latents = torch.stack([(1 - alpha) * mean[0] + alpha * mean[1] for alpha in alphas])
        images = model.decode(latents)
    figure, axes = plt.subplots(1, 7, figsize=(14, 2.2), constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.imshow(images[index].permute(1, 2, 0).clamp(0, 1))
        axis.set_title(f"{float(alphas[index]):.2f}")
        axis.axis("off")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def export_latent_cache(
    root: Path,
    output_dir: Path,
    model: TinyImageVAE,
    datasets: Sequence[VAEImageDataset],
    *,
    source_checkpoint: Path,
) -> dict[str, Any]:
    means: list[np.ndarray] = []
    scene_ids: list[str] = []
    splits: list[str] = []
    variants: list[str] = []
    model.eval()
    with torch.no_grad():
        for dataset in datasets:
            loader = _loader(dataset, batch_size=64)
            for batch in loader:
                mean, _ = model.encode(batch["image"])
                means.append(mean.cpu().numpy())
                scene_ids.extend(item["scene_id"] for item in batch["metadata"])
                splits.extend(item["split"] for item in batch["metadata"])
                variants.extend(item["variant"] for item in batch["metadata"])
    values = np.concatenate(means, axis=0)
    train_values = values[np.asarray(splits) == "train"]
    latent_std = float(train_values.std())
    scaling_factor = 1.0 / max(latent_std, 1e-8)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "posterior_means.npz"
    np.savez_compressed(
        cache_path,
        posterior_mean=values.astype(np.float16),
        scene_ids=np.asarray(scene_ids),
        splits=np.asarray(splits),
        variants=np.asarray(variants),
    )
    with torch.no_grad():
        original = torch.from_numpy(values[:16]).float()
        restored = torch.from_numpy(values[:16].astype(np.float16)).float()
        decode_mae = float((model.decode(original) - model.decode(restored)).abs().mean())
    manifest = {
        "artifact_version": 1,
        "count": len(scene_ids),
        "shape": list(values.shape),
        "storage_dtype": "float16",
        "latent_std": latent_std,
        "scaling_factor": scaling_factor,
        "float16_decode_mae": decode_mae,
        "source_checkpoint_sha256": _sha(source_checkpoint),
        "files": {
            "posterior_means.npz": {
                "bytes": cache_path.stat().st_size,
                "sha256": _sha(cache_path),
            }
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def run_p2_w02_experiments(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_dir = root / config["output_data"]
    latent_dir = root / config["output_latents"]
    result_dir = root / config["output_results"]
    result_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    start_rss = process.memory_info().rss
    started = perf_counter()
    seed = int(config["seed"])
    threads = int(config["cpu_threads"])
    set_deterministic_seed(seed, cpu_threads=threads)

    export_p2_w02_data(
        root,
        data_dir,
        source_index=config["source_index"],
        source_images=config["source_images"],
        overfit_count=int(config["training"]["overfit_count"]),
    )
    data_report = verify_p2_w02_data(root, data_dir)
    _write_json(result_dir / "data_readiness_report.json", data_report["statistics"])
    split_ids = json.loads((data_dir / "split_ids.json").read_text(encoding="utf-8"))
    overfit_ids = json.loads((data_dir / "overfit_ids.json").read_text(encoding="utf-8"))
    datasets = {
        name: VAEImageDataset(
            root,
            source_index=config["source_index"],
            source_images=config["source_images"],
            scene_ids=ids,
        )
        for name, ids in split_ids.items()
    }
    overfit_dataset = VAEImageDataset(
        root,
        source_index=config["source_index"],
        source_images=config["source_images"],
        scene_ids=overfit_ids,
    )
    loss_weights = _weights(config["loss"])

    # Full-scene small-data capacity/gradient gate. The balanced diagnostic
    # loss is intentional here: global L1 can lower its aggregate value on 20
    # sparse scenes by learning only the background, which does not test
    # whether object pixels receive a usable gradient. Formal training below
    # uses the mask-free global-MSE + KL objective.
    overfit_weights = _weights(config["ablation_loss"])
    set_deterministic_seed(seed, cpu_threads=threads)
    overfit_model, _, overfit_trainer = _new_trainer(
        result_dir / "overfit_run",
        config,
        epochs=int(config["training"]["overfit_epochs"]),
        loss_weights=overfit_weights,
    )
    overfit_loader = _loader(
        overfit_dataset,
        batch_size=len(overfit_dataset),
        shuffle=True,
        seed=seed,
    )
    overfit_history = overfit_trainer.fit(overfit_loader, overfit_loader)
    initial = overfit_history[0]["validation"]
    final = overfit_history[-1]["validation"]
    overfit_report = {
        "count": len(overfit_dataset),
        "purpose": "capacity_and_foreground_gradient_gate",
        "loss_weights": asdict(overfit_weights),
        "formal_training_uses_same_loss": False,
        "initial": initial,
        "final": final,
        "reconstruction_ratio": final["reconstruction_loss"] / initial["reconstruction_loss"],
        "thresholds": {
            "maximum_ratio": config["thresholds"]["overfit_reconstruction_ratio"],
            "foreground_mae": config["thresholds"]["overfit_foreground_mae"],
        },
    }
    overfit_report["verified"] = (
        overfit_report["reconstruction_ratio"] <= overfit_report["thresholds"]["maximum_ratio"]
        and final["foreground_mae"] <= overfit_report["thresholds"]["foreground_mae"]
    )
    _write_json(result_dir / "overfit_report.json", overfit_report)
    if not overfit_report["verified"]:
        raise RuntimeError("P2-W02 full-scene overfit gate failed")

    # Formal weighted model.
    set_deterministic_seed(seed, cpu_threads=threads)
    model, optimizer, trainer = _new_trainer(
        result_dir,
        config,
        epochs=int(config["training"]["epochs"]),
        loss_weights=loss_weights,
    )
    train_loader = _loader(
        datasets["train"],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(datasets["validation"], batch_size=64)
    history = trainer.fit(train_loader, validation_loader)
    best_path = result_dir / "checkpoints" / "best.pt"
    best_state = load_checkpoint(best_path, model=model, optimizer=optimizer, restore_rng=False)

    shape_model = TinyConvClassifier()
    load_checkpoint(
        root / config["source_shape_checkpoint"],
        model=shape_model,
        optimizer=None,
        restore_rng=False,
    )
    test_loader = _loader(datasets["test"], batch_size=64)
    small_threshold = int(data_report["statistics"]["small_object_area_threshold"])
    metrics = evaluate_model(
        model,
        test_loader,
        weights=loss_weights,
        shape_model=shape_model,
        small_area_threshold=small_threshold,
    )
    thresholds = config["thresholds"]
    checks = {
        "global_mae": metrics["global_mae"] <= thresholds["global_mae"],
        "foreground_mae": metrics["foreground_mae"] <= thresholds["foreground_mae"],
        "background_mae": metrics["background_mae"] <= thresholds["background_mae"],
        "mask_iou": metrics["mask_iou"] >= thresholds["mask_iou"],
        "color_accuracy": metrics["color_accuracy"] >= thresholds["color_accuracy"],
        "shape_accuracy": metrics["shape_accuracy"] >= thresholds["shape_accuracy"],
        "small_foreground_mae": metrics["small_foreground_mae"] <= thresholds["small_foreground_mae"],
        "small_color_accuracy": metrics["small_color_accuracy"] >= thresholds["small_color_accuracy"],
        "minimum_kl": metrics["kl"] > thresholds["minimum_kl"],
    }
    model_report = {
        "best_epoch": int(best_state["epoch"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": metrics,
        "checks": checks,
        "verified": all(checks.values()),
    }
    _write_json(result_dir / "vae_metrics.json", model_report)

    # Strong foreground/boundary weighting retained as a falsifiable ablation.
    ablation_weights = _weights(config["ablation_loss"])
    set_deterministic_seed(seed, cpu_threads=threads)
    ablation_model, ablation_optimizer, ablation_trainer = _new_trainer(
        result_dir / "strong_foreground_ablation",
        config,
        epochs=int(config["training"]["ablation_epochs"]),
        loss_weights=ablation_weights,
    )
    ablation_loader = _loader(
        datasets["train"],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    ablation_trainer.fit(ablation_loader, validation_loader)
    load_checkpoint(
        result_dir / "strong_foreground_ablation" / "checkpoints" / "best.pt",
        model=ablation_model,
        optimizer=ablation_optimizer,
        restore_rng=False,
    )
    ablation_metrics = evaluate_model(
        ablation_model,
        test_loader,
        weights=ablation_weights,
        shape_model=shape_model,
        small_area_threshold=small_threshold,
    )
    foreground_change = ablation_metrics["foreground_mae"] - metrics["foreground_mae"]
    semantic_change = max(
        abs(metrics["mask_iou"] - ablation_metrics["mask_iou"]),
        abs(metrics["color_accuracy"] - ablation_metrics["color_accuracy"]),
        abs(metrics["shape_accuracy"] - ablation_metrics["shape_accuracy"]),
    )
    ablation_report = {
        "primary_global_mse_plus_kl": metrics,
        "strong_foreground_boundary_weighting": ablation_metrics,
        "foreground_mae_change_ablation_minus_primary": foreground_change,
        "maximum_absolute_semantic_change": semantic_change,
        "threshold": thresholds["ablation_improvement"],
        "verified": semantic_change >= thresholds["ablation_improvement"],
    }
    _write_json(result_dir / "loss_ablation_report.json", ablation_report)

    compressed_metrics = evaluate_model(
        model,
        test_loader,
        weights=loss_weights,
        shape_model=shape_model,
        small_area_threshold=small_threshold,
        compress_to_8=True,
    )
    compression_report = {
        "latent_16x16": metrics,
        "simulated_8x8": compressed_metrics,
        "foreground_mae_increase": compressed_metrics["foreground_mae"] - metrics["foreground_mae"],
        "mask_iou_drop": metrics["mask_iou"] - compressed_metrics["mask_iou"],
        "threshold": thresholds["compression_degradation"],
    }
    compression_report["verified"] = (
        compression_report["foreground_mae_increase"] > 0
        or compression_report["mask_iou_drop"] >= thresholds["compression_degradation"]
    )
    _write_json(result_dir / "compression_ablation_report.json", compression_report)

    # Posterior sampling observation.
    sample_batch = collate_vae_images([datasets["test"][0], datasets["test"][1]])
    with torch.no_grad():
        mean, logvar = model.encode(sample_batch["image"])
        first = model.decode(model.reparameterize(mean, logvar, generator=torch.Generator().manual_seed(1)))
        second = model.decode(model.reparameterize(mean, logvar, generator=torch.Generator().manual_seed(2)))
        sample_difference = float((first - second).abs().mean())
    posterior_report = {
        "test_kl": metrics["kl"],
        "two_sample_reconstruction_mae": sample_difference,
        "mean_reconstruction_deterministic": True,
        "verified": metrics["kl"] > thresholds["minimum_kl"] and sample_difference > 0,
    }
    _write_json(result_dir / "posterior_report.json", posterior_report)

    latent_report = export_latent_cache(
        root,
        latent_dir,
        model,
        [datasets["train"], datasets["validation"], datasets["test"]],
        source_checkpoint=best_path,
    )
    latent_report["verified"] = (
        latent_report["count"] == 1720
        and latent_report["shape"] == [1720, 4, 16, 16]
        and latent_report["float16_decode_mae"] <= thresholds["latent_decode_mae"]
    )
    _write_json(result_dir / "latent_cache_report.json", latent_report)

    # Short deterministic repeat.
    digests: list[str] = []
    sequences: list[list[dict[str, float]]] = []
    for name in ("a", "b"):
        set_deterministic_seed(seed + 31, cpu_threads=threads)
        repro_model, _, repro_trainer = _new_trainer(
            result_dir / f"repro_run_{name}",
            config,
            epochs=2,
            loss_weights=loss_weights,
        )
        repro_loader = _loader(overfit_dataset, batch_size=10, shuffle=True, seed=seed + 31)
        repro_history = repro_trainer.fit(repro_loader, repro_loader)
        digests.append(_state_digest(repro_model))
        sequences.append([
            {"loss": row["train"]["loss"], "foreground_mae": row["train"]["foreground_mae"]}
            for row in repro_history
        ])
    reproducibility = {
        "state_sha256": digests,
        "state_equal": digests[0] == digests[1],
        "metrics_equal": sequences[0] == sequences[1],
    }
    reproducibility["verified"] = reproducibility["state_equal"] and reproducibility["metrics_equal"]
    _write_json(result_dir / "reproducibility_report.json", reproducibility)

    _save_training_curve(result_dir / "training_curve.png", history)
    _save_reconstruction_grid(result_dir / "reconstruction_grid.png", model, datasets["test"])
    _save_interpolation(result_dir / "latent_interpolation.png", model, datasets["test"])

    elapsed = perf_counter() - started
    memory = process.memory_info()
    peak_rss = int(getattr(memory, "peak_wset", memory.rss))
    resource = {
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "wall_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "parameter_count": model_report["parameter_count"],
        "limits": {
            "max_wall_seconds": thresholds["max_wall_seconds"],
            "max_peak_rss_bytes": thresholds["max_peak_rss_bytes"],
            "max_parameters": thresholds["max_parameters"],
        },
    }
    resource["verified"] = (
        elapsed <= thresholds["max_wall_seconds"]
        and peak_rss <= thresholds["max_peak_rss_bytes"]
        and model_report["parameter_count"] < thresholds["max_parameters"]
    )
    _write_json(result_dir / "resource_benchmark.json", resource)

    summary = {
        "data_verified": data_report["verified"],
        "overfit_verified": overfit_report["verified"],
        "model_verified": model_report["verified"],
        "loss_ablation_verified": ablation_report["verified"],
        "compression_ablation_verified": compression_report["verified"],
        "posterior_verified": posterior_report["verified"],
        "latent_cache_verified": latent_report["verified"],
        "reproducibility_verified": reproducibility["verified"],
        "resource_verified": resource["verified"],
    }
    summary["verified"] = all(summary.values())
    _write_json(result_dir / "experiment_summary.json", summary)
    if not summary["verified"]:
        raise RuntimeError("one or more P2-W02 experiment gates failed")
    return summary
