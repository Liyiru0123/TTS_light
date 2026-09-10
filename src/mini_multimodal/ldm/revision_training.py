"""Training and denoising checks for the P2-W04 coordinate-aware revision."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import psutil
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

from mini_multimodal.training import Trainer, TrainerConfig, load_checkpoint, seeded_generator, set_deterministic_seed

from .data import CharacterTokenizer, LatentTextDataset, collate_latent_text, verify_p2_w04_data
from .evaluator import load_shape_prototypes, probe_image_v2
from .inference import load_vae_decoder, sample_latents
from .model_v2 import TinyImageDenoiserV2, resolve_spatial_token_coordinates
from .schedule import DiffusionSchedule
from .task import LatentDiffusionTask


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_config(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    return json.loads((root / "configs/p2_w04_tiny_image_ldm_v2.json").read_text(encoding="utf-8"))


def schedule_from_config(config: Mapping[str, Any]) -> DiffusionSchedule:
    return DiffusionSchedule(
        timesteps=int(config["diffusion"]["train_timesteps"]),
        beta_start=float(config["diffusion"]["beta_start"]),
        beta_end=float(config["diffusion"]["beta_end"]),
    )


def model_from_config(config: Mapping[str, Any], tokenizer: CharacterTokenizer) -> TinyImageDenoiserV2:
    keys = ("latent_channels", "base_channels", "text_dim", "time_dim", "attention_heads", "max_text_length")
    return TinyImageDenoiserV2(
        vocab_size=len(tokenizer),
        **{key: config["model"][key] for key in keys},
        spatial_token_coordinates=resolve_spatial_token_coordinates(tokenizer.token_to_id),
    )


def _task(
    config: Mapping[str, Any],
    tokenizer: CharacterTokenizer,
    *,
    high_noise_training_fraction: float | None = None,
) -> LatentDiffusionTask:
    return LatentDiffusionTask(
        schedule_from_config(config),
        condition_dropout=float(config["diffusion"]["condition_dropout"]),
        high_noise_training_fraction=float(
            config["diffusion"]["high_noise_training_fraction"]
            if high_noise_training_fraction is None
            else high_noise_training_fraction
        ),
        foreground_noise_weight=float(config["diffusion"]["foreground_noise_weight"]),
        null_id=tokenizer.null_id,
    )


def _trainer(
    output: Path,
    config: Mapping[str, Any],
    tokenizer: CharacterTokenizer,
    *,
    epochs: int,
    high_noise_training_fraction: float | None = None,
) -> tuple[TinyImageDenoiserV2, torch.optim.Optimizer, Trainer]:
    model = model_from_config(config, tokenizer)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=_task(config, tokenizer, high_noise_training_fraction=high_noise_training_fraction),
        config=TrainerConfig(
            epochs=epochs,
            device="cpu",
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            monitor="noise_mse",
            maximize=False,
        ),
        output_dir=output,
    )
    return model, optimizer, trainer


def _loader(
    dataset: LatentTextDataset,
    config: Mapping[str, Any],
    *,
    shuffle: bool,
    seed: int,
    balanced_objects: bool = False,
) -> DataLoader:
    sampler = None
    if balanced_objects:
        single_weight = float(config["training"].get("single_object_sampling_weight", 1.0))
        weights = [single_weight if len(row["objects"]) == 1 else 1.0 for row in dataset.rows]
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=seeded_generator(seed),
        )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_latent_text,
        generator=seeded_generator(seed) if shuffle and sampler is None else None,
    )


@torch.no_grad()
def evaluate_denoising_v2(
    model: TinyImageDenoiserV2,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    *,
    seed: int,
    forward_options: Mapping[str, bool] | None = None,
) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(seed)
    total = baseline = 0.0
    count = 0
    model.eval()
    options = dict(forward_options or {})
    for batch in loader:
        clean = batch["latent"]
        steps = torch.randint(0, schedule.timesteps, (clean.shape[0],), generator=generator)
        noise = torch.randn(clean.shape, generator=generator)
        noisy = schedule.q_sample(clean, steps, noise)
        predicted = model(noisy, steps, batch["text_ids"], batch["text_mask"], **options).predicted_noise
        total += float(F.mse_loss(predicted, noise, reduction="sum"))
        baseline += float(noise.square().sum())
        count += noise.numel()
    return {
        "sample_count": len(loader.dataset),
        "element_count": count,
        "noise_mse": total / count,
        "zero_prediction_mse": baseline / count,
        "improvement": (baseline - total) / count,
    }


def _horizontal_triplet(dataset: LatentTextDataset) -> list[dict[str, Any]]:
    required = ("center_left", "center", "center_right")
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in dataset.rows:
        if len(row["objects"]) != 1:
            continue
        obj = row["objects"][0]
        if obj["position_region"] not in required:
            continue
        key = (obj["color"], obj["shape"], obj["size"])
        groups.setdefault(key, {}).setdefault(obj["position_region"], row)
    for key in sorted(groups):
        if all(region in groups[key] for region in required):
            return [groups[key][region] for region in required]
    raise RuntimeError("v2 training view has no same-attribute horizontal position triplet")


def _overfit_rows(dataset: LatentTextDataset, count: int) -> list[str]:
    singles = [row for row in dataset.rows if len(row["objects"]) == 1]
    singles.sort(key=lambda row: (row["objects"][0]["position_region"], row["objects"][0]["color"], row["record_id"]))
    selected: list[dict[str, Any]] = list(_horizontal_triplet(dataset))
    selected_ids = {row["record_id"] for row in selected}
    regions = {row["objects"][0]["position_region"] for row in selected}
    for row in singles:
        region = row["objects"][0]["position_region"]
        if region not in regions and row["record_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["record_id"])
            regions.add(region)
    selected.extend(row for row in singles if row["record_id"] not in selected_ids)
    return [row["record_id"] for row in selected[:count]]


def run_overfit(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result_dir = root / str(config["output_results"])
    seed = int(config["seed"])
    set_deterministic_seed(seed + 1, cpu_threads=int(config["cpu_threads"]))
    train = LatentTextDataset(root, data_dir=config["output_data"], split="train", max_text_length=int(config["model"]["max_text_length"]))
    count = int(config["training"]["overfit_count"])
    selected_ids = _overfit_rows(train, count)
    subset = LatentTextDataset(root, data_dir=config["output_data"], record_ids=selected_ids, max_text_length=int(config["model"]["max_text_length"]))
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, collate_fn=collate_latent_text)
    overfit_high_noise = float(config["training"]["overfit_high_noise_training_fraction"])
    model, _, trainer = _trainer(
        result_dir / "overfit_v2",
        config,
        subset.tokenizer,
        epochs=int(config["training"]["overfit_epochs"]),
        high_noise_training_fraction=overfit_high_noise,
    )
    started = perf_counter()
    history = trainer.fit(loader, loader)
    initial = float(history[0]["train"]["noise_mse"])
    final = float(history[-1]["train"]["noise_mse"])
    triplet = _horizontal_triplet(subset)
    initial_latent = torch.randn(
        (1, int(config["model"]["latent_channels"]), 16, 16),
        generator=torch.Generator().manual_seed(seed + 500),
    ).repeat(3, 1, 1, 1)
    sampled = sample_latents(
        model,
        schedule_from_config(config),
        subset.tokenizer,
        [row["prompt"] for row in triplet],
        sample_steps=int(config["diffusion"]["sample_steps"]),
        guidance_scale=float(config["diffusion"]["guidance_scale"]),
        seed=seed + 500,
        initial_latent=initial_latent,
    )
    vae = load_vae_decoder(root, config["source_vae_checkpoint"])
    images = vae.decode(sampled["latent"] / subset.scaling_factor).clamp(0, 1)
    prototypes = load_shape_prototypes(result_dir / "evaluator_shape_templates.npz")
    probe_dir = result_dir / "overfit_v2/spatial_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    spatial_probe: list[dict[str, Any]] = []
    for index, (row, image) in enumerate(zip(triplet, images, strict=True)):
        probe = probe_image_v2(image, shape_prototypes=prototypes)
        path = probe_dir / f"horizontal_{index}.png"
        Image.fromarray(image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), mode="RGB").save(path)
        center = None if not probe["objects"] else probe["objects"][0]["center_xy"]
        spatial_probe.append({
            "record_id": row["record_id"],
            "target_region": row["objects"][0]["position_region"],
            "prompt": row["prompt"],
            "predicted_center_xy": center,
            "generated_image": path.relative_to(root).as_posix(),
        })
    x_centers = [None if row["predicted_center_xy"] is None else float(row["predicted_center_xy"][0]) for row in spatial_probe]
    direction_verified = all(value is not None for value in x_centers) and bool(x_centers[0] < x_centers[1] < x_centers[2])

    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "acceptance_item_id": "COMP-LDM-001",
        "record_ids": selected_ids,
        "sample_count": len(subset),
        "covered_position_regions": sorted({row["objects"][0]["position_region"] for row in subset.rows}),
        "initial_noise_mse": initial,
        "final_noise_mse": final,
        "loss_ratio": final / initial,
        "zero_prediction_mse": float(history[-1]["train"]["zero_prediction_mse"]),
        "threshold": float(config["training_gates"]["overfit_loss_ratio"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_sha256": _state_sha(model),
        "wall_seconds": perf_counter() - started,
        "overfit_high_noise_training_fraction": overfit_high_noise,
        "same_noise_horizontal_probe": {
            "seed": seed + 500,
            "rows": spatial_probe,
            "x_centers": x_centers,
            "left_to_right_monotonic": direction_verified,
        },
    }
    report["checks"] = {
        "all_nine_regions_present": len(report["covered_position_regions"]) == 9,
        "loss_ratio": report["loss_ratio"] <= report["threshold"],
        "below_zero_prediction": final < report["zero_prediction_mse"],
        "parameter_limit": report["parameter_count"] < int(config["training_gates"]["max_parameters"]),
        "same_noise_position_direction": direction_verified,
    }
    report["verified_for_full_training"] = all(report["checks"].values())
    _write_json(result_dir / "overfit_report.json", report)
    if not report["verified_for_full_training"]:
        raise RuntimeError("P2-W04 v2 overfit gate failed")
    return report


def run_training(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result_dir = root / str(config["output_results"])
    overfit = json.loads((result_dir / "overfit_report.json").read_text(encoding="utf-8"))
    if not overfit.get("verified_for_full_training"):
        raise RuntimeError("full training requires a passing overfit gate")
    readiness = verify_p2_w04_data(root, config["output_data"])
    holdout = json.loads((root / config["output_data"] / "holdout_audit.json").read_text(encoding="utf-8"))
    if not holdout.get("verified") or holdout["final_training_holdout_object_count"] != 0:
        raise RuntimeError("v2 training holdout audit failed")
    frozen = json.loads((result_dir / "threshold_freeze.json").read_text(encoding="utf-8"))
    if not frozen.get("verified") or frozen.get("status") != "frozen_before_model_revision":
        raise RuntimeError("frozen evaluation thresholds are missing")

    seed = int(config["seed"])
    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))
    train = LatentTextDataset(root, data_dir=config["output_data"], split="train", max_text_length=int(config["model"]["max_text_length"]))
    validation = LatentTextDataset(root, data_dir=config["output_data"], split="validation", max_text_length=int(config["model"]["max_text_length"]))
    test = LatentTextDataset(root, data_dir=config["output_data"], split="test", max_text_length=int(config["model"]["max_text_length"]))
    model, optimizer, trainer = _trainer(result_dir, config, train.tokenizer, epochs=int(config["training"]["epochs"]))
    process = psutil.Process()
    started = perf_counter()
    history = trainer.fit(
        _loader(train, config, shuffle=False, seed=seed, balanced_objects=True),
        _loader(validation, config, shuffle=False, seed=seed),
    )
    checkpoint = result_dir / "checkpoints/best.pt"
    best = load_checkpoint(checkpoint, model=model, optimizer=optimizer, restore_rng=False)
    test_loader = _loader(test, config, shuffle=False, seed=seed)
    schedule = schedule_from_config(config)
    core = evaluate_denoising_v2(model, schedule, test_loader, seed=seed + 101)
    no_text = evaluate_denoising_v2(model, schedule, test_loader, seed=seed + 101, forward_options={"disable_cross_attention": True})
    no_coordinates = evaluate_denoising_v2(model, schedule, test_loader, seed=seed + 101, forward_options={"disable_coordinates": True})
    no_high = evaluate_denoising_v2(model, schedule, test_loader, seed=seed + 101, forward_options={"disable_cross_attention_16": True})
    no_spatial_bias = evaluate_denoising_v2(model, schedule, test_loader, seed=seed + 101, forward_options={"disable_spatial_attention_bias": True})
    no_spatial_map = evaluate_denoising_v2(model, schedule, test_loader, seed=seed + 101, forward_options={"disable_text_spatial_map": True})
    elapsed = perf_counter() - started
    memory = process.memory_info()
    peak_rss = int(getattr(memory, "peak_wset", memory.rss))
    metrics = {
        "schema_version": 2,
        "work_package": "P2-W04",
        "acceptance_item_id": "MODEL-LDM-001",
        "checkpoint": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": _sha(checkpoint),
        "best_epoch": int(best["epoch"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "core": core,
        "ablations": {
            "disable_all_cross_attention": no_text,
            "disable_coordinates": no_coordinates,
            "disable_16x16_cross_attention": no_high,
            "disable_spatial_attention_bias": no_spatial_bias,
            "disable_text_spatial_map": no_spatial_map,
        },
        "training": {
            "epochs": len(history),
            "final_train_noise_mse": float(history[-1]["train"]["noise_mse"]),
            "final_validation_noise_mse": float(history[-1]["validation"]["noise_mse"]),
            "wall_seconds": elapsed,
            "peak_rss_bytes": peak_rss,
        },
        "provenance": {
            "training_view": str(config["output_data"]),
            "data_verified": bool(readiness["verified"]),
            "training_holdout_object_count": int(holdout["final_training_holdout_object_count"]),
            "holdout_policy_sha256": holdout["policy_sha256"],
            "threshold_freeze_sha256": _sha(result_dir / "threshold_freeze.json"),
        },
    }
    gates = config["training_gates"]
    metrics["checks"] = {
        "noise_mse": core["noise_mse"] <= float(gates["test_noise_mse"]),
        "baseline_improvement": core["improvement"] >= float(gates["noise_baseline_improvement"]),
        "parameter_limit": metrics["parameter_count"] < int(gates["max_parameters"]),
        "time_limit": elapsed <= float(gates["max_wall_seconds"]),
        "memory_limit": peak_rss <= int(gates["max_peak_rss_bytes"]),
    }
    metrics["verified_for_generation_evaluation"] = all(metrics["checks"].values())
    _write_json(result_dir / "denoising_metrics.json", metrics)
    if not metrics["verified_for_generation_evaluation"]:
        raise RuntimeError("P2-W04 v2 denoising gate failed")
    return metrics


def run_reproducibility_probe(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result_dir = root / str(config["output_results"])
    train = LatentTextDataset(root, data_dir=config["output_data"], split="train", max_text_length=int(config["model"]["max_text_length"]))
    subset_ids = _overfit_rows(train, 16)
    subset = LatentTextDataset(root, data_dir=config["output_data"], record_ids=subset_ids, max_text_length=int(config["model"]["max_text_length"]))
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, collate_fn=collate_latent_text)
    states: list[str] = []
    histories: list[list[dict[str, Any]]] = []
    for suffix in ("a", "b"):
        set_deterministic_seed(int(config["seed"]) + 77, cpu_threads=int(config["cpu_threads"]))
        model, _, trainer = _trainer(result_dir / f"repro_v2_{suffix}", config, subset.tokenizer, epochs=2)
        histories.append(trainer.fit(loader, loader))
        states.append(_state_sha(model))
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "acceptance_item_id": "REG-LDM-001",
        "state_sha256": states,
        "state_equal": states[0] == states[1],
        "metrics_equal": histories[0] == histories[1],
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    report["verified"] = report["state_equal"] and report["metrics_equal"]
    _write_json(result_dir / "reproducibility_report.json", report)
    return report
