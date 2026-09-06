"""Third-round training-only foreground and boundary supervision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from mini_multimodal.training import Trainer, TrainerConfig, load_checkpoint, seeded_generator, set_deterministic_seed
from mini_multimodal.training.core import StepOutput

from .geometry_data import GeometryLatentTextDataset, collate_geometry_text, soft_foreground_probability
from .grounded_training import _overfit_ids, _semantic_probes, grounded_model_from_config
from .inference import load_vae_decoder
from .revision_training import _state_sha, evaluate_denoising_v2, load_config, schedule_from_config


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boundary_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = (probability[:, :, :, 1:] - probability[:, :, :, :-1]).abs()
    true_x = (target[:, :, :, 1:] - target[:, :, :, :-1]).abs()
    pred_y = (probability[:, :, 1:, :] - probability[:, :, :-1, :]).abs()
    true_y = (target[:, :, 1:, :] - target[:, :, :-1, :]).abs()
    return 0.5 * (F.l1_loss(pred_x, true_x) + F.l1_loss(pred_y, true_y))


def _region_loss(probability: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pixel_weight = 1.0 + 4.0 * target
    bce = (F.binary_cross_entropy(probability, target, reduction="none") * pixel_weight).sum() / pixel_weight.sum()
    intersection = (probability * target).flatten(1).sum(1)
    dice = (2.0 * intersection + 1.0) / (
        probability.flatten(1).sum(1) + target.flatten(1).sum(1) + 1.0
    )
    dice_loss = 1.0 - dice.mean()
    return 0.5 * (bce + dice_loss), bce, dice_loss


class GeometryAuxiliaryDiffusionTask:
    def __init__(
        self,
        schedule: Any,
        vae: torch.nn.Module,
        *,
        latent_scaling_factor: float,
        condition_dropout: float,
        high_noise_training_fraction: float,
        foreground_noise_weight: float,
        null_id: int,
        soft_chroma_threshold: float,
        soft_chroma_temperature: float,
        geometry_max_timestep: int,
        max_geometry_samples: int,
        geometry_foreground_weight: float,
        geometry_boundary_weight: float,
    ) -> None:
        self.schedule = schedule
        self.vae = vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        self.latent_scaling_factor = float(latent_scaling_factor)
        self.condition_dropout = float(condition_dropout)
        self.high_noise_training_fraction = float(high_noise_training_fraction)
        self.foreground_noise_weight = float(foreground_noise_weight)
        self.null_id = int(null_id)
        self.soft_chroma_threshold = float(soft_chroma_threshold)
        self.soft_chroma_temperature = float(soft_chroma_temperature)
        self.geometry_max_timestep = int(geometry_max_timestep)
        self.max_geometry_samples = int(max_geometry_samples)
        self.geometry_foreground_weight = float(geometry_foreground_weight)
        self.geometry_boundary_weight = float(geometry_boundary_weight)
        if not 0 <= self.condition_dropout < 1 or not 0 <= self.high_noise_training_fraction <= 1:
            raise ValueError("invalid diffusion task probabilities")
        if min(self.geometry_foreground_weight, self.geometry_boundary_weight) < 0:
            raise ValueError("geometry weights must be non-negative")

    def step(self, model: torch.nn.Module, batch: dict[str, Any], *, training: bool) -> StepOutput:
        clean = batch["latent"]
        batch_size = clean.shape[0]
        timestep = torch.randint(0, self.schedule.timesteps, (batch_size,), device=clean.device)
        high_noise = torch.zeros(batch_size, dtype=torch.bool, device=clean.device)
        if training and self.high_noise_training_fraction:
            high_noise = torch.rand(batch_size, device=clean.device) < self.high_noise_training_fraction
            timestep[high_noise] = torch.randint(
                self.schedule.timesteps // 2,
                self.schedule.timesteps,
                (int(high_noise.sum()),),
                device=clean.device,
            )
        noise = torch.randn_like(clean)
        noisy = self.schedule.q_sample(clean, timestep, noise)
        text_ids = batch["text_ids"].clone()
        text_mask = batch["text_mask"].clone()
        dropped = torch.zeros(batch_size, dtype=torch.bool, device=clean.device)
        if training and self.condition_dropout:
            dropped = torch.rand(batch_size, device=clean.device) < self.condition_dropout
            text_ids[dropped] = 0
            text_mask[dropped] = False
            text_ids[dropped, 0] = self.null_id
            text_mask[dropped, 0] = True
        output = model(noisy, timestep, text_ids, text_mask)
        squared_error = (output.predicted_noise - noise).square()
        spatial_weight = 1.0 + self.foreground_noise_weight * batch["foreground_latent_mask"]
        noise_loss = (squared_error * spatial_weight).sum() / (spatial_weight.sum() * squared_error.shape[1])
        unweighted = F.mse_loss(output.predicted_noise, noise)

        geometry_region = noise_loss.new_zeros(())
        geometry_bce = noise_loss.new_zeros(())
        geometry_dice = noise_loss.new_zeros(())
        geometry_boundary = noise_loss.new_zeros(())
        selected_count = 0
        if self.geometry_foreground_weight or self.geometry_boundary_weight:
            eligible = (~dropped) & (timestep <= self.geometry_max_timestep)
            selected = eligible.nonzero(as_tuple=False).flatten()[: self.max_geometry_samples]
            selected_count = int(selected.numel())
            if selected_count:
                predicted_clean = self.schedule.predict_clean(
                    noisy[selected], timestep[selected], output.predicted_noise[selected]
                )
                decoded = self.vae.decode(predicted_clean / self.latent_scaling_factor)
                probability = soft_foreground_probability(
                    decoded,
                    threshold=self.soft_chroma_threshold,
                    temperature=self.soft_chroma_temperature,
                )
                target = batch["geometry_mask"][selected]
                geometry_region, geometry_bce, geometry_dice = _region_loss(probability, target)
                geometry_boundary = _boundary_loss(probability, target)
        loss = (
            noise_loss
            + self.geometry_foreground_weight * geometry_region
            + self.geometry_boundary_weight * geometry_boundary
        )
        return StepOutput(
            loss=loss,
            metrics={
                "noise_mse": float(noise_loss.detach()),
                "unweighted_noise_mse": float(unweighted.detach()),
                "zero_prediction_mse": float(noise.square().mean().detach()),
                "geometry_region_loss": float(geometry_region.detach()),
                "geometry_bce": float(geometry_bce.detach()),
                "geometry_dice_loss": float(geometry_dice.detach()),
                "geometry_boundary_loss": float(geometry_boundary.detach()),
                "geometry_selected_fraction": selected_count / batch_size,
                "condition_drop_fraction": float(dropped.float().mean()),
                "high_noise_fraction": float(high_noise.float().mean()),
            },
            weight=batch_size,
        )


def _dataset(root: Path, config: Mapping[str, Any], **kwargs: Any) -> GeometryLatentTextDataset:
    return GeometryLatentTextDataset(
        root,
        data_dir=config["output_data"],
        geometry_data_dir=config["geometry_auxiliary"]["data_dir"],
        max_text_length=int(config["model"]["max_text_length"]),
        **kwargs,
    )


def _loader(dataset: GeometryLatentTextDataset, config: Mapping[str, Any], *, seed: int, balanced: bool) -> DataLoader:
    sampler = None
    if balanced:
        single_weight = float(config["training"]["single_object_sampling_weight"])
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
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_geometry_text,
    )


def _task(
    root: Path,
    config: Mapping[str, Any],
    dataset: GeometryLatentTextDataset,
    *,
    geometry_variant: str,
) -> GeometryAuxiliaryDiffusionTask:
    aux = config["geometry_auxiliary"]
    if geometry_variant == "control":
        foreground_weight = boundary_weight = 0.0
    elif geometry_variant == "geometry":
        foreground_weight = float(aux["foreground_weight"])
        boundary_weight = float(aux["boundary_weight"])
    elif geometry_variant == "geometry_lite":
        foreground_weight = float(aux["lite_foreground_weight"])
        boundary_weight = float(aux["lite_boundary_weight"])
    else:
        raise ValueError("unsupported geometry variant")
    return GeometryAuxiliaryDiffusionTask(
        schedule_from_config(config),
        load_vae_decoder(root, config["source_vae_checkpoint"]),
        latent_scaling_factor=dataset.scaling_factor,
        condition_dropout=float(config["diffusion"]["condition_dropout"]),
        high_noise_training_fraction=float(config["diffusion"]["high_noise_training_fraction"]),
        foreground_noise_weight=float(config["diffusion"]["foreground_noise_weight"]),
        null_id=dataset.tokenizer.null_id,
        soft_chroma_threshold=float(aux["soft_chroma_threshold"]),
        soft_chroma_temperature=float(aux["soft_chroma_temperature"]),
        geometry_max_timestep=int(aux["max_timestep"]),
        max_geometry_samples=int(aux["max_samples_per_batch"]),
        geometry_foreground_weight=foreground_weight,
        geometry_boundary_weight=boundary_weight,
    )


def _trainer(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    dataset: GeometryLatentTextDataset,
    *,
    epochs: int,
    geometry_variant: str,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, Trainer]:
    model = grounded_model_from_config(config, dataset.tokenizer)
    warmstart = root / config["output_results"] / "training_grounded/checkpoints/best.pt"
    load_checkpoint(warmstart, model=model, optimizer=None, restore_rng=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["geometry_auxiliary"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=_task(root, config, dataset, geometry_variant=geometry_variant),
        config=TrainerConfig(
            epochs=epochs,
            device="cpu",
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            monitor="loss",
            maximize=False,
        ),
        output_dir=output,
    )
    return model, optimizer, trainer


def run_geometry_overfit(
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    geometry_variant: str = "geometry",
) -> dict[str, Any]:
    if geometry_variant not in {"geometry", "geometry_lite"}:
        raise ValueError("overfit variant must be geometry or geometry_lite")
    root = Path(project_root).resolve()
    result_dir = root / config["output_results"]
    calibration = json.loads((result_dir / "attempt3_geometry_calibration.json").read_text(encoding="utf-8"))
    if not calibration.get("verified_for_training"):
        raise RuntimeError("geometry overfit requires calibrated foreground metric")
    seed = int(config["seed"]) + 3
    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))
    full = _dataset(root, config, split="train")
    ids = _overfit_ids(full, int(config["training"]["overfit_count"]))
    subset = _dataset(root, config, record_ids=ids)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, collate_fn=collate_geometry_text)
    output_name = "attempt3_overfit" if geometry_variant == "geometry" else "attempt3_geometry_lite_overfit"
    model, _, trainer = _trainer(
        root,
        result_dir / output_name,
        config,
        subset,
        epochs=int(config["geometry_auxiliary"]["overfit_epochs"]),
        geometry_variant=geometry_variant,
    )
    initial_probes = _semantic_probes(
        root,
        result_dir,
        config,
        subset,
        model,
        probe_subdir=f"{output_name}/probes_initial",
    )
    started = perf_counter()
    history = trainer.fit(loader, loader)
    final_probes = _semantic_probes(
        root,
        result_dir,
        config,
        subset,
        model,
        probe_subdir=f"{output_name}/probes_final",
    )
    initial_counts = [item["predicted_object_count"] for group in initial_probes.values() for item in group["items"]]
    final_counts = [item["predicted_object_count"] for group in final_probes.values() for item in group["items"]]
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "acceptance_item_id": "COMP-LDG-002",
        "attempt": "training_only_geometry_auxiliary_v3",
        "geometry_variant": geometry_variant,
        "record_ids": ids,
        "sample_count": len(subset),
        "epochs": len(history),
        "initial_total_loss": float(history[0]["train"]["loss"]),
        "final_total_loss": float(history[-1]["train"]["loss"]),
        "initial_geometry_region_loss": float(history[0]["train"]["geometry_region_loss"]),
        "final_geometry_region_loss": float(history[-1]["train"]["geometry_region_loss"]),
        "initial_probes": initial_probes,
        "final_probes": final_probes,
        "initial_probe_object_count_mean": sum(initial_counts) / len(initial_counts),
        "final_probe_object_count_mean": sum(final_counts) / len(final_counts),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_sha256": _state_sha(model),
        "wall_seconds": perf_counter() - started,
    }
    report["checks"] = {
        "total_loss_declined": report["final_total_loss"] < report["initial_total_loss"],
        "geometry_region_loss_declined": report["final_geometry_region_loss"] < report["initial_geometry_region_loss"],
        "same_noise_position_direction": bool(final_probes["horizontal"]["left_to_right_monotonic"]),
        "same_noise_shape_correct_at_least_4_of_5": int(final_probes["shapes"]["correct_count"]) >= 4,
        "same_noise_shape_diversity_at_least_4": int(final_probes["shapes"]["unique_predicted_shapes"]) >= 4,
        "probe_fragments_declined": report["final_probe_object_count_mean"] < report["initial_probe_object_count_mean"],
        "parameter_limit": report["parameter_count"] < int(config["training_gates"]["max_parameters"]),
    }
    report["verified_for_finetuning"] = all(report["checks"].values())
    report_name = "attempt3_overfit_report.json" if geometry_variant == "geometry" else "attempt3_geometry_lite_overfit_report.json"
    _write_json(result_dir / report_name, report)
    if not report["verified_for_finetuning"]:
        raise RuntimeError("attempt-3 geometry overfit gate failed")
    return report


def run_attempt3_finetuning(project_root: str | Path, config: Mapping[str, Any], *, branch: str) -> dict[str, Any]:
    if branch not in {"control", "geometry", "geometry_lite"}:
        raise ValueError("unsupported attempt-3 branch")
    root = Path(project_root).resolve()
    result_dir = root / config["output_results"]
    if branch != "control":
        overfit_name = "attempt3_overfit_report.json" if branch == "geometry" else "attempt3_geometry_lite_overfit_report.json"
        overfit = json.loads((result_dir / overfit_name).read_text(encoding="utf-8"))
        if not overfit.get("verified_for_finetuning"):
            raise RuntimeError("geometry branch requires passing overfit gate")
    seed = int(config["seed"]) + 300
    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))
    train = _dataset(root, config, split="train")
    validation = _dataset(root, config, split="validation")
    test = _dataset(root, config, split="test")
    output = result_dir / f"training_attempt3_{branch}"
    model, optimizer, trainer = _trainer(
        root,
        output,
        config,
        train,
        epochs=int(config["geometry_auxiliary"]["finetune_epochs"]),
        geometry_variant=branch,
    )
    process = psutil.Process()
    started = perf_counter()
    history = trainer.fit(_loader(train, config, seed=seed, balanced=True), _loader(validation, config, seed=seed, balanced=False))
    checkpoint = output / "checkpoints/best.pt"
    best = load_checkpoint(checkpoint, model=model, optimizer=optimizer, restore_rng=False)
    core = evaluate_denoising_v2(
        model,
        schedule_from_config(config),
        _loader(test, config, seed=seed, balanced=False),
        seed=seed + 101,
    )
    elapsed = perf_counter() - started
    memory = process.memory_info()
    peak_rss = int(getattr(memory, "peak_wset", memory.rss))
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "acceptance_item_id": "MODEL-LDG-001",
        "attempt": "training_only_geometry_auxiliary_v3",
        "branch": branch,
        "geometry_enabled": branch != "control",
        "geometry_variant": branch,
        "warmstart": f"{config['output_results']}/training_grounded/checkpoints/best.pt",
        "warmstart_sha256": _sha(result_dir / "training_grounded/checkpoints/best.pt"),
        "checkpoint": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": _sha(checkpoint),
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["best_metric"]),
        "epochs": len(history),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_denoising": core,
        "training": {
            "final": history[-1],
            "wall_seconds": elapsed,
            "peak_rss_bytes": peak_rss,
        },
    }
    report["checks"] = {
        "noise_mse": core["noise_mse"] <= float(config["training_gates"]["test_noise_mse"]),
        "parameter_limit": report["parameter_count"] < int(config["training_gates"]["max_parameters"]),
        "time_limit": elapsed <= float(config["training_gates"]["max_wall_seconds"]),
        "memory_limit": peak_rss <= int(config["training_gates"]["max_peak_rss_bytes"]),
    }
    report["verified_for_generation_evaluation"] = all(report["checks"].values())
    _write_json(result_dir / f"attempt3_{branch}_training_metrics.json", report)
    if not report["verified_for_generation_evaluation"]:
        raise RuntimeError(f"attempt-3 {branch} training gate failed")
    return report


def main_overfit(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    return run_geometry_overfit(root, load_config(root))
