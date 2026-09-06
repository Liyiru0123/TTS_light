"""Training, evaluation, visual evidence, and export for P2-W04."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from PIL import Image
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
from mini_multimodal.vae.model import TinyImageVAE
from mini_multimodal.vision.renderer import COLOR_RGB
from mini_multimodal.world.schema import Color

from .data import CharacterTokenizer, LatentTextDataset, collate_latent_text, verify_p2_w04_data
from .inference import sample_latents
from .model import TinyImageDenoiser
from .schedule import DiffusionSchedule
from .task import LatentDiffusionTask


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_kwargs(config: Mapping[str, Any], vocab_size: int) -> dict[str, Any]:
    return {"vocab_size": vocab_size, **{key: config["model"][key] for key in (
        "latent_channels", "base_channels", "text_dim", "time_dim", "attention_heads", "max_text_length"
    )}}


def _schedule(config: Mapping[str, Any]) -> DiffusionSchedule:
    return DiffusionSchedule(
        timesteps=int(config["diffusion"]["train_timesteps"]),
        beta_start=float(config["diffusion"]["beta_start"]),
        beta_end=float(config["diffusion"]["beta_end"]),
    )


def _loader(dataset: LatentTextDataset, config: Mapping[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_latent_text,
        generator=seeded_generator(seed) if shuffle else None,
    )


def _new_trainer(
    output: Path,
    config: Mapping[str, Any],
    tokenizer: CharacterTokenizer,
    *,
    epochs: int,
) -> tuple[TinyImageDenoiser, torch.optim.Optimizer, Trainer]:
    model = TinyImageDenoiser(**_model_kwargs(config, len(tokenizer)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=LatentDiffusionTask(
            _schedule(config),
            condition_dropout=float(config["diffusion"]["condition_dropout"]),
            high_noise_training_fraction=float(config["diffusion"].get("high_noise_training_fraction", 0.0)),
            foreground_noise_weight=float(config["diffusion"].get("foreground_noise_weight", 0.0)),
            null_id=tokenizer.null_id,
        ),
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


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate_denoising(
    model: TinyImageDenoiser,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    *,
    seed: int,
    disable_cross_attention: bool = False,
) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(seed)
    total_loss = baseline = 0.0
    count = 0
    model.eval()
    for batch in loader:
        clean = batch["latent"]
        batch_size = clean.shape[0]
        timestep = torch.randint(0, schedule.timesteps, (batch_size,), generator=generator)
        noise = torch.randn(clean.shape, generator=generator)
        noisy = schedule.q_sample(clean, timestep, noise)
        output = model(
            noisy,
            timestep,
            batch["text_ids"],
            batch["text_mask"],
            disable_cross_attention=disable_cross_attention,
        )
        total_loss += float(F.mse_loss(output.predicted_noise, noise, reduction="sum"))
        baseline += float(noise.square().sum())
        count += noise.numel()
    return {
        "sample_count": len(loader.dataset),
        "element_count": count,
        "noise_mse": total_loss / count,
        "zero_prediction_mse": baseline / count,
        "improvement": (baseline - total_loss) / count,
    }


def _load_vae(root: Path, config: Mapping[str, Any]) -> TinyImageVAE:
    vae_config = json.loads((root / "configs" / "p2_w02_image_vae.json").read_text(encoding="utf-8"))
    model = TinyImageVAE(**{key: vae_config["model"][key] for key in (
        "base_channels", "latent_channels", "logvar_min", "logvar_max"
    )})
    load_checkpoint(root / config["source_vae_checkpoint"], model=model, optimizer=None, restore_rng=False)
    model.eval()
    return model


def _load_shape_model(root: Path, config: Mapping[str, Any]) -> TinyConvClassifier:
    model = TinyConvClassifier(class_count=5)
    load_checkpoint(root / config["source_shape_checkpoint"], model=model, optimizer=None, restore_rng=False)
    model.eval()
    return model


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for y, x in zip(*np.nonzero(mask), strict=True):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) >= 20:
            component = np.zeros_like(mask, dtype=bool)
            yy, xx = zip(*pixels, strict=True)
            component[np.asarray(yy), np.asarray(xx)] = True
            result.append(component)
    return sorted(result, key=lambda value: int(value.sum()), reverse=True)


@torch.no_grad()
def probe_image(image: torch.Tensor, shape_model: TinyConvClassifier) -> dict[str, Any]:
    array = image.permute(1, 2, 0).cpu().numpy()
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]), axis=0)
    background = np.median(border, axis=0)
    palette_names = [color.value for color in Color]
    palette = np.asarray([COLOR_RGB[color] for color in Color], dtype=np.float32) / 255.0
    palette_distance = ((array[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=-1)
    background_distance = ((array - background[None, None, :]) ** 2).sum(axis=-1)
    mask = palette_distance.min(axis=-1) + 0.005 < background_distance
    components = _components(mask)
    objects: list[dict[str, Any]] = []
    for component in components[:3]:
        ys, xs = np.nonzero(component)
        mean = array[component].mean(axis=0)
        color_index = int(((palette - mean[None, :]) ** 2).sum(axis=1).argmin())
        left, right = max(0, int(xs.min()) - 2), min(64, int(xs.max()) + 3)
        top, bottom = max(0, int(ys.min()) - 2), min(64, int(ys.max()) + 3)
        crop = image[:, top:bottom, left:right].unsqueeze(0)
        crop = F.interpolate(crop, size=(64, 64), mode="bilinear", align_corners=False)
        shape_index = int(shape_model(crop).argmax(dim=1))
        objects.append({
            "color": palette_names[color_index],
            "shape": SHAPE_CLASSES[shape_index],
            "area": int(component.sum()),
            "center_xy": [float(xs.mean() / 63.0), float(ys.mean() / 63.0)],
        })
    return {
        "foreground_present": bool(components),
        "predicted_object_count": len(objects),
        "objects": objects,
        "foreground_pixel_count": int(mask.sum()),
    }


def _semantic_match(targets: Sequence[Mapping[str, Any]], predicted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    remaining = list(range(len(predicted)))
    color_correct = shape_correct = joint_correct = 0
    pairs = []
    for target in targets:
        if not remaining:
            pairs.append({"target": target, "prediction": None})
            continue
        best = max(
            remaining,
            key=lambda index: int(predicted[index]["color"] == target["color"]) + int(predicted[index]["shape"] == target["shape"]),
        )
        remaining.remove(best)
        item = predicted[best]
        color = item["color"] == target["color"]
        shape = item["shape"] == target["shape"]
        color_correct += color
        shape_correct += shape
        joint_correct += color and shape
        pairs.append({"target": target, "prediction": item, "color_correct": color, "shape_correct": shape})
    denominator = max(len(targets), 1)
    return {
        "target_object_count": len(targets),
        "count_correct": len(targets) == len(predicted),
        "color_accuracy": color_correct / denominator,
        "shape_accuracy": shape_correct / denominator,
        "joint_accuracy": joint_correct / denominator,
        "pairs": pairs,
    }


@torch.no_grad()
def generate_records(
    model: TinyImageDenoiser,
    schedule: DiffusionSchedule,
    tokenizer: CharacterTokenizer,
    vae: TinyImageVAE,
    shape_model: TinyConvClassifier,
    rows: Sequence[Mapping[str, Any]],
    *,
    scaling_factor: float,
    sample_steps: int,
    guidance_scale: float,
    seed: int,
    prompt_override: Sequence[str] | None = None,
    save_dir: Path | None = None,
    prefix: str = "sample",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for start in range(0, len(rows), 8):
        batch_rows = rows[start : start + 8]
        prompts = [row["prompt"] for row in batch_rows]
        if prompt_override is not None:
            prompts = list(prompt_override[start : start + len(batch_rows)])
        sampled = sample_latents(
            model,
            schedule,
            tokenizer,
            prompts,
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
            seed=seed + start,
        )
        images = vae.decode(sampled["latent"] / scaling_factor).clamp(0, 1)
        for offset, (row, prompt, image) in enumerate(zip(batch_rows, prompts, images, strict=True)):
            probe = probe_image(image, shape_model)
            match = _semantic_match(row["objects"], probe["objects"])
            image_path = None
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                image_path = save_dir / f"{prefix}_{start + offset:03d}.png"
                array = image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                Image.fromarray(array, mode="RGB").save(image_path)
            results.append({
                "record_id": row["record_id"],
                "scene_id": row["scene_id"],
                "variant": row["variant"],
                "prompt": prompt,
                "target_prompt": row["prompt"],
                "target_objects": row["objects"],
                "guidance_scale": guidance_scale,
                "sample_seed": seed + start,
                "generated_image": None if image_path is None else image_path.as_posix(),
                "probe": probe,
                "semantic_match": match,
                "is_fully_correct": bool(
                    probe["foreground_present"]
                    and match["count_correct"]
                    and match["joint_accuracy"] == 1.0
                ),
            })
    return results


def _aggregate_generation(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    target_count = sum(len(row["target_objects"]) for row in rows)
    result = {
        "sample_count": count,
        "target_object_count": target_count,
        "foreground_presence": sum(row["probe"]["foreground_present"] for row in rows) / count,
        "count_accuracy": sum(row["semantic_match"]["count_correct"] for row in rows) / count,
        "color_accuracy": sum(row["semantic_match"]["color_accuracy"] * len(row["target_objects"]) for row in rows) / target_count,
        "shape_accuracy": sum(row["semantic_match"]["shape_accuracy"] * len(row["target_objects"]) for row in rows) / target_count,
        "joint_accuracy": sum(row["semantic_match"]["joint_accuracy"] * len(row["target_objects"]) for row in rows) / target_count,
        "fully_correct_rate": sum(row["is_fully_correct"] for row in rows) / count,
    }
    result["semantic_mean"] = (
        result["foreground_presence"] + result["color_accuracy"] + result["shape_accuracy"]
    ) / 3.0
    return result


def _save_training_curve(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["epoch"] for row in history], [row["train"]["noise_mse"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["validation"]["noise_mse"] for row in history], label="validation")
    axis.set(xlabel="epoch", ylabel="noise MSE", title="TinyImageLDM noise prediction")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def _save_forward_noise(path: Path, schedule: DiffusionSchedule, dataset: LatentTextDataset, vae: TinyImageVAE) -> None:
    clean = dataset[0]["latent"].unsqueeze(0)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(17))
    steps = [0, schedule.timesteps // 4, schedule.timesteps // 2, schedule.timesteps - 1]
    images = []
    for step in steps:
        noisy = schedule.q_sample(clean, torch.tensor([step]), noise)
        images.append(vae.decode(noisy / dataset.scaling_factor)[0].clamp(0, 1))
    figure, axes = plt.subplots(1, len(images), figsize=(8, 2.2))
    for axis, image, step in zip(axes, images, steps, strict=True):
        axis.imshow(image.permute(1, 2, 0).cpu())
        axis.set_title(f"t={step}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


@torch.no_grad()
def _save_trajectory(
    path: Path,
    model: TinyImageDenoiser,
    schedule: DiffusionSchedule,
    tokenizer: CharacterTokenizer,
    vae: TinyImageVAE,
    prompt: str,
    scaling_factor: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    sampled = sample_latents(
        model,
        schedule,
        tokenizer,
        [prompt],
        sample_steps=int(config["diffusion"]["sample_steps"]),
        guidance_scale=float(config["diffusion"]["guidance_scale"]),
        seed=int(config["seed"]) + 900,
    )
    indices = np.linspace(0, len(sampled["latent_trajectory"]) - 1, 6).round().astype(int).tolist()
    images = [vae.decode(sampled["latent_trajectory"][index] / scaling_factor)[0].clamp(0, 1) for index in indices]
    figure, axes = plt.subplots(1, len(images), figsize=(12, 2.2))
    for axis, image, index in zip(axes, images, indices, strict=True):
        axis.imshow(image.permute(1, 2, 0).cpu())
        axis.set_title(f"state {index}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    changes = [
        float((right - left).abs().mean())
        for left, right in zip(sampled["latent_trajectory"], sampled["latent_trajectory"][1:])
    ]
    return {"exported_state_count": len(indices), "trajectory_state_count": len(sampled["latent_trajectory"]), "adjacent_latent_mae": changes}


def run_experiments(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result_dir = root / config["output_results"]
    result_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    process = psutil.Process()
    seed = int(config["seed"])
    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))
    data_report = verify_p2_w04_data(root, config["output_data"])
    vocabulary = json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocabulary)
    train = LatentTextDataset(root, data_dir=config["output_data"], split="train", max_text_length=int(config["model"]["max_text_length"]))
    validation = LatentTextDataset(root, data_dir=config["output_data"], split="validation", max_text_length=int(config["model"]["max_text_length"]))
    test = LatentTextDataset(root, data_dir=config["output_data"], split="test", max_text_length=int(config["model"]["max_text_length"]))
    thresholds = config["thresholds"]

    overfit_ids = [row["record_id"] for row in train.rows[: int(config["data"]["overfit_count"])]]
    overfit = LatentTextDataset(root, data_dir=config["output_data"], record_ids=overfit_ids, max_text_length=int(config["model"]["max_text_length"]))
    set_deterministic_seed(seed + 1, cpu_threads=int(config["cpu_threads"]))
    overfit_model, _, overfit_trainer = _new_trainer(
        result_dir / "overfit", config, tokenizer, epochs=int(config["training"]["overfit_epochs"])
    )
    overfit_loader = DataLoader(overfit, batch_size=len(overfit), shuffle=False, collate_fn=collate_latent_text)
    overfit_history = overfit_trainer.fit(overfit_loader, overfit_loader)
    initial_overfit = overfit_history[0]["train"]["noise_mse"]
    final_overfit = overfit_history[-1]["train"]["noise_mse"]
    overfit_report = {
        "sample_count": len(overfit),
        "initial_noise_mse": initial_overfit,
        "final_noise_mse": final_overfit,
        "loss_ratio": final_overfit / initial_overfit,
        "zero_prediction_mse": overfit_history[-1]["train"]["zero_prediction_mse"],
        "threshold": thresholds["overfit_loss_ratio"],
    }
    overfit_report["verified"] = (
        overfit_report["loss_ratio"] <= thresholds["overfit_loss_ratio"]
        and final_overfit < overfit_report["zero_prediction_mse"]
    )
    _write_json(result_dir / "overfit_report.json", overfit_report)
    if not overfit_report["verified"]:
        raise RuntimeError("P2-W04 overfit acceptance item failed")

    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))
    model, optimizer, trainer = _new_trainer(
        result_dir / "training", config, tokenizer, epochs=int(config["training"]["epochs"])
    )
    history = trainer.fit(
        _loader(train, config, shuffle=True, seed=seed),
        _loader(validation, config, shuffle=False, seed=seed),
    )
    best_path = result_dir / "training" / "checkpoints" / "best.pt"
    best_state = load_checkpoint(best_path, model=model, optimizer=optimizer, restore_rng=False)
    schedule = _schedule(config)
    denoising = evaluate_denoising(
        model, schedule, _loader(test, config, shuffle=False, seed=seed), seed=seed + 101
    )
    no_cross = evaluate_denoising(
        model, schedule, _loader(test, config, shuffle=False, seed=seed), seed=seed + 101, disable_cross_attention=True
    )
    denoising.update({
        "best_epoch": int(best_state["epoch"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "cross_attention_disabled_noise_mse": no_cross["noise_mse"],
        "cross_attention_mse_increase": no_cross["noise_mse"] - denoising["noise_mse"],
        "checks": {
            "noise_mse": denoising["noise_mse"] <= thresholds["test_noise_mse"],
            "baseline_improvement": denoising["improvement"] >= thresholds["noise_baseline_improvement"],
        },
    })
    denoising["verified"] = all(denoising["checks"].values())
    _write_json(result_dir / "denoising_metrics.json", denoising)
    if not denoising["verified"]:
        raise RuntimeError("P2-W04 denoising acceptance item failed")

    vae = _load_vae(root, config)
    shape_model = _load_shape_model(root, config)
    single_rows = [row for row in test.rows if len(row["objects"]) == 1 and row["variant"] != "compositional"][: int(config["data"]["generation_evaluation_count"])]
    composition_rows = [row for row in test.rows if len(row["objects"]) == 1 and row["variant"] == "compositional"][: int(config["data"]["composition_evaluation_count"])]
    evaluation_rows = single_rows + composition_rows
    generated_dir = result_dir / "generated_samples"
    full_records = generate_records(
        model, schedule, tokenizer, vae, shape_model, evaluation_rows,
        scaling_factor=test.scaling_factor,
        sample_steps=int(config["diffusion"]["sample_steps"]),
        guidance_scale=float(config["diffusion"]["guidance_scale"]),
        seed=seed + 200,
        save_dir=generated_dir,
        prefix="conditional",
    )
    single_metrics = _aggregate_generation(full_records[: len(single_rows)])
    composition_metrics = _aggregate_generation(full_records[len(single_rows) :])
    generation = {
        "single_object": single_metrics,
        "unseen_composition": composition_metrics,
        "checks": {
            "single_color_accuracy": single_metrics["color_accuracy"] >= thresholds["single_color_accuracy"],
            "single_shape_accuracy": single_metrics["shape_accuracy"] >= thresholds["single_shape_accuracy"],
            "foreground_presence": single_metrics["foreground_presence"] >= thresholds["foreground_presence"],
            "composition_mean_accuracy": (composition_metrics["color_accuracy"] + composition_metrics["shape_accuracy"]) / 2 >= thresholds["composition_mean_accuracy"],
        },
    }
    generation["verified"] = all(generation["checks"].values())
    _write_json(result_dir / "generation_metrics.json", generation)
    _write_jsonl(result_dir / "generation_predictions.jsonl", full_records)
    if not generation["verified"]:
        raise RuntimeError("P2-W04 generation semantics acceptance item failed")

    shuffled_prompts = [row["prompt"] for row in evaluation_rows[1:]] + [evaluation_rows[0]["prompt"]]
    ablations = {}
    for name, scale, prompts in (
        ("no_guidance", 1.0, None),
        ("unconditional", 0.0, None),
        ("shuffled_text", float(config["diffusion"]["guidance_scale"]), shuffled_prompts),
    ):
        records = generate_records(
            model, schedule, tokenizer, vae, shape_model, evaluation_rows,
            scaling_factor=test.scaling_factor,
            sample_steps=int(config["diffusion"]["sample_steps"]),
            guidance_scale=scale,
            seed=seed + 200,
            prompt_override=prompts,
        )
        ablations[name] = _aggregate_generation(records)
    full_metrics = _aggregate_generation(full_records)
    best_control = max(ablations["unconditional"]["semantic_mean"], ablations["shuffled_text"]["semantic_mean"])
    conditioning = {
        "conditional_cfg": full_metrics,
        **ablations,
        "advantage_over_best_unconditional_or_shuffled": full_metrics["semantic_mean"] - best_control,
        "threshold": thresholds["conditioning_advantage"],
    }
    conditioning["verified"] = (
        conditioning["advantage_over_best_unconditional_or_shuffled"] >= thresholds["conditioning_advantage"]
    )
    _write_json(result_dir / "conditioning_ablation.json", conditioning)
    if not conditioning["verified"]:
        raise RuntimeError("P2-W04 conditioning acceptance item failed")

    trajectory = _save_trajectory(
        result_dir / "denoising_trajectory.png", model, schedule, tokenizer, vae,
        evaluation_rows[0]["prompt"], test.scaling_factor, config,
    )
    guidance = {
        "conditional_cfg_semantic_mean": full_metrics["semantic_mean"],
        "no_guidance_semantic_mean": ablations["no_guidance"]["semantic_mean"],
        **trajectory,
    }
    guidance["verified"] = (
        guidance["conditional_cfg_semantic_mean"] >= guidance["no_guidance_semantic_mean"]
        and guidance["exported_state_count"] >= 6
        and all(value > 0 for value in guidance["adjacent_latent_mae"])
    )
    _write_json(result_dir / "guidance_report.json", guidance)
    if not guidance["verified"]:
        raise RuntimeError("P2-W04 guidance acceptance item failed")

    digests = []
    sequences = []
    for suffix in ("a", "b"):
        set_deterministic_seed(seed + 77, cpu_threads=int(config["cpu_threads"]))
        repro_model, _, repro_trainer = _new_trainer(result_dir / f"repro_{suffix}", config, tokenizer, epochs=2)
        repro_history = repro_trainer.fit(overfit_loader, overfit_loader)
        digests.append(_state_digest(repro_model))
        sequences.append(repro_history)
    reproducibility = {
        "state_sha256": digests,
        "state_equal": digests[0] == digests[1],
        "metrics_equal": sequences[0] == sequences[1],
    }
    reproducibility["verified"] = reproducibility["state_equal"] and reproducibility["metrics_equal"]
    _write_json(result_dir / "reproducibility_report.json", reproducibility)

    _save_training_curve(result_dir / "training_curve.png", history)
    _save_forward_noise(result_dir / "forward_noise_process.png", schedule, test, vae)
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
        "parameter_count": denoising["parameter_count"],
        "limits": {
            "max_wall_seconds": thresholds["max_wall_seconds"],
            "max_peak_rss_bytes": thresholds["max_peak_rss_bytes"],
            "max_parameters": thresholds["max_parameters"],
        },
    }
    resource["verified"] = (
        elapsed <= thresholds["max_wall_seconds"]
        and peak_rss <= thresholds["max_peak_rss_bytes"]
        and denoising["parameter_count"] < thresholds["max_parameters"]
    )
    _write_json(result_dir / "resource_benchmark.json", resource)

    summary = {
        "data_verified": data_report["verified"],
        "overfit_verified": overfit_report["verified"],
        "denoising_verified": denoising["verified"],
        "generation_verified": generation["verified"],
        "conditioning_verified": conditioning["verified"],
        "guidance_verified": guidance["verified"],
        "reproducibility_verified": reproducibility["verified"],
        "resource_verified": resource["verified"],
    }
    summary["verified"] = all(summary.values())
    _write_json(result_dir / "experiment_summary.json", summary)
    if not summary["verified"]:
        raise RuntimeError("one or more P2-W04 acceptance items failed")
    return summary
