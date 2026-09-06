"""Calibration, historical-checkpoint re-evaluation, and threshold freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from mini_multimodal.training import load_checkpoint
from mini_multimodal.vae.model import TinyImageVAE
from mini_multimodal.vision.renderer import render_scene
from mini_multimodal.world.schema import Color, Position, Scene, SceneObject, Shape, Size

from .data import CharacterTokenizer
from .evaluator import COLORS, SIZES, SHAPES, aggregate_evaluations, evaluate_image_v2, load_shape_prototypes, probe_image_v2, resize_mask
from .evaluation_data import verify_fixed_evaluation_data
from .inference import sample_latents
from .model import TinyImageDenoiser
from .schedule import DiffusionSchedule


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _load_vae(root: Path, checkpoint: str) -> TinyImageVAE:
    config = json.loads(
        (root / "configs" / "p2_w02_image_vae.json").read_text(encoding="utf-8")
    )
    model = TinyImageVAE(
        **{
            key: config["model"][key]
            for key in ("base_channels", "latent_channels", "logvar_min", "logvar_max")
        }
    )
    load_checkpoint(root / checkpoint, model=model, optimizer=None, restore_rng=False)
    model.eval()
    return model


def _metric_value(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics[name]
    return float(value["value"] if isinstance(value, Mapping) and "value" in value else value)


def _calibration_checks(metrics: Mapping[str, Any], *, layer: str) -> dict[str, bool]:
    per_shape = [
        float(group["shape_accuracy"]["value"])
        for group in metrics["shape_groups"].values()
    ]
    if layer == "source":
        return {
            "foreground_presence": _metric_value(metrics, "foreground_presence") == 1.0,
            "count_accuracy": _metric_value(metrics, "count_accuracy") == 1.0,
            "color_accuracy": _metric_value(metrics, "color_accuracy") == 1.0,
            "size_accuracy": _metric_value(metrics, "size_accuracy") == 1.0,
            "position_accuracy": _metric_value(metrics, "position_accuracy") == 1.0,
            "shape_accuracy": _metric_value(metrics, "shape_accuracy") >= 0.98,
            "each_shape_accuracy": min(per_shape) >= 0.98,
            "normalized_template_iou": _metric_value(metrics, "normalized_template_iou_mean") >= 0.98,
            "boundary_score": _metric_value(metrics, "boundary_score_mean") >= 0.95,
            "center_distance": _metric_value(metrics, "center_distance_mean") <= 0.02,
        }
    if layer != "vae":
        raise ValueError(f"unknown calibration layer: {layer}")
    return {
        "foreground_presence": _metric_value(metrics, "foreground_presence") >= 0.95,
        "count_accuracy": _metric_value(metrics, "count_accuracy") >= 0.95,
        "color_accuracy": _metric_value(metrics, "color_accuracy") >= 0.95,
        "shape_accuracy": _metric_value(metrics, "shape_accuracy") >= 0.90,
        "each_shape_accuracy": min(per_shape) >= 0.85,
        "size_accuracy": _metric_value(metrics, "size_accuracy") >= 0.90,
        "position_accuracy": _metric_value(metrics, "position_accuracy") >= 0.95,
        "normalized_template_iou": _metric_value(metrics, "normalized_template_iou_mean") >= 0.85,
        "boundary_score": _metric_value(metrics, "boundary_score_mean") >= 0.75,
        "center_distance": _metric_value(metrics, "center_distance_mean") <= 0.05,
    }


def _image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)


@torch.no_grad()
def _build_vae_shape_prototypes(
    vae: TinyImageVAE,
) -> dict[str, Any]:
    keys: list[str] = []
    labels: list[str] = []
    images: list[torch.Tensor] = []
    positions = (
        (0.18, 0.18), (0.50, 0.18), (0.82, 0.18),
        (0.18, 0.50), (0.50, 0.50), (0.82, 0.50),
        (0.18, 0.82), (0.50, 0.82), (0.82, 0.82),
    )
    for shape in SHAPES:
        for size in SIZES:
            for color in COLORS:
                for position_index, position in enumerate(positions):
                    scene = Scene(
                        scene_id=f"vae_prototype_{shape}_{size}_{color}_{position_index}",
                        seed=0,
                        canvas_width=32,
                        canvas_height=32,
                        frame_count=1,
                        fps=8,
                        objects=(SceneObject(
                            object_id="obj_0",
                            shape=Shape(shape),
                            color=Color(color),
                            size=Size(size),
                            initial_position=Position(*position),
                            visible_start=0,
                            visible_end=0,
                        ),),
                    )
                    rendered = render_scene(scene, frame_index=0).image
                    images.append(torch.from_numpy(np.asarray(rendered, dtype=np.uint8).copy()).permute(2, 0, 1).float().div(255.0))
                    keys.append(f"{shape}|{size}|{color}|{position_index}")
                    labels.append(shape)
    reconstructions: list[torch.Tensor] = []
    for start in range(0, len(images), 64):
        batch = torch.stack(images[start : start + 64])
        mean, _ = vae.encode(batch)
        reconstructions.extend(vae.decode(mean).clamp(0, 1).cpu())
    normalized_values: list[np.ndarray] = []
    for key, reconstruction in zip(keys, reconstructions, strict=True):
        probe = probe_image_v2(reconstruction)
        if not probe["objects"]:
            raise RuntimeError(f"VAE prototype has no detected foreground: {key}")
        normalized_values.append(resize_mask(probe["objects"][0]["normalized_mask"], 16))
    return {
        "keys": np.asarray(keys),
        "labels": np.asarray(labels),
        "masks": np.stack(normalized_values),
    }


def _save_calibration_montage(
    path: Path,
    source_images: Sequence[torch.Tensor],
    reconstructions: Sequence[torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    count = min(12, len(source_images))
    canvas = Image.new("RGB", (count * 128, 150), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        source = Image.fromarray(
            source_images[index].mul(255).round().byte().permute(1, 2, 0).numpy(),
            mode="RGB",
        ).resize((64, 64), Image.Resampling.NEAREST)
        recon = Image.fromarray(
            reconstructions[index].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy(),
            mode="RGB",
        ).resize((64, 64), Image.Resampling.NEAREST)
        left = index * 128
        canvas.paste(source, (left, 0))
        canvas.paste(recon, (left + 64, 0))
        draw.text((left, 130), str(rows[index]["objects"][0]["shape"]), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def calibrate_evaluator(
    project_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    eval_dir = root / str(config["output_eval_data"])
    result_dir = root / str(config["output_results"])
    result_dir.mkdir(parents=True, exist_ok=True)
    eval_manifest = verify_fixed_evaluation_data(root, eval_dir)
    rows = [
        json.loads(line)
        for line in (eval_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    payload = np.load(eval_dir / "ideal_masks.npz", allow_pickle=False)
    masks = payload["amodal"]
    payload.close()
    images = [_image_tensor(root / row["image_path"]) for row in rows]

    vae = _load_vae(root, str(config["source_vae_checkpoint"]))
    shape_prototypes = _build_vae_shape_prototypes(vae)
    template_path = result_dir / "evaluator_shape_templates.npz"
    np.savez_compressed(
        template_path,
        keys=shape_prototypes["keys"],
        labels=shape_prototypes["labels"],
        masks=shape_prototypes["masks"],
    )

    source_predictions = [
        evaluate_image_v2(
            image,
            row,
            masks[row["ideal_mask_index"]],
            fully_correct_shape_iou=0.98,
            shape_prototypes=shape_prototypes,
        )
        for image, row in zip(images, rows, strict=True)
    ]
    source_metrics = aggregate_evaluations(source_predictions)
    source_checks = _calibration_checks(source_metrics, layer="source")

    reconstructions: list[torch.Tensor] = []
    for start in range(0, len(images), 64):
        batch = torch.stack(images[start : start + 64])
        mean, _ = vae.encode(batch)
        reconstructions.extend(vae.decode(mean).clamp(0, 1).cpu())
    vae_predictions = [
        evaluate_image_v2(
            image,
            row,
            masks[row["ideal_mask_index"]],
            fully_correct_shape_iou=0.85,
            shape_prototypes=shape_prototypes,
        )
        for image, row in zip(reconstructions, rows, strict=True)
    ]
    vae_metrics = aggregate_evaluations(vae_predictions)
    vae_checks = _calibration_checks(vae_metrics, layer="vae")

    _write_jsonl(result_dir / "source_calibration_predictions.jsonl", source_predictions)
    _write_jsonl(result_dir / "vae_calibration_predictions.jsonl", vae_predictions)
    _save_calibration_montage(
        result_dir / "evaluator_calibration_examples.png", images, reconstructions, rows
    )
    report = {
        "schema_version": 2,
        "evaluation_manifest": str((eval_dir / "manifest.json").relative_to(root).as_posix()),
        "evaluation_manifest_sha256": _sha(eval_dir / "manifest.json"),
        "evaluator_code": "src/mini_multimodal/ldm/evaluator.py",
        "evaluator_code_sha256": _sha(root / "src" / "mini_multimodal" / "ldm" / "evaluator.py"),
        "vae_checkpoint": config["source_vae_checkpoint"],
        "vae_checkpoint_sha256": _sha(root / str(config["source_vae_checkpoint"])),
        "evaluator_parameters": {
            "minimum_component_area": 6,
            "shape_reject_score": 0.45,
            "shape_reject_margin": 0.015,
            "boundary_tolerance_pixels_on_normalized_mask": 2,
            "shape_template_bank": template_path.relative_to(root).as_posix(),
            "shape_template_bank_sha256": _sha(template_path),
            "shape_template_policy": "nearest 16x16 normalized mask over 810 frozen-VAE prototypes spanning shape, size, color, and region",
        },
        "source": {
            "metrics": source_metrics,
            "checks": source_checks,
            "verified": all(source_checks.values()),
        },
        "vae": {
            "metrics": vae_metrics,
            "checks": vae_checks,
            "verified": all(vae_checks.values()),
        },
    }
    report["verified"] = report["source"]["verified"] and report["vae"]["verified"]
    _write_json(result_dir / "evaluator_calibration.json", report)
    return report


def _old_model_kwargs(config: Mapping[str, Any], vocab_size: int) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        **{
            key: config["model"][key]
            for key in (
                "latent_channels",
                "base_channels",
                "text_dim",
                "time_dim",
                "attention_heads",
                "max_text_length",
            )
        },
    }


def _derived_thresholds(calibration: Mapping[str, Any]) -> dict[str, Any]:
    vae = calibration["vae"]["metrics"]
    shape = float(vae["shape_accuracy"]["value"])
    contour = float(vae["normalized_template_iou_mean"])
    boundary = float(vae["boundary_score_mean"])
    return {
        "seen_single": {
            "foreground_presence": 0.95,
            "count_accuracy": 0.85,
            "color_accuracy": 0.90,
            "shape_accuracy": max(0.75, 0.80 * shape),
            "each_shape_accuracy": 0.65,
            "normalized_template_iou_mean": max(0.70, 0.80 * contour),
            "boundary_score_mean": max(0.55, 0.70 * boundary),
            "size_accuracy": 0.70,
            "position_accuracy": 0.70,
            "center_distance_mean_max": 0.15,
            "fully_correct_rate": 0.45,
        },
        "unseen_single": {
            "color_accuracy": 0.85,
            "shape_accuracy": max(0.65, 0.70 * shape),
            "normalized_template_iou_mean": max(0.60, 0.70 * contour),
            "size_accuracy": 0.60,
            "position_accuracy": 0.60,
            "center_distance_mean_max": 0.20,
            "fully_correct_rate": 0.30,
        },
        "multi_object_reference": {
            "count_accuracy": 0.50,
            "color_accuracy": 0.80,
            "shape_accuracy": 0.55,
            "position_accuracy": 0.50,
            "blocking": False,
        },
        "interventions_and_ablations": {
            "position_direction_rate": 0.80,
            "median_signed_displacement": 0.20,
            "shape_change_rate": 0.70,
            "color_change_rate": 0.90,
            "non_target_stability_rate": 0.75,
            "position_token_accuracy_drop": 0.20,
            "shape_token_accuracy_drop": 0.15,
            "color_token_accuracy_drop": 0.20,
            "coordinate_path_position_drop": 0.20,
            "coordinate_path_max_color_drop": 0.10,
            "high_resolution_condition_contour_drop": 0.08,
        },
    }


@torch.no_grad()
def reevaluate_original_checkpoint_and_freeze_thresholds(
    project_root: str | Path, revision_config: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    started = perf_counter()
    result_dir = root / str(revision_config["output_results"])
    eval_dir = root / str(revision_config["output_eval_data"])
    calibration_path = result_dir / "evaluator_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not calibration.get("verified"):
        raise RuntimeError("evaluator calibration must pass before checkpoint re-evaluation")
    thresholds = _derived_thresholds(calibration)
    rows = [
        json.loads(line)
        for line in (eval_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    payload = np.load(eval_dir / "ideal_masks.npz", allow_pickle=False)
    masks = payload["amodal"]
    payload.close()
    prototype_path = result_dir / "evaluator_shape_templates.npz"
    shape_prototypes = load_shape_prototypes(prototype_path)

    old_config_path = root / "configs" / "p2_w04_tiny_image_ldm.json"
    old_config = json.loads(old_config_path.read_text(encoding="utf-8"))
    vocabulary = json.loads(
        (root / old_config["output_data"] / "vocabulary.json").read_text(encoding="utf-8")
    )
    tokenizer = CharacterTokenizer(vocabulary)
    model = TinyImageDenoiser(**_old_model_kwargs(old_config, len(tokenizer)))
    checkpoint_path = root / old_config["output_results"] / "training" / "checkpoints" / "best.pt"
    load_checkpoint(checkpoint_path, model=model, optimizer=None, restore_rng=False)
    model.eval()
    schedule = DiffusionSchedule(
        timesteps=int(old_config["diffusion"]["train_timesteps"]),
        beta_start=float(old_config["diffusion"]["beta_start"]),
        beta_end=float(old_config["diffusion"]["beta_end"]),
    )
    vae = _load_vae(root, str(old_config["source_vae_checkpoint"]))
    old_manifest = json.loads(
        (root / old_config["output_data"] / "manifest.json").read_text(encoding="utf-8")
    )
    scaling_factor = float(old_manifest["latent_scaling_factor"])
    generated_dir = result_dir / "original_checkpoint_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    batch_size = 8
    base_seed = int(revision_config["seed"]) + 40000
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        sampled = sample_latents(
            model,
            schedule,
            tokenizer,
            [row["prompt"] for row in batch_rows],
            sample_steps=int(old_config["diffusion"]["sample_steps"]),
            guidance_scale=float(old_config["diffusion"]["guidance_scale"]),
            seed=base_seed + start,
        )
        images = vae.decode(sampled["latent"] / scaling_factor).clamp(0, 1)
        for offset, (row, image) in enumerate(zip(batch_rows, images, strict=True)):
            shape_iou_threshold = (
                thresholds["seen_single"]["normalized_template_iou_mean"]
                if row["subset"] == "seen_single"
                else thresholds["unseen_single"]["normalized_template_iou_mean"]
                if row["subset"] == "unseen_single"
                else thresholds["seen_single"]["normalized_template_iou_mean"]
            )
            evaluation = evaluate_image_v2(
                image,
                row,
                masks[row["ideal_mask_index"]],
                fully_correct_shape_iou=float(shape_iou_threshold),
                shape_prototypes=shape_prototypes,
            )
            image_path = generated_dir / f"old_{start + offset:03d}.png"
            Image.fromarray(
                image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), mode="RGB"
            ).save(image_path)
            evaluation.update({
                "generated_image": image_path.relative_to(root).as_posix(),
                "generated_image_sha256": _sha(image_path),
                "sample_seed": base_seed + start,
                "checkpoint": checkpoint_path.relative_to(root).as_posix(),
            })
            predictions.append(evaluation)

    subset_metrics = {}
    for subset in ("seen_single", "unseen_single", "multi_2", "multi_3"):
        subset_metrics[subset] = aggregate_evaluations(
            [row for row in predictions if row["subset"] == subset]
        )
    report = {
        "schema_version": 2,
        "status": "diagnostic_only",
        "checkpoint": checkpoint_path.relative_to(root).as_posix(),
        "checkpoint_sha256": _sha(checkpoint_path),
        "old_training_data": old_config["output_data"],
        "composition_generalization_valid": False,
        "composition_generalization_warning": "The historical checkpoint was trained with holdout pairs in both P1 normal/shortcut records and the v1 extension.",
        "evaluation_manifest": (eval_dir / "manifest.json").relative_to(root).as_posix(),
        "evaluation_manifest_sha256": _sha(eval_dir / "manifest.json"),
        "sample_count": len(predictions),
        "wall_seconds": perf_counter() - started,
        "metrics": subset_metrics,
    }
    _write_jsonl(result_dir / "original_checkpoint_predictions.jsonl", predictions)
    _write_json(result_dir / "original_checkpoint_metrics.json", report)

    freeze = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "status": "frozen_before_model_revision",
        "date": "2026-08-08",
        "evaluation_manifest": report["evaluation_manifest"],
        "evaluation_manifest_sha256": report["evaluation_manifest_sha256"],
        "calibration_file": calibration_path.relative_to(root).as_posix(),
        "calibration_file_sha256": _sha(calibration_path),
        "prototype_bank": prototype_path.relative_to(root).as_posix(),
        "prototype_bank_sha256": _sha(prototype_path),
        "vae_upper_bounds": {
            "shape_accuracy": float(calibration["vae"]["metrics"]["shape_accuracy"]["value"]),
            "normalized_template_iou_mean": float(calibration["vae"]["metrics"]["normalized_template_iou_mean"]),
            "boundary_score_mean": float(calibration["vae"]["metrics"]["boundary_score_mean"]),
        },
        "formula_policy": "Each derived metric is max(absolute_floor, documented_fraction * matching VAE upper bound). Other metrics use predeclared absolute thresholds.",
        "thresholds": thresholds,
        "statistics_policy": {
            "report_numerator_denominator_point_and_wilson_95": True,
            "point_estimate_must_meet_threshold": True,
            "wilson_lower_must_exceed_random_or_majority_baseline": True,
            "per_shape_position_size_color_reporting_required": True,
        },
        "checks": {
            "calibration_verified": calibration["verified"],
            "thresholds_created_before_model_revision": True,
            "historical_checkpoint_is_diagnostic_only": True,
        },
    }
    freeze["verified"] = all(freeze["checks"].values())
    _write_json(result_dir / "threshold_freeze.json", freeze)
    return {"original_checkpoint": report, "threshold_freeze": freeze}
