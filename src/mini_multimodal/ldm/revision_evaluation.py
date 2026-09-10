"""Frozen-set generation evaluation for the P2-W04 revised checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from mini_multimodal.training import load_checkpoint

from .data import CharacterTokenizer
from .evaluator import aggregate_evaluations, evaluate_image_v2, load_shape_prototypes
from .inference import _text_batch, load_vae_decoder
from .revision_training import load_config, model_from_config, schedule_from_config


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def sample_latents_v2(
    model: torch.nn.Module,
    schedule: Any,
    tokenizer: CharacterTokenizer,
    prompts: Sequence[str],
    *,
    sample_steps: int,
    guidance_scale: float,
    seed: int,
    initial_latent: torch.Tensor | None = None,
    forward_options: Mapping[str, bool] | None = None,
    keep_trajectory: bool = False,
) -> dict[str, Any]:
    ids, mask, null_ids, null_mask = _text_batch(prompts, tokenizer, model.max_text_length)
    generator = torch.Generator().manual_seed(int(seed))
    latent = (
        torch.randn((len(prompts), model.latent_channels, 16, 16), generator=generator)
        if initial_latent is None
        else initial_latent.clone()
    )
    if latent.shape != (len(prompts), model.latent_channels, 16, 16):
        raise ValueError("initial latent shape mismatch")
    options = dict(forward_options or {})
    trajectory = [latent.clone()] if keep_trajectory else []
    attentions: list[dict[str, torch.Tensor]] = []
    timesteps = schedule.sampling_timesteps(sample_steps)
    model.eval()
    for index, current in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        step = torch.full((len(prompts),), current, dtype=torch.long)
        conditional = model(latent, step, ids, mask, **options)
        unconditional = model(latent, step, null_ids, null_mask, **options)
        predicted = unconditional.predicted_noise + guidance_scale * (
            conditional.predicted_noise - unconditional.predicted_noise
        )
        latent = schedule.ddim_step(latent, step, torch.full_like(step, previous), predicted)
        if keep_trajectory:
            trajectory.append(latent.clone())
            attentions.append({
                "16x16": conditional.cross_attention_16.clone(),
                "8x8": conditional.cross_attention_8.clone(),
            })
    return {"latent": latent, "latent_trajectory": trajectory, "attention_trajectory": attentions, "timesteps": timesteps}


def load_revision_checkpoint(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    vocabulary = json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocabulary)
    model = model_from_config(config, tokenizer)
    checkpoint = root / config["output_results"] / "checkpoints/best.pt"
    load_checkpoint(checkpoint, model=model, optimizer=None, restore_rng=False)
    model.eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "checkpoint": checkpoint,
        "schedule": schedule_from_config(config),
        "vae": load_vae_decoder(root, config["source_vae_checkpoint"]),
    }


def _value(metric: Any) -> float:
    return float(metric["value"] if isinstance(metric, Mapping) and "value" in metric else metric)


def _fixed_checks(metrics: Mapping[str, Any], thresholds: Mapping[str, Any], subset: str) -> dict[str, bool]:
    checks = {
        "color_accuracy": _value(metrics["color_accuracy"]) >= float(thresholds["color_accuracy"]),
        "shape_accuracy": _value(metrics["shape_accuracy"]) >= float(thresholds["shape_accuracy"]),
        "normalized_template_iou_mean": _value(metrics["normalized_template_iou_mean"]) >= float(thresholds["normalized_template_iou_mean"]),
        "size_accuracy": _value(metrics["size_accuracy"]) >= float(thresholds["size_accuracy"]),
        "position_accuracy": _value(metrics["position_accuracy"]) >= float(thresholds["position_accuracy"]),
        "center_distance_mean": _value(metrics["center_distance_mean"]) <= float(thresholds["center_distance_mean_max"]),
        "fully_correct_rate": _value(metrics["fully_correct_rate"]) >= float(thresholds["fully_correct_rate"]),
    }
    if subset == "seen_single":
        checks.update({
            "foreground_presence": _value(metrics["foreground_presence"]) >= float(thresholds["foreground_presence"]),
            "count_accuracy": _value(metrics["count_accuracy"]) >= float(thresholds["count_accuracy"]),
            "boundary_score_mean": _value(metrics["boundary_score_mean"]) >= float(thresholds["boundary_score_mean"]),
            "each_shape_accuracy": all(
                _value(group["shape_accuracy"]) >= float(thresholds["each_shape_accuracy"])
                for group in metrics["shape_groups"].values()
            ),
        })
    return checks


@torch.no_grad()
def evaluate_fixed_generation(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result_dir = root / config["output_results"]
    eval_dir = root / config["output_eval_data"]
    freeze = json.loads((result_dir / "threshold_freeze.json").read_text(encoding="utf-8"))
    if _sha(eval_dir / "manifest.json") != freeze["evaluation_manifest_sha256"]:
        raise RuntimeError("fixed evaluation manifest changed after threshold freeze")
    holdout = json.loads((root / config["output_data"] / "holdout_audit.json").read_text(encoding="utf-8"))
    if not holdout.get("verified") or holdout["final_training_holdout_object_count"] != 0:
        raise RuntimeError("unseen-composition evaluation lacks a clean training-side audit")
    rows = [json.loads(line) for line in (eval_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    mask_payload = np.load(eval_dir / "ideal_masks.npz", allow_pickle=False)
    ideal_masks = mask_payload["amodal"]
    mask_payload.close()
    prototypes = load_shape_prototypes(result_dir / "evaluator_shape_templates.npz")
    loaded = load_revision_checkpoint(root, config)
    model, tokenizer, schedule, vae = loaded["model"], loaded["tokenizer"], loaded["schedule"], loaded["vae"]
    scaling = float(json.loads((root / config["output_data"] / "manifest.json").read_text(encoding="utf-8"))["latent_scaling_factor"])
    generated_dir = result_dir / "generated_v2"
    generated_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    predictions: list[dict[str, Any]] = []
    batch_size = 8
    base_seed = int(config["seed"]) + 60000
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        sampled = sample_latents_v2(
            model,
            schedule,
            tokenizer,
            [row["prompt"] for row in batch_rows],
            sample_steps=int(config["diffusion"]["sample_steps"]),
            guidance_scale=float(config["diffusion"]["guidance_scale"]),
            seed=base_seed + start,
        )
        images = vae.decode(sampled["latent"] / scaling).clamp(0, 1)
        for offset, (row, image) in enumerate(zip(batch_rows, images, strict=True)):
            subset_thresholds = freeze["thresholds"]["unseen_single" if row["subset"] == "unseen_single" else "seen_single"]
            evaluation = evaluate_image_v2(
                image,
                row,
                ideal_masks[row["ideal_mask_index"]],
                fully_correct_shape_iou=float(subset_thresholds["normalized_template_iou_mean"]),
                shape_prototypes=prototypes,
            )
            image_path = generated_dir / f"revised_{start + offset:03d}.png"
            Image.fromarray(image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), mode="RGB").save(image_path)
            evaluation.update({
                "generated_image": image_path.relative_to(root).as_posix(),
                "generated_image_sha256": _sha(image_path),
                "sample_seed": base_seed + start,
                "checkpoint": loaded["checkpoint"].relative_to(root).as_posix(),
            })
            predictions.append(evaluation)

    metrics = {
        subset: aggregate_evaluations([row for row in predictions if row["subset"] == subset])
        for subset in ("seen_single", "unseen_single", "multi_2", "multi_3")
    }
    seen_checks = _fixed_checks(metrics["seen_single"], freeze["thresholds"]["seen_single"], "seen_single")
    unseen_checks = _fixed_checks(metrics["unseen_single"], freeze["thresholds"]["unseen_single"], "unseen_single")
    multi_reference = freeze["thresholds"]["multi_object_reference"]
    multi_checks = {
        subset: {
            "count_accuracy": _value(metrics[subset]["count_accuracy"]) >= float(multi_reference["count_accuracy"]),
            "color_accuracy": _value(metrics[subset]["color_accuracy"]) >= float(multi_reference["color_accuracy"]),
            "shape_accuracy": _value(metrics[subset]["shape_accuracy"]) >= float(multi_reference["shape_accuracy"]),
            "position_accuracy": _value(metrics[subset]["position_accuracy"]) >= float(multi_reference["position_accuracy"]),
        }
        for subset in ("multi_2", "multi_3")
    }
    report = {
        "schema_version": 2,
        "work_package": "P2-W04",
        "checkpoint": loaded["checkpoint"].relative_to(root).as_posix(),
        "checkpoint_sha256": _sha(loaded["checkpoint"]),
        "training_view": str(config["output_data"]),
        "training_holdout_object_count": int(holdout["final_training_holdout_object_count"]),
        "composition_generalization_valid": True,
        "evaluation_manifest": (eval_dir / "manifest.json").relative_to(root).as_posix(),
        "evaluation_manifest_sha256": _sha(eval_dir / "manifest.json"),
        "sample_count": len(predictions),
        "guidance_scale": float(config["diffusion"]["guidance_scale"]),
        "sample_steps": int(config["diffusion"]["sample_steps"]),
        "wall_seconds": perf_counter() - started,
        "metrics": metrics,
        "checks": {
            "seen_single": seen_checks,
            "unseen_single": unseen_checks,
            "multi_object_nonblocking_reference": multi_checks,
        },
        "seen_single_verified": all(seen_checks.values()),
        "unseen_single_verified": all(unseen_checks.values()),
        "multi_object_reference_blocking": False,
    }
    report["core_generation_verified"] = report["seen_single_verified"] and report["unseen_single_verified"]
    _write_jsonl(result_dir / "generation_predictions.jsonl", predictions)
    _write_json(result_dir / "generation_metrics.json", report)
    return report


def main_fixed(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    return evaluate_fixed_generation(root, load_config(root))
