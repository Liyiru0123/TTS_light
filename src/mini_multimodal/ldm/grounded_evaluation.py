"""Frozen generation evaluation for the second-round grounded model."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from mini_multimodal.training import load_checkpoint

from .data import CharacterTokenizer
from .evaluator import aggregate_evaluations, evaluate_image_v2, load_shape_prototypes
from .grounded_training import grounded_model_from_config
from .inference import load_vae_decoder
from .revision_evaluation import _fixed_checks, _sha, _write_json, _write_jsonl, sample_latents_v2
from .revision_training import load_config, schedule_from_config


def load_grounded_checkpoint(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    vocabulary = json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocabulary)
    model = grounded_model_from_config(config, tokenizer)
    checkpoint = root / config["output_results"] / "training_grounded/checkpoints/best.pt"
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


@torch.no_grad()
def evaluate_grounded_fixed(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result_dir = root / config["output_results"]
    eval_dir = root / config["output_eval_data"]
    freeze = json.loads((result_dir / "threshold_freeze.json").read_text(encoding="utf-8"))
    if _sha(eval_dir / "manifest.json") != freeze["evaluation_manifest_sha256"]:
        raise RuntimeError("fixed evaluation manifest changed after threshold freeze")
    holdout = json.loads((root / config["output_data"] / "holdout_audit.json").read_text(encoding="utf-8"))
    if not holdout.get("verified") or holdout["final_training_holdout_object_count"] != 0:
        raise RuntimeError("grounded evaluation requires zero held-out training objects")
    rows = [json.loads(line) for line in (eval_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    payload = np.load(eval_dir / "ideal_masks.npz", allow_pickle=False)
    masks = payload["amodal"]
    payload.close()
    prototypes = load_shape_prototypes(result_dir / "evaluator_shape_templates.npz")
    loaded = load_grounded_checkpoint(root, config)
    scaling = float(json.loads((root / config["output_data"] / "manifest.json").read_text(encoding="utf-8"))["latent_scaling_factor"])
    generated_dir = result_dir / "generated_grounded"
    generated_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    predictions: list[dict[str, Any]] = []
    base_seed = int(config["seed"]) + 60000
    for start in range(0, len(rows), 8):
        batch_rows = rows[start : start + 8]
        sampled = sample_latents_v2(
            loaded["model"],
            loaded["schedule"],
            loaded["tokenizer"],
            [row["prompt"] for row in batch_rows],
            sample_steps=int(config["diffusion"]["sample_steps"]),
            guidance_scale=float(config["diffusion"]["guidance_scale"]),
            seed=base_seed + start,
        )
        images = loaded["vae"].decode(sampled["latent"] / scaling).clamp(0, 1)
        for offset, (row, image) in enumerate(zip(batch_rows, images, strict=True)):
            threshold = freeze["thresholds"]["unseen_single" if row["subset"] == "unseen_single" else "seen_single"]
            evaluation = evaluate_image_v2(
                image,
                row,
                masks[row["ideal_mask_index"]],
                fully_correct_shape_iou=float(threshold["normalized_template_iou_mean"]),
                shape_prototypes=prototypes,
            )
            path = generated_dir / f"grounded_{start + offset:03d}.png"
            Image.fromarray(image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), mode="RGB").save(path)
            evaluation.update({
                "generated_image": path.relative_to(root).as_posix(),
                "generated_image_sha256": _sha(path),
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
    reference = freeze["thresholds"]["multi_object_reference"]
    multi_checks = {
        subset: {
            "count_accuracy": _value(metrics[subset]["count_accuracy"]) >= float(reference["count_accuracy"]),
            "color_accuracy": _value(metrics[subset]["color_accuracy"]) >= float(reference["color_accuracy"]),
            "shape_accuracy": _value(metrics[subset]["shape_accuracy"]) >= float(reference["shape_accuracy"]),
            "position_accuracy": _value(metrics[subset]["position_accuracy"]) >= float(reference["position_accuracy"]),
        }
        for subset in ("multi_2", "multi_3")
    }
    report = {
        "schema_version": 2,
        "work_package": "P2-W04",
        "attempt": "text_derived_object_grounding_v2",
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
    _write_jsonl(result_dir / "grounded_generation_predictions.jsonl", predictions)
    _write_json(result_dir / "grounded_generation_metrics.json", report)
    return report


def main_grounded(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    return evaluate_grounded_fixed(root, load_config(root))
