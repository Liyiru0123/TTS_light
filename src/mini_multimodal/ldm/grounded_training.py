"""Second-round training with text-derived object grounding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import psutil
import torch
from PIL import Image
from torch.utils.data import DataLoader

from mini_multimodal.training import Trainer, TrainerConfig, load_checkpoint, set_deterministic_seed

from .data import CharacterTokenizer, LatentTextDataset, collate_latent_text, verify_p2_w04_data
from .evaluator import load_shape_prototypes, probe_image_v2
from .inference import load_vae_decoder
from .model_grounded import TinyImageDenoiserGrounded, resolve_grounding_token_ids
from .model_v2 import resolve_spatial_token_coordinates
from .revision_evaluation import sample_latents_v2
from .revision_training import (
    _loader,
    _state_sha,
    _task,
    evaluate_denoising_v2,
    load_config,
    schedule_from_config,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_horizontal_triplet(dataset: LatentTextDataset) -> list[dict[str, Any]]:
    required = ("center_left", "center", "center_right")
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in dataset.rows:
        if len(row["objects"]) != 1 or row.get("source") != "p2_w04_extension":
            continue
        obj = row["objects"][0]
        if obj["position_region"] not in required:
            continue
        key = (obj["color"], obj["shape"], obj["size"])
        groups.setdefault(key, {}).setdefault(obj["position_region"], row)
    for key in sorted(groups):
        if all(region in groups[key] for region in required):
            return [groups[key][region] for region in required]
    raise RuntimeError("clean v2 extension has no same-attribute horizontal triplet")


def grounded_model_from_config(config: Mapping[str, Any], tokenizer: CharacterTokenizer) -> TinyImageDenoiserGrounded:
    keys = ("latent_channels", "base_channels", "text_dim", "time_dim", "attention_heads", "max_text_length")
    return TinyImageDenoiserGrounded(
        vocab_size=len(tokenizer),
        **{key: config["model"][key] for key in keys},
        spatial_token_coordinates=resolve_spatial_token_coordinates(tokenizer.token_to_id),
        grounding_token_ids=resolve_grounding_token_ids(tokenizer.token_to_id),
        max_objects=3,
    )


def _trainer(
    output: Path,
    config: Mapping[str, Any],
    tokenizer: CharacterTokenizer,
    *,
    epochs: int,
    high_noise_training_fraction: float | None = None,
) -> tuple[TinyImageDenoiserGrounded, torch.optim.Optimizer, Trainer]:
    model = grounded_model_from_config(config, tokenizer)
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


def _shape_quintet(dataset: LatentTextDataset) -> list[dict[str, Any]]:
    required = {"circle", "square", "triangle", "rectangle", "pentagon"}
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in dataset.rows:
        if len(row["objects"]) != 1 or row.get("source") != "p2_w04_extension":
            continue
        obj = row["objects"][0]
        key = (obj["color"], obj["size"], obj["position_region"])
        groups.setdefault(key, {}).setdefault(obj["shape"], row)
    for key in sorted(groups):
        if required <= set(groups[key]):
            order = ("circle", "square", "triangle", "rectangle", "pentagon")
            return [groups[key][shape] for shape in order]
    raise RuntimeError("v2 training view has no same-attribute five-shape group")


def _overfit_ids(dataset: LatentTextDataset, count: int) -> list[str]:
    selected = _clean_horizontal_triplet(dataset) + _shape_quintet(dataset)
    unique: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in selected:
        if row["record_id"] not in used:
            unique.append(row)
            used.add(row["record_id"])
    regions = {row["objects"][0]["position_region"] for row in unique}
    singles = sorted(
        (row for row in dataset.rows if len(row["objects"]) == 1 and row.get("source") == "p2_w04_extension"),
        key=lambda row: (row["objects"][0]["position_region"], row["record_id"]),
    )
    for row in singles:
        region = row["objects"][0]["position_region"]
        if region not in regions and row["record_id"] not in used:
            unique.append(row)
            used.add(row["record_id"])
            regions.add(region)
    unique.extend(row for row in singles if row["record_id"] not in used)
    return [row["record_id"] for row in unique[:count]]


@torch.no_grad()
def _semantic_probes(
    root: Path,
    result_dir: Path,
    config: Mapping[str, Any],
    dataset: LatentTextDataset,
    model: TinyImageDenoiserGrounded,
    *,
    probe_subdir: str = "grounded_overfit/spatial_shape_probe",
) -> dict[str, Any]:
    vae = load_vae_decoder(root, config["source_vae_checkpoint"])
    prototypes = load_shape_prototypes(result_dir / "evaluator_shape_templates.npz")
    schedule = schedule_from_config(config)
    probe_dir = result_dir / probe_subdir
    probe_dir.mkdir(parents=True, exist_ok=True)

    horizontal = _clean_horizontal_triplet(dataset)
    shape_rows = _shape_quintet(dataset)
    groups = (("horizontal", horizontal), ("shapes", shape_rows))
    report: dict[str, Any] = {}
    for group_name, rows in groups:
        seed = int(config["seed"]) + (700 if group_name == "horizontal" else 701)
        initial = torch.randn((1, int(config["model"]["latent_channels"]), 16, 16), generator=torch.Generator().manual_seed(seed)).repeat(len(rows), 1, 1, 1)
        sampled = sample_latents_v2(
            model,
            schedule,
            dataset.tokenizer,
            [row["prompt"] for row in rows],
            sample_steps=int(config["diffusion"]["sample_steps"]),
            guidance_scale=float(config["diffusion"]["guidance_scale"]),
            seed=seed,
            initial_latent=initial,
        )
        images = vae.decode(sampled["latent"] / dataset.scaling_factor).clamp(0, 1)
        items: list[dict[str, Any]] = []
        for index, (row, image) in enumerate(zip(rows, images, strict=True)):
            probe = probe_image_v2(image, shape_prototypes=prototypes)
            path = probe_dir / f"{group_name}_{index}.png"
            Image.fromarray(image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), mode="RGB").save(path)
            first = None if not probe["objects"] else probe["objects"][0]
            items.append({
                "record_id": row["record_id"],
                "prompt": row["prompt"],
                "target_shape": row["objects"][0]["shape"],
                "target_region": row["objects"][0]["position_region"],
                "predicted_shape": None if first is None else first["shape"],
                "shape_rejected": True if first is None else bool(first["shape_rejected"]),
                "predicted_center_xy": None if first is None else first["center_xy"],
                "predicted_object_count": int(probe["predicted_object_count"]),
                "generated_image": path.relative_to(root).as_posix(),
            })
        report[group_name] = {"seed": seed, "items": items}
    x = [item["predicted_center_xy"] for item in report["horizontal"]["items"]]
    report["horizontal"]["left_to_right_monotonic"] = all(value is not None for value in x) and bool(x[0][0] < x[1][0] < x[2][0])
    shape_items = report["shapes"]["items"]
    report["shapes"]["correct_count"] = sum(
        item["predicted_shape"] == item["target_shape"] and not item["shape_rejected"]
        for item in shape_items
    )
    report["shapes"]["unique_predicted_shapes"] = len({item["predicted_shape"] for item in shape_items if item["predicted_shape"] is not None})
    return report


def run_grounded_overfit(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result_dir = root / config["output_results"]
    seed = int(config["seed"]) + 2
    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))
    train = LatentTextDataset(root, data_dir=config["output_data"], split="train", max_text_length=int(config["model"]["max_text_length"]))
    ids = _overfit_ids(train, int(config["training"]["overfit_count"]))
    subset = LatentTextDataset(root, data_dir=config["output_data"], record_ids=ids, max_text_length=int(config["model"]["max_text_length"]))
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, collate_fn=collate_latent_text)
    model, _, trainer = _trainer(
        result_dir / "grounded_overfit",
        config,
        subset.tokenizer,
        epochs=int(config["training"]["overfit_epochs"]),
        high_noise_training_fraction=float(config["training"]["grounded_overfit_high_noise_training_fraction"]),
    )
    started = perf_counter()
    history = trainer.fit(loader, loader)
    probes = _semantic_probes(root, result_dir, config, subset, model)
    initial = float(history[0]["train"]["noise_mse"])
    final = float(history[-1]["train"]["noise_mse"])
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "acceptance_item_id": "COMP-LDM-001",
        "attempt": "text_derived_object_grounding_v2",
        "record_ids": ids,
        "sample_count": len(subset),
        "initial_noise_mse": initial,
        "final_noise_mse": final,
        "loss_ratio": final / initial,
        "zero_prediction_mse": float(history[-1]["train"]["zero_prediction_mse"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_sha256": _state_sha(model),
        "wall_seconds": perf_counter() - started,
        "semantic_probes": probes,
    }
    report["checks"] = {
        "loss_ratio": report["loss_ratio"] <= float(config["training_gates"]["overfit_loss_ratio"]),
        "below_zero_prediction": final < report["zero_prediction_mse"],
        "parameter_limit": report["parameter_count"] < int(config["training_gates"]["max_parameters"]),
        "same_noise_position_direction": bool(probes["horizontal"]["left_to_right_monotonic"]),
        "same_noise_shape_correct_at_least_3_of_5": int(probes["shapes"]["correct_count"]) >= 3,
        "same_noise_shape_diversity_at_least_3": int(probes["shapes"]["unique_predicted_shapes"]) >= 3,
    }
    report["verified_for_full_training"] = all(report["checks"].values())
    _write_json(result_dir / "grounded_overfit_report.json", report)
    if not report["verified_for_full_training"]:
        raise RuntimeError("P2-W04 grounded overfit gate failed")
    return report


def run_grounded_training(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result_dir = root / config["output_results"]
    overfit = json.loads((result_dir / "grounded_overfit_report.json").read_text(encoding="utf-8"))
    if not overfit.get("verified_for_full_training"):
        raise RuntimeError("grounded full training requires a passing overfit gate")
    readiness = verify_p2_w04_data(root, config["output_data"])
    holdout = json.loads((root / config["output_data"] / "holdout_audit.json").read_text(encoding="utf-8"))
    freeze = json.loads((result_dir / "threshold_freeze.json").read_text(encoding="utf-8"))
    if not holdout.get("verified") or holdout["final_training_holdout_object_count"] != 0 or not freeze.get("verified"):
        raise RuntimeError("data or threshold precondition failed")

    seed = int(config["seed"]) + 100
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
    loader = _loader(test, config, shuffle=False, seed=seed)
    schedule = schedule_from_config(config)
    core = evaluate_denoising_v2(model, schedule, loader, seed=seed + 101)
    ablations = {
        "disable_object_grounding": evaluate_denoising_v2(model, schedule, loader, seed=seed + 101, forward_options={"disable_object_grounding": True}),
        "disable_coordinates": evaluate_denoising_v2(model, schedule, loader, seed=seed + 101, forward_options={"disable_coordinates": True}),
        "disable_16x16_cross_attention": evaluate_denoising_v2(model, schedule, loader, seed=seed + 101, forward_options={"disable_cross_attention_16": True}),
        "disable_all_cross_attention": evaluate_denoising_v2(model, schedule, loader, seed=seed + 101, forward_options={"disable_cross_attention": True}),
    }
    elapsed = perf_counter() - started
    memory = process.memory_info()
    peak_rss = int(getattr(memory, "peak_wset", memory.rss))
    metrics = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "attempt": "text_derived_object_grounding_v2",
        "acceptance_item_id": "MODEL-LDM-001",
        "checkpoint": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": _sha(checkpoint),
        "best_epoch": int(best["epoch"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "core": core,
        "ablations": ablations,
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
    _write_json(result_dir / "grounded_denoising_metrics.json", metrics)
    if not metrics["verified_for_generation_evaluation"]:
        raise RuntimeError("P2-W04 grounded denoising gate failed")
    return metrics
