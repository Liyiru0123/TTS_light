"""Controlled P2-W04 I005 overfit, training, evaluation, and evidence pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import psutil
from PIL import Image
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from mini_multimodal.iteration_management import atomic_write_json, sha256_file
from mini_multimodal.training import (
    Trainer,
    TrainerConfig,
    load_checkpoint,
    seeded_generator,
    set_deterministic_seed,
)

from .data import CharacterTokenizer
from .evaluator import aggregate_evaluations, evaluate_image_v2, load_shape_prototypes, probe_image_v2
from .geometry_data import GeometryLatentTextDataset, collate_geometry_text
from .grounded_training import _clean_horizontal_triplet, _overfit_ids, _shape_quintet, grounded_model_from_config
from .inference import load_vae_decoder
from .refiner_inference import sample_latents_refined
from .refiner_training import CleanRefinerTask, load_i005_config, refiner_model_from_config
from .revision_evaluation import _fixed_checks, sample_latents_v2
from .revision_training import _task, load_config, schedule_from_config


ITERATION_ID = "I005"
BRANCH_IDS = ("continuation", "clean_refiner")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _value(metric: Any) -> float:
    return float(metric["value"] if isinstance(metric, Mapping) and "value" in metric else metric)


def _branch_dir(root: Path, branch_id: str) -> Path:
    if branch_id not in BRANCH_IDS:
        raise ValueError(f"unsupported I005 branch: {branch_id}")
    path = root / "results/P2-W04_tiny_image_ldm/iterations" / ITERATION_ID / "branches" / branch_id
    if not (path / "branch_manifest.json").is_file():
        raise RuntimeError(f"formal I005 branch is not prepared: {branch_id}")
    return path


def _dataset(
    root: Path,
    base: Mapping[str, Any],
    *,
    split: str | None = None,
    record_ids: Sequence[str] | None = None,
) -> GeometryLatentTextDataset:
    return GeometryLatentTextDataset(
        root,
        data_dir=base["output_data"],
        geometry_data_dir=base["geometry_auxiliary"]["data_dir"],
        split=split,
        record_ids=record_ids,
        max_text_length=int(base["model"]["max_text_length"]),
    )


def _loader(
    dataset: GeometryLatentTextDataset,
    base: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    seed: int,
    balanced: bool,
) -> DataLoader:
    sampler = None
    if balanced:
        single_weight = float(base["training"].get("single_object_sampling_weight", 1.0))
        weights = torch.tensor(
            [single_weight if len(row["objects"]) == 1 else 1.0 for row in dataset.rows],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=seeded_generator(seed),
        )
    return DataLoader(
        dataset,
        batch_size=int(spec["formal_control"]["batch_size"]),
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_geometry_text,
    )


def _refiner_task(
    root: Path,
    base: Mapping[str, Any],
    spec: Mapping[str, Any],
    tokenizer: CharacterTokenizer,
) -> CleanRefinerTask:
    geometry = base["geometry_auxiliary"]
    weights = spec["clean_refiner"]["loss_weights"]
    return CleanRefinerTask(
        load_vae_decoder(root, base["source_vae_checkpoint"]),
        latent_scaling_factor=float(_json(root / base["output_data"] / "manifest.json")["latent_scaling_factor"]),
        null_id=tokenizer.null_id,
        condition_dropout=float(spec["clean_refiner"]["condition_dropout"]),
        max_geometry_samples=int(spec["clean_refiner"]["max_geometry_samples_per_batch"]),
        latent_weight=float(weights["latent"]),
        foreground_weight=float(weights["foreground"]),
        boundary_weight=float(weights["boundary"]),
        residual_regularization_weight=float(weights["residual_regularization"]),
        soft_chroma_threshold=float(geometry["soft_chroma_threshold"]),
        soft_chroma_temperature=float(geometry["soft_chroma_temperature"]),
    )


def _fixed_task_metrics(
    model: torch.nn.Module,
    task: CleanRefinerTask,
    loader: DataLoader,
    *,
    seed: int,
) -> dict[str, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-6)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=task,
        config=TrainerConfig(epochs=1, monitor="loss", maximize=False),
        output_dir=Path("."),
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return trainer.evaluate(loader)


def _target_margin(probe: Mapping[str, Any], target_shape: str) -> float:
    if not probe["objects"]:
        return -1.0
    scores = probe["objects"][0]["shape_scores"]
    return float(scores[target_shape]) - max(float(value) for key, value in scores.items() if key != target_shape)


@torch.no_grad()
def _semantic_probe(
    root: Path,
    base: Mapping[str, Any],
    spec: Mapping[str, Any],
    dataset: GeometryLatentTextDataset,
    model: torch.nn.Module,
    output: Path,
    *,
    label: str,
) -> dict[str, Any]:
    rows = _clean_horizontal_triplet(dataset) + _shape_quintet(dataset)
    unique = {row["record_id"]: row for row in rows}
    rows = list(unique.values())
    seed = int(base["seed"]) + 9505
    initial = torch.randn(
        (1, int(base["model"]["latent_channels"]), 16, 16),
        generator=torch.Generator().manual_seed(seed),
    ).repeat(len(rows), 1, 1, 1)
    sampled = sample_latents_refined(
        model,
        dataset.tokenizer,
        [row["prompt"] for row in rows],
        sample_steps=int(spec["formal_sampling"]["sample_steps"]),
        guidance_scale=float(spec["formal_sampling"]["guidance_scale"]),
        seed=seed,
        initial_latent=initial,
    )
    vae = load_vae_decoder(root, base["source_vae_checkpoint"])
    images = vae.decode(sampled["latent"] / dataset.scaling_factor).clamp(0, 1)
    prototypes = load_shape_prototypes(root / spec["frozen_assets"]["shape_prototypes"])
    image_dir = output / label
    image_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for index, (row, image) in enumerate(zip(rows, images, strict=True)):
        target = row["objects"][0]
        probe = probe_image_v2(image, shape_prototypes=prototypes)
        first = None if not probe["objects"] else probe["objects"][0]
        image_path = image_dir / f"probe_{index:02d}.png"
        Image.fromarray(image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), mode="RGB").save(image_path)
        items.append({
            "record_id": row["record_id"],
            "prompt": row["prompt"],
            "target_shape": target["shape"],
            "target_region": target["position_region"],
            "predicted_object_count": int(probe["predicted_object_count"]),
            "fragment_count": max(0, int(probe["predicted_object_count"]) - 1),
            "predicted_shape": None if first is None else first["shape"],
            "shape_rejected": True if first is None else bool(first["shape_rejected"]),
            "target_shape_margin": _target_margin(probe, target["shape"]),
            "predicted_region": None if first is None else first["position_region"],
            "generated_image": image_path.relative_to(root).as_posix(),
            "generated_image_sha256": sha256_file(image_path),
        })
    summary = {
        "sample_count": len(items),
        "count_correct": sum(item["predicted_object_count"] == 1 for item in items),
        "fragment_count": sum(item["fragment_count"] for item in items),
        "accepted_target_shape_count": sum(
            item["predicted_shape"] == item["target_shape"] and not item["shape_rejected"] for item in items
        ),
        "mean_target_shape_margin": float(np.mean([item["target_shape_margin"] for item in items])),
        "position_correct": sum(item["predicted_region"] == item["target_region"] for item in items),
    }
    report = {"label": label, "seed": seed, "summary": summary, "items": items}
    atomic_write_json(output / f"{label}.json", report)
    return report


def _configure_trainer_paths(trainer: Trainer, checkpoint_dir: Path, log_path: Path) -> None:
    trainer.checkpoint_dir = checkpoint_dir
    trainer.log_path = log_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)


def run_clean_refiner_overfit(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    base, spec = load_config(root), load_i005_config(root)
    branch = _branch_dir(root, "clean_refiner")
    seed = int(base["seed"]) + 405
    set_deterministic_seed(seed, cpu_threads=int(base["cpu_threads"]))
    full = _dataset(root, base, split="train")
    record_ids = _overfit_ids(full, int(spec["clean_refiner"]["overfit_count"]))
    subset = _dataset(root, base, record_ids=record_ids)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, collate_fn=collate_geometry_text)
    model = refiner_model_from_config(root, base, spec, subset.tokenizer)
    task = _refiner_task(root, base, spec, subset.tokenizer)
    optimizer = torch.optim.AdamW(
        model.refiner.parameters(),
        lr=float(spec["clean_refiner"]["learning_rate"]),
        weight_decay=float(spec["clean_refiner"]["weight_decay"]),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=task,
        config=TrainerConfig(
            epochs=int(spec["clean_refiner"]["overfit_epochs"]),
            gradient_clip_norm=float(base["training"]["gradient_clip_norm"]),
            monitor="loss",
            maximize=False,
        ),
        output_dir=branch,
    )
    _configure_trainer_paths(trainer, branch / "checkpoints/overfit", branch / "logs/overfit_training_log.jsonl")
    process = psutil.Process()
    before = _fixed_task_metrics(model, task, loader, seed=seed + 1)
    initial_probe = _semantic_probe(root, base, spec, subset, model, branch / "examples/overfit", label="initial")
    started = perf_counter()
    history = trainer.fit(loader, loader)
    elapsed = perf_counter() - started
    after = _fixed_task_metrics(model, task, loader, seed=seed + 1)
    final_probe = _semantic_probe(root, base, spec, subset, model, branch / "examples/overfit", label="final")
    gates = spec["overfit_gates"]
    initial_summary, final_summary = initial_probe["summary"], final_probe["summary"]
    semantic_checks = {
        "fewer_fragments": final_summary["fragment_count"] < initial_summary["fragment_count"],
        "more_count_correct": final_summary["count_correct"] > initial_summary["count_correct"],
        "more_accepted_target_shapes": final_summary["accepted_target_shape_count"] > initial_summary["accepted_target_shape_count"],
        "target_shape_margin_increase": (
            final_summary["mean_target_shape_margin"] - initial_summary["mean_target_shape_margin"]
            >= float(gates["minimum_target_shape_margin_increase"])
        ),
    }
    parameter_report = model.parameter_report()
    memory = process.memory_info()
    peak_rss = int(getattr(memory, "peak_wset", memory.rss))
    checks = {
        "refined_clean_loss_ratio": after["refined_clean_latent_loss"] / before["refined_clean_latent_loss"] <= float(gates["refined_clean_loss_ratio_max"]),
        "foreground_loss_ratio": after["foreground_loss"] / before["foreground_loss"] <= float(gates["foreground_loss_ratio_max"]),
        "boundary_loss_ratio": after["boundary_loss"] / before["boundary_loss"] <= float(gates["boundary_loss_ratio_max"]),
        "fragments_not_increased": final_summary["fragment_count"] - initial_summary["fragment_count"] <= int(gates["maximum_fragment_increase"]),
        "position_not_materially_worse": initial_summary["position_correct"] - final_summary["position_correct"] <= int(gates["maximum_position_correct_decrease"]),
        "semantic_improvement": any(semantic_checks.values()),
        "denoiser_frozen": not any(parameter.requires_grad for parameter in model.denoiser.parameters()),
        "parameter_limit": parameter_report["total"] <= int(spec["resource_limits"]["max_parameters"]),
        "wall_time_limit": elapsed <= float(spec["resource_limits"]["max_branch_wall_seconds"]),
        "memory_limit": peak_rss <= int(spec["resource_limits"]["max_peak_rss_bytes"]),
    }
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "iteration_id": ITERATION_ID,
        "branch": "clean_refiner",
        "acceptance_items": ["MODEL-REF-004", "MODEL-REF-005"],
        "formal_weights_reused": False,
        "record_ids": record_ids,
        "sample_count": len(subset),
        "fixed_metrics_before": before,
        "fixed_metrics_after": after,
        "semantic_probe_before": initial_probe,
        "semantic_probe_after": final_probe,
        "semantic_improvement_checks": semantic_checks,
        "parameter_report": parameter_report,
        "wall_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "final_epoch": history[-1]["epoch"],
        "checks": checks,
        "verified_for_formal_training": all(checks.values()),
    }
    atomic_write_json(branch / "metrics/overfit_report.json", report)
    atomic_write_json(branch / "tests/overfit_gate.json", {"verified": report["verified_for_formal_training"], "checks": checks})
    return report


def _strict_reload_check(model_factory: Any, checkpoint: Path) -> dict[str, Any]:
    model = model_factory()
    state = load_checkpoint(checkpoint, model=model, optimizer=None, restore_rng=False)
    return {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "epoch": int(state["epoch"]),
        "best_metric": float(state["best_metric"]),
        "strict_reload": True,
        "state_sha256": _state_sha(model),
    }


def train_formal_branch(project_root: str | Path, branch_id: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    base, spec = load_config(root), load_i005_config(root)
    branch = _branch_dir(root, branch_id)
    overfit = _json(_branch_dir(root, "clean_refiner") / "metrics/overfit_report.json")
    if not overfit.get("verified_for_formal_training"):
        raise RuntimeError("formal I005 training requires a passing clean-refiner overfit gate")
    seed = int(base["seed"]) + int(spec["formal_control"]["seed_offset"])
    set_deterministic_seed(seed, cpu_threads=int(base["cpu_threads"]))
    train = _dataset(root, base, split="train")
    validation = _dataset(root, base, split="validation")
    train_loader = _loader(train, base, spec, seed=seed, balanced=True)
    validation_loader = _loader(validation, base, spec, seed=seed, balanced=False)
    epochs = int(spec["clean_refiner"]["finetune_epochs"])
    if branch_id == "continuation":
        model = grounded_model_from_config(base, train.tokenizer)
        load_checkpoint(root / spec["frozen_assets"]["warmstart"], model=model, optimizer=None, restore_rng=False)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["clean_refiner"]["learning_rate"]),
            weight_decay=float(spec["clean_refiner"]["weight_decay"]),
        )
        task = _task(base, train.tokenizer)
        factory = lambda: grounded_model_from_config(base, train.tokenizer)
        monitor = "noise_mse"
    else:
        model = refiner_model_from_config(root, base, spec, train.tokenizer)
        optimizer = torch.optim.AdamW(
            model.refiner.parameters(),
            lr=float(spec["clean_refiner"]["learning_rate"]),
            weight_decay=float(spec["clean_refiner"]["weight_decay"]),
        )
        task = _refiner_task(root, base, spec, train.tokenizer)
        factory = lambda: refiner_model_from_config(root, base, spec, train.tokenizer, load_warmstart=False)
        monitor = "loss"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=task,
        config=TrainerConfig(
            epochs=epochs,
            gradient_clip_norm=float(base["training"]["gradient_clip_norm"]),
            monitor=monitor,
            maximize=False,
        ),
        output_dir=branch,
    )
    _configure_trainer_paths(trainer, branch / "checkpoints", branch / "logs/training_log.jsonl")
    process = psutil.Process()
    started = perf_counter()
    history = trainer.fit(train_loader, validation_loader)
    elapsed = perf_counter() - started
    best_checkpoint = branch / "checkpoints/best.pt"
    last_checkpoint = branch / "checkpoints/last.pt"
    reload_report = _strict_reload_check(factory, best_checkpoint)
    memory = process.memory_info()
    peak_rss = int(getattr(memory, "peak_wset", memory.rss))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "iteration_id": ITERATION_ID,
        "branch": branch_id,
        "acceptance_item": "EXP-I005-001",
        "warmstart": spec["frozen_assets"]["warmstart"],
        "warmstart_sha256": sha256_file(root / spec["frozen_assets"]["warmstart"]),
        "formal_overfit_weights_reused": False,
        "seed": seed,
        "epochs": epochs,
        "batch_size": int(spec["formal_control"]["batch_size"]),
        "training_sample_count": len(train),
        "validation_sample_count": len(validation),
        "balanced_single_object_sampling": True,
        "learning_rate": float(spec["clean_refiner"]["learning_rate"]),
        "weight_decay": float(spec["clean_refiner"]["weight_decay"]),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "wall_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "last_epoch_metrics": history[-1],
        "checkpoint_reload": reload_report,
        "last_checkpoint": {
            "path": last_checkpoint.relative_to(root).as_posix(),
            "sha256": sha256_file(last_checkpoint),
            "bytes": last_checkpoint.stat().st_size,
        },
        "checks": {
            "same_formal_budget": epochs == int(spec["clean_refiner"]["finetune_epochs"]),
            "parameter_limit": parameter_count <= int(spec["resource_limits"]["max_parameters"]),
            "wall_time_limit": elapsed <= float(spec["resource_limits"]["max_branch_wall_seconds"]),
            "memory_limit": peak_rss <= int(spec["resource_limits"]["max_peak_rss_bytes"]),
            "strict_checkpoint_reload": reload_report["strict_reload"],
        },
    }
    report["verified_for_generation_evaluation"] = all(report["checks"].values())
    atomic_write_json(branch / "metrics/training_metrics.json", report)
    atomic_write_json(branch / "checkpoints/checkpoint_manifest.json", {
        "schema_version": 1,
        "best": {
            "path": best_checkpoint.relative_to(root).as_posix(),
            "sha256": sha256_file(best_checkpoint),
            "bytes": best_checkpoint.stat().st_size,
            "epoch": reload_report["epoch"],
            "best_metric": reload_report["best_metric"],
        },
        "last": report["last_checkpoint"],
        "usage": "Load model_state with mini_multimodal.training.load_checkpoint; see examples/inference_usage.md",
    })
    return report


def _load_formal_model(
    root: Path,
    base: Mapping[str, Any],
    spec: Mapping[str, Any],
    branch_id: str,
) -> tuple[torch.nn.Module, CharacterTokenizer]:
    vocabulary = _json(root / base["output_data"] / "vocabulary.json")
    tokenizer = CharacterTokenizer(vocabulary)
    if branch_id == "continuation":
        model = grounded_model_from_config(base, tokenizer)
    else:
        model = refiner_model_from_config(root, base, spec, tokenizer, load_warmstart=False)
    load_checkpoint(_branch_dir(root, branch_id) / "checkpoints/best.pt", model=model, optimizer=None, restore_rng=False)
    model.eval()
    return model, tokenizer


def _capability_examples(root: Path, branch: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row for row in rows if row["is_fully_correct"]]
    wrong = [row for row in rows if not row["is_fully_correct"]]
    selected = [("正确", row) for row in correct[:4]] + [("错误", row) for row in wrong[:4]]
    payload = {"schema_version": 1, "selection_policy": "前4个完全正确样例和前4个错误样例", "examples": []}
    lines = ["# I005直观能力样例", "", "> 同时展示正确和错误输出；判断来自冻结评价器。", ""]
    for index, (category, row) in enumerate(selected, 1):
        pairs = row["pairs"]
        item = {
            "index": index,
            "category": category,
            "record_id": row["record_id"],
            "prompt": row["prompt"],
            "target_object_count": row["target_object_count"],
            "predicted_object_count": row["predicted_object_count"],
            "is_fully_correct": row["is_fully_correct"],
            "generated_image": row["generated_image"],
            "pairs": pairs,
        }
        payload["examples"].append(item)
        target_shapes = [pair["target"]["shape"] for pair in pairs]
        predicted_shapes = [None if pair["prediction"] is None else pair["prediction"]["shape"] for pair in pairs]
        lines.extend([
            f"## 样例{index}：{category}", "",
            f"- 输入文本：{row['prompt']}",
            f"- 对象数：目标{row['target_object_count']}，预测{row['predicted_object_count']}",
            f"- 形状：目标{target_shapes}，预测{predicted_shapes}",
            f"- 完全正确：{row['is_fully_correct']}",
            f"- 图像：`{row['generated_image']}`", "",
        ])
    atomic_write_json(branch / "examples/capability_examples.json", payload)
    (branch / "examples/capability_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


@torch.no_grad()
def evaluate_formal_branch(project_root: str | Path, branch_id: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    base, spec = load_config(root), load_i005_config(root)
    branch = _branch_dir(root, branch_id)
    training = _json(branch / "metrics/training_metrics.json")
    if not training.get("verified_for_generation_evaluation"):
        raise RuntimeError(f"{branch_id} did not pass the training/resource checks")
    eval_dir = root / spec["frozen_assets"]["evaluation_data"]
    freeze = _json(root / spec["frozen_assets"]["thresholds"])
    if sha256_file(eval_dir / "manifest.json") != freeze["evaluation_manifest_sha256"]:
        raise RuntimeError("fixed evaluation manifest changed")
    rows = [json.loads(line) for line in (eval_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    masks_payload = np.load(eval_dir / "ideal_masks.npz", allow_pickle=False)
    masks = masks_payload["amodal"]
    masks_payload.close()
    prototypes = load_shape_prototypes(root / spec["frozen_assets"]["shape_prototypes"])
    model, tokenizer = _load_formal_model(root, base, spec, branch_id)
    vae = load_vae_decoder(root, base["source_vae_checkpoint"])
    scaling = float(_json(root / base["output_data"] / "manifest.json")["latent_scaling_factor"])
    image_dir = branch / "predictions/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    base_seed = int(base["seed"]) + int(spec["formal_control"]["evaluation_seed_offset"])
    started = perf_counter()
    for start in range(0, len(rows), 8):
        batch_rows = rows[start:start + 8]
        prompts = [row["prompt"] for row in batch_rows]
        if branch_id == "continuation":
            sampled = sample_latents_v2(
                model, schedule_from_config(base), tokenizer, prompts,
                sample_steps=int(spec["formal_sampling"]["sample_steps"]),
                guidance_scale=float(spec["formal_sampling"]["guidance_scale"]),
                seed=base_seed + start,
            )
        else:
            sampled = sample_latents_refined(
                model, tokenizer, prompts,
                sample_steps=int(spec["formal_sampling"]["sample_steps"]),
                guidance_scale=float(spec["formal_sampling"]["guidance_scale"]),
                seed=base_seed + start,
            )
        images = vae.decode(sampled["latent"] / scaling).clamp(0, 1)
        for offset, (row, image) in enumerate(zip(batch_rows, images, strict=True)):
            threshold_name = "unseen_single" if row["subset"] == "unseen_single" else "seen_single"
            evaluation = evaluate_image_v2(
                image,
                row,
                masks[row["ideal_mask_index"]],
                fully_correct_shape_iou=float(freeze["thresholds"][threshold_name]["normalized_template_iou_mean"]),
                shape_prototypes=prototypes,
            )
            image_path = image_dir / f"{branch_id}_{start + offset:03d}.png"
            Image.fromarray(image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), mode="RGB").save(image_path)
            evaluation.update({
                "generated_image": image_path.relative_to(root).as_posix(),
                "generated_image_sha256": sha256_file(image_path),
                "sample_seed": base_seed + start,
                "checkpoint": (branch / "checkpoints/best.pt").relative_to(root).as_posix(),
                "sample_steps": int(spec["formal_sampling"]["sample_steps"]),
                "guidance_scale": float(spec["formal_sampling"]["guidance_scale"]),
            })
            predictions.append(evaluation)
    metrics = {
        subset: aggregate_evaluations([row for row in predictions if row["subset"] == subset])
        for subset in ("seen_single", "unseen_single", "multi_2", "multi_3")
    }
    seen_checks = _fixed_checks(metrics["seen_single"], freeze["thresholds"]["seen_single"], "seen_single")
    unseen_checks = _fixed_checks(metrics["unseen_single"], freeze["thresholds"]["unseen_single"], "unseen_single")
    multi_ref = freeze["thresholds"]["multi_object_reference"]
    multi_checks = {
        subset: {name: _value(metrics[subset][name]) >= float(multi_ref[name]) for name in ("count_accuracy", "color_accuracy", "shape_accuracy", "position_accuracy")}
        for subset in ("multi_2", "multi_3")
    }
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "iteration_id": ITERATION_ID,
        "branch": branch_id,
        "acceptance_items": ["EXP-I005-002", "EXP-I005-003", "EXP-I005-005", "EXP-I005-006"],
        "sample_count": len(predictions),
        "evaluation_manifest": (eval_dir / "manifest.json").relative_to(root).as_posix(),
        "evaluation_manifest_sha256": sha256_file(eval_dir / "manifest.json"),
        "checkpoint": (branch / "checkpoints/best.pt").relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(branch / "checkpoints/best.pt"),
        "sample_steps": int(spec["formal_sampling"]["sample_steps"]),
        "guidance_scale": float(spec["formal_sampling"]["guidance_scale"]),
        "base_seed": base_seed,
        "wall_seconds": perf_counter() - started,
        "metrics": metrics,
        "checks": {"seen_single": seen_checks, "unseen_single": unseen_checks, "multi_object_nonblocking_reference": multi_checks},
        "seen_single_verified": all(seen_checks.values()),
        "unseen_single_verified": all(unseen_checks.values()),
        "multi_object_reference_blocking": False,
    }
    report["all_blocking_generation_items_passed"] = report["seen_single_verified"] and report["unseen_single_verified"]
    _write_jsonl(branch / "predictions/fixed_predictions.jsonl", predictions)
    atomic_write_json(branch / "metrics/generation_metrics.json", report)
    _capability_examples(root, branch, predictions)
    usage = (
        f"# {ITERATION_ID}/{branch_id}推理包使用方法\n\n"
        "检查点见`../checkpoints/checkpoint_manifest.json`。纯文本推理必须使用训练前快照中的相同配置、词表和模型类；"
        "评价时固定为25步、CFG 2.5，并按记录批次使用报告中的种子。\n\n"
        "```python\n"
        "from mini_multimodal.training import load_checkpoint\n"
        "# 先按分支构造模型model，再严格加载：\n"
        "load_checkpoint(checkpoint_path, model=model, optimizer=None, restore_rng=False)\n"
        "model.eval()\n"
        "# continuation调用sample_latents_v2；clean_refiner调用sample_latents_refined。\n"
        "```\n"
    )
    (branch / "examples/inference_usage.md").write_text(usage, encoding="utf-8")
    atomic_write_json(branch / "tests/runtime_validation.json", {
        "sample_count_420": len(predictions) == 420,
        "prediction_images_all_exist": all((root / row["generated_image"]).is_file() for row in predictions),
        "prediction_image_hashes_match": all(sha256_file(root / row["generated_image"]) == row["generated_image_sha256"] for row in predictions),
        "fixed_manifest_matches": report["evaluation_manifest_sha256"] == freeze["evaluation_manifest_sha256"],
    })
    return report


def compare_formal_branches(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    iteration = root / "results/P2-W04_tiny_image_ldm/iterations" / ITERATION_ID
    reports = {branch: _json(_branch_dir(root, branch) / "metrics/generation_metrics.json") for branch in BRANCH_IDS}
    keys = ("count_accuracy", "shape_accuracy", "normalized_template_iou_mean", "boundary_score_mean", "position_accuracy", "fully_correct_rate")
    comparison: dict[str, Any] = {}
    for subset in ("seen_single", "unseen_single", "multi_2", "multi_3"):
        comparison[subset] = {
            key: {
                branch: _value(reports[branch]["metrics"][subset][key]) for branch in BRANCH_IDS
            } | {
                "clean_refiner_minus_continuation": _value(reports["clean_refiner"]["metrics"][subset][key]) - _value(reports["continuation"]["metrics"][subset][key])
            }
            for key in keys
        }
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "iteration_id": ITERATION_ID,
        "acceptance_item": "EXP-I005-004",
        "fixed_evaluation_sample_count_per_branch": 420,
        "comparison": comparison,
        "blocking_pass": {branch: reports[branch]["all_blocking_generation_items_passed"] for branch in BRANCH_IDS},
        "automatic_candidate_selected": False,
        "decision_required_from_teacher": True,
        "source_files": {branch: (_branch_dir(root, branch) / "metrics/generation_metrics.json").relative_to(root).as_posix() for branch in BRANCH_IDS},
    }
    atomic_write_json(iteration / "comparisons/branch_comparison.json", report)
    lines = ["# I005双分支比较", "", "> 本报告不自动选择候选版本。所有数值均可在下列机器文件中核对。", ""]
    for subset in ("seen_single", "unseen_single"):
        lines.extend([f"## {subset}", "", "| 指标 | continuation | clean_refiner | 差值 |", "|---|---:|---:|---:|"])
        for key in keys:
            row = comparison[subset][key]
            lines.append(f"| {key} | {row['continuation']:.4f} | {row['clean_refiner']:.4f} | {row['clean_refiner_minus_continuation']:+.4f} |")
        lines.append("")
    lines.extend(["## 证据路径", ""] + [f"- `{branch}`：`{path}`" for branch, path in report["source_files"].items()] + [""])
    (iteration / "comparisons/branch_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return report
