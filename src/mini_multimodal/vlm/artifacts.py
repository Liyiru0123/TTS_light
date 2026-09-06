"""Training, evaluation, ablation, visualization, and reports for P2-W03."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from mini_multimodal.training import (
    StepOutput,
    Trainer,
    TrainerConfig,
    load_checkpoint,
    seeded_generator,
    set_deterministic_seed,
)

from .baselines import ImageOnlyVQABaseline, TextOnlyVQABaseline
from .data import CharacterTokenizer, VLMQADataset, collate_vlm_qa, export_p2_w03_data, verify_p2_w03_data
from .model import TinyVLM
from .task import VLMTask


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _loader(
    dataset: VLMQADataset,
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
        collate_fn=collate_vlm_qa,
        generator=seeded_generator(seed) if shuffle else None,
    )


def _model_kwargs(config: Mapping[str, Any], tokenizer: CharacterTokenizer, answer_count: int) -> dict[str, Any]:
    return {
        "vocabulary_size": len(tokenizer),
        "answer_class_count": answer_count,
        "pad_id": tokenizer.id("<pad>"),
        "bos_id": tokenizer.id("<bos>"),
        "eos_id": tokenizer.id("<eos>"),
        "mask_id": tokenizer.id("<mask>"),
        "target_marker_id": tokenizer.id("甲"),
        "reference_marker_id": tokenizer.id("乙"),
        **config["model"],
    }


def _new_model(config: Mapping[str, Any], tokenizer: CharacterTokenizer, answer_count: int) -> TinyVLM:
    return TinyVLM(**_model_kwargs(config, tokenizer, answer_count))


def _new_trainer(
    model: nn.Module,
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    epochs: int,
    classification_weight: float,
    generation_weight: float,
    monitor: str = "loss",
    maximize: bool = False,
) -> tuple[torch.optim.Optimizer, Trainer]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=VLMTask(
            pad_id=0,
            classification_weight=classification_weight,
            generation_weight=generation_weight,
            grounding_weight=float(config["training"].get("grounding_loss_weight", 0.0)),
        ),
        config=TrainerConfig(
            epochs=epochs,
            device="cpu",
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            monitor=monitor,
            maximize=maximize,
        ),
        output_dir=output_dir,
    )
    return optimizer, trainer


def _group_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[bool]]] = {
        "question_kind": defaultdict(list),
        "variant": defaultdict(list),
        "object_group": defaultdict(list),
    }
    for row in rows:
        for metric in ("classification_correct", "generation_correct"):
            groups["question_kind"][f"{row['question_kind']}::{metric}"].append(bool(row[metric]))
            groups["variant"][f"{row['variant']}::{metric}"].append(bool(row[metric]))
            label = "multi_object" if row["object_count"] > 1 else "single_object"
            groups["object_group"][f"{label}::{metric}"].append(bool(row[metric]))
    report: dict[str, Any] = {}
    for group_name, values in groups.items():
        nested: dict[str, dict[str, float | int]] = defaultdict(dict)
        for key, flags in values.items():
            label, metric = key.split("::", 1)
            nested[label][metric.replace("_correct", "_accuracy")] = sum(flags) / len(flags)
            nested[label]["count"] = len(flags)
        report[group_name] = dict(sorted(nested.items()))
    return report


@torch.no_grad()
def evaluate_vlm(
    model: TinyVLM,
    loader: DataLoader,
    *,
    tokenizer: CharacterTokenizer,
    answer_classes: Sequence[str],
    ablations: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    model.eval()
    settings = dict(ablations or {})
    prediction_rows: list[dict[str, Any]] = []
    for batch in loader:
        output = model(batch["visual"], batch["question_ids"], **settings)
        class_ids = output.classification_logits.argmax(dim=-1).tolist()
        generated, probabilities = model.generate(
            batch["visual"], batch["question_ids"], **settings
        )
        for index, metadata in enumerate(batch["metadata"]):
            predicted_class = answer_classes[class_ids[index]]
            predicted_answer = tokenizer.decode_answer(generated[index].tolist())
            target_coordinates = output.target_coordinates[index].tolist()
            reference_coordinates = output.reference_coordinates[index].tolist()
            grounding_prediction = None
            if metadata["question_kind"] == "horizontal_relation":
                grounding_prediction = "在左边" if target_coordinates[0] < reference_coordinates[0] else "在右边"
            elif metadata["question_kind"] == "vertical_relation":
                grounding_prediction = "在上方" if target_coordinates[1] < reference_coordinates[1] else "在下方"
            prediction_rows.append({
                **metadata,
                "classification_prediction": predicted_class,
                "generation_prediction": predicted_answer,
                "classification_correct": predicted_class == metadata["answer_zh"],
                "generation_correct": predicted_answer == metadata["answer_zh"],
                "generated_token_ids": generated[index].tolist(),
                "generated_token_probabilities": probabilities[index].tolist(),
                "target_coordinate_prediction": target_coordinates,
                "reference_coordinate_prediction": reference_coordinates,
                "grounding_relation_prediction": grounding_prediction,
                "grounding_relation_correct": (
                    grounding_prediction == metadata["answer_zh"]
                    if grounding_prediction is not None
                    else None
                ),
            })
    total = len(prediction_rows)
    classification_accuracy = sum(row["classification_correct"] for row in prediction_rows) / total
    generation_accuracy = sum(row["generation_correct"] for row in prediction_rows) / total
    relation_rows = [
        row for row in prediction_rows if row["grounding_relation_correct"] is not None
    ]
    coordinate_errors: list[float] = []
    for row in prediction_rows:
        for key, prediction_key in (
            ("target_position", "target_coordinate_prediction"),
            ("reference_position", "reference_coordinate_prediction"),
        ):
            if row[key] is not None:
                truth = np.asarray(row[key], dtype=np.float32) * 2.0 - 1.0
                prediction = np.asarray(row[prediction_key], dtype=np.float32)
                coordinate_errors.append(float(np.abs(truth - prediction).mean()))
    return {
        "count": total,
        "classification_accuracy": classification_accuracy,
        "generation_accuracy": generation_accuracy,
        "grounding_mae": float(np.mean(coordinate_errors)),
        "grounding_relation_accuracy": sum(
            bool(row["grounding_relation_correct"]) for row in relation_rows
        ) / len(relation_rows),
        "groups": _group_report(prediction_rows),
        "predictions": prediction_rows,
    }


class _BaselineTask:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def step(self, model: nn.Module, batch: Mapping[str, Any], *, training: bool) -> StepOutput:
        logits = model(batch["question_ids"] if self.mode == "text" else batch["visual"])
        per_sample = F.cross_entropy(logits, batch["answer_class"], reduction="none")
        weights = batch["sample_weight"]
        loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
        accuracy = (logits.argmax(dim=-1) == batch["answer_class"]).float().mean()
        return StepOutput(loss=loss, metrics={"accuracy": float(accuracy)}, weight=len(logits))


@torch.no_grad()
def _evaluate_baseline(model: nn.Module, loader: DataLoader, mode: str) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        logits = model(batch["question_ids"] if mode == "text" else batch["visual"])
        correct += int((logits.argmax(dim=-1) == batch["answer_class"]).sum())
        total += len(logits)
    return correct / total


def _train_baseline(
    model: nn.Module,
    mode: str,
    output_dir: Path,
    config: Mapping[str, Any],
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=_BaselineTask(mode),
        config=TrainerConfig(
            epochs=int(config["training"]["baseline_epochs"]),
            device="cpu",
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            monitor="loss",
            maximize=False,
        ),
        output_dir=output_dir,
    )
    history = trainer.fit(train_loader, validation_loader)
    load_checkpoint(output_dir / "checkpoints" / "best.pt", model=model, optimizer=optimizer, restore_rng=False)
    return model, history


def _cue_baseline(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    training = [row for row in records if row["split"] == "train" and row["variant"] == "shortcut_iid"]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in training:
        counts[str(row["cue_token"])][row["answer_zh"]] += 1
    mapping = {cue: answers.most_common(1)[0][0] for cue, answers in counts.items()}

    def evaluate(split: str, variant: str) -> dict[str, float | int]:
        rows = [row for row in records if row["split"] == split and row["variant"] == variant]
        correct = sum(mapping.get(str(row["cue_token"])) == row["answer_zh"] for row in rows)
        return {"count": len(rows), "accuracy": correct / len(rows)}

    return {
        "mapping": mapping,
        "shortcut_iid_validation": evaluate("validation", "shortcut_iid"),
        "counterfactual_test": evaluate("test", "counterfactual"),
    }


def _save_curves(path: Path, classification_history: Sequence[dict[str, Any]], generation_history: Sequence[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for history, label in ((classification_history, "classification"), (generation_history, "generation")):
        axes[0].plot([row["epoch"] for row in history], [row["validation"]["loss"] for row in history], label=label)
        metric = "classification_accuracy" if label == "classification" else "sequence_accuracy"
        axes[1].plot([row["epoch"] for row in history], [row["validation"][metric] for row in history], label=label)
    axes[0].set_title("validation loss")
    axes[1].set_title("stage validation accuracy")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_confusion(path: Path, predictions: Sequence[dict[str, Any]], answer_classes: Sequence[str]) -> None:
    index = {answer: i for i, answer in enumerate(answer_classes)}
    matrix = np.zeros((len(answer_classes), len(answer_classes)), dtype=np.int64)
    for row in predictions:
        matrix[index[row["answer_zh"]], index[row["classification_prediction"]]] += 1
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(answer_classes)), answer_classes, rotation=90, fontsize=7)
    axis.set_yticks(range(len(answer_classes)), answer_classes, fontsize=7)
    axis.set_xlabel("prediction")
    axis.set_ylabel("target")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_attention(
    path: Path,
    root: Path,
    source_index: str,
    model: TinyVLM,
    dataset: VLMQADataset,
    tokenizer: CharacterTokenizer,
) -> None:
    row_by_scene = {
        row["scene_id"]: row
        for row in (
            json.loads(line)
            for line in (root / source_index / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    selected_indices = [
        index for index, row in enumerate(dataset.rows)
        if row["target_position"] is not None
    ][:4]
    samples = [dataset[index] for index in selected_indices]
    batch = collate_vlm_qa(samples)
    model.eval()
    with torch.no_grad():
        output = model(batch["visual"], batch["question_ids"])
        generated, _ = model.generate(batch["visual"], batch["question_ids"])
    figure, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 7))
    for index, metadata in enumerate(batch["metadata"]):
        image_path = root / row_by_scene[metadata["scene_id"]]["assets"]["image"]["path"]
        with Image.open(image_path) as source:
            image = np.array(source.convert("RGB"), copy=True)
        heatmap = output.fusion_attention[index, :, 0].mean(dim=0).reshape(16, 16).cpu().numpy()
        heatmap = np.array(Image.fromarray(heatmap).resize((64, 64), Image.Resampling.BILINEAR))
        axes[0, index].imshow(image)
        axes[0, index].axis("off")
        axes[0, index].set_title(metadata["question_zh"], fontsize=8)
        axes[1, index].imshow(image)
        axes[1, index].imshow(heatmap, cmap="magma", alpha=0.55)
        axes[1, index].axis("off")
        prediction = tokenizer.decode_answer(generated[index].tolist())
        axes[1, index].set_title(f"目标:{metadata['answer_zh']} 生成:{prediction}", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _metric_minimum(report: Mapping[str, Any], group: str, metric: str) -> float:
    return min(float(values[metric]) for values in report["groups"][group].values())


def run_p2_w03_experiments(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_dir = root / config["output_data"]
    result_dir = root / config["output_results"]
    result_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    started = perf_counter()
    seed = int(config["seed"])
    threads = int(config["cpu_threads"])
    set_deterministic_seed(seed, cpu_threads=threads)

    export_p2_w03_data(
        root,
        data_dir,
        source_index=config["source_index"],
        source_latents=config["source_latents"],
        overfit_count=int(config["training"]["overfit_count"]),
    )
    data_report = verify_p2_w03_data(root, data_dir)
    _write_json(result_dir / "data_readiness_report.json", data_report["statistics"])
    vocabulary = json.loads((data_dir / "vocabulary.json").read_text(encoding="utf-8"))
    answer_classes = json.loads((data_dir / "answer_classes.json").read_text(encoding="utf-8"))
    overfit_ids = json.loads((data_dir / "overfit_ids.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocabulary)
    datasets = {
        split: VLMQADataset(
            root,
            data_dir=data_dir,
            source_latents=config["source_latents"],
            splits=split,
        )
        for split in ("train", "validation", "test")
    }
    overfit_dataset = VLMQADataset(
        root,
        data_dir=data_dir,
        source_latents=config["source_latents"],
        splits="train",
        record_ids=overfit_ids,
    )
    batch_size = int(config["training"]["batch_size"])
    train_loader = _loader(datasets["train"], batch_size=batch_size, shuffle=True, seed=seed)
    validation_loader = _loader(datasets["validation"], batch_size=batch_size)
    test_loader = _loader(datasets["test"], batch_size=batch_size)
    thresholds = config["thresholds"]

    # Small-data gate for the complete classifier + causal decoder path.
    set_deterministic_seed(seed, cpu_threads=threads)
    overfit_model = _new_model(config, tokenizer, len(answer_classes))
    _, overfit_trainer = _new_trainer(
        overfit_model,
        result_dir / "overfit_run",
        config,
        epochs=int(config["training"]["overfit_epochs"]),
        classification_weight=0.5,
        generation_weight=1.0,
    )
    overfit_loader = _loader(overfit_dataset, batch_size=len(overfit_dataset), shuffle=True, seed=seed)
    overfit_history = overfit_trainer.fit(overfit_loader, overfit_loader)
    overfit_eval = evaluate_vlm(
        overfit_model,
        overfit_loader,
        tokenizer=tokenizer,
        answer_classes=answer_classes,
    )
    overfit_report = {
        "count": len(overfit_dataset),
        "initial": overfit_history[0]["validation"],
        "final": overfit_history[-1]["validation"],
        "greedy_classification_accuracy": overfit_eval["classification_accuracy"],
        "greedy_generation_accuracy": overfit_eval["generation_accuracy"],
    }
    overfit_report["loss_ratio"] = overfit_report["final"]["loss"] / overfit_report["initial"]["loss"]
    overfit_report["verified"] = (
        overfit_report["loss_ratio"] <= thresholds["overfit_loss_ratio"]
        and overfit_report["greedy_classification_accuracy"] >= thresholds["overfit_classification_accuracy"]
        and overfit_report["greedy_generation_accuracy"] >= thresholds["overfit_sequence_accuracy"]
    )
    _write_json(result_dir / "overfit_report.json", overfit_report)
    if not overfit_report["verified"]:
        raise RuntimeError("P2-W03 full-path overfit gate failed")

    # Stage A: closed-set multimodal classifier.
    set_deterministic_seed(seed, cpu_threads=threads)
    model = _new_model(config, tokenizer, len(answer_classes))
    classification_optimizer, classification_trainer = _new_trainer(
        model,
        result_dir / "classification",
        config,
        epochs=int(config["training"]["classification_epochs"]),
        classification_weight=1.0,
        generation_weight=0.0,
    )
    classification_history = classification_trainer.fit(train_loader, validation_loader)
    classification_state = load_checkpoint(
        result_dir / "classification" / "checkpoints" / "best.pt",
        model=model,
        optimizer=classification_optimizer,
        restore_rng=False,
    )
    classification_eval = evaluate_vlm(
        model, test_loader, tokenizer=tokenizer, answer_classes=answer_classes
    )
    class_groups = classification_eval["groups"]
    classification_checks = {
        "overall": classification_eval["classification_accuracy"] >= thresholds["classification_accuracy"],
        "kind_minimum": min(values["classification_accuracy"] for values in class_groups["question_kind"].values()) >= thresholds["classification_kind_accuracy"],
        "multi_object": class_groups["object_group"]["multi_object"]["classification_accuracy"] >= thresholds["classification_multi_object_accuracy"],
        "grounding_relation": classification_eval["grounding_relation_accuracy"] >= thresholds["grounding_relation_accuracy"],
    }
    classification_report = {
        "best_epoch": int(classification_state["epoch"]),
        "classification_accuracy": classification_eval["classification_accuracy"],
        "grounding_mae": classification_eval["grounding_mae"],
        "grounding_relation_accuracy": classification_eval["grounding_relation_accuracy"],
        "groups": class_groups,
        "checks": classification_checks,
        "verified": all(classification_checks.values()),
    }
    _write_json(result_dir / "classification_metrics.json", classification_report)
    (result_dir / "classification_predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in classification_eval["predictions"]
        ),
        encoding="utf-8",
        newline="\n",
    )
    if not classification_report["verified"]:
        raise RuntimeError("P2-W03 classification stage failed")

    # Stage B: causal token generation, initialized from Stage A.
    generation_optimizer, generation_trainer = _new_trainer(
        model,
        result_dir / "generative",
        config,
        epochs=int(config["training"]["generation_epochs"]),
        classification_weight=float(config["training"]["classification_loss_weight"]),
        generation_weight=float(config["training"]["generation_loss_weight"]),
        monitor="sequence_accuracy",
        maximize=True,
    )
    generation_history = generation_trainer.fit(train_loader, validation_loader)
    generation_state = load_checkpoint(
        result_dir / "generative" / "checkpoints" / "best.pt",
        model=model,
        optimizer=generation_optimizer,
        restore_rng=False,
    )
    final_eval = evaluate_vlm(
        model, test_loader, tokenizer=tokenizer, answer_classes=answer_classes
    )
    generation_groups = final_eval["groups"]
    variant = generation_groups["variant"]
    generation_checks = {
        "overall": final_eval["generation_accuracy"] >= thresholds["generation_accuracy"],
        "kind_minimum": min(values["generation_accuracy"] for values in generation_groups["question_kind"].values()) >= thresholds["generation_kind_accuracy"],
        "retained_classification": final_eval["classification_accuracy"] >= thresholds["retained_classification_accuracy"],
        "generation_classification_gap": final_eval["classification_accuracy"] - final_eval["generation_accuracy"] <= thresholds["generation_classification_gap"],
        "normal": variant["normal"]["generation_accuracy"] >= thresholds["normal_accuracy"],
        "compositional": variant["compositional"]["generation_accuracy"] >= thresholds["compositional_accuracy"],
        "counterfactual": variant["counterfactual"]["generation_accuracy"] >= thresholds["counterfactual_accuracy"],
    }
    generation_report = {
        "best_epoch": int(generation_state["epoch"]),
        "classification_accuracy": final_eval["classification_accuracy"],
        "generation_accuracy": final_eval["generation_accuracy"],
        "grounding_mae": final_eval["grounding_mae"],
        "grounding_relation_accuracy": final_eval["grounding_relation_accuracy"],
        "groups": generation_groups,
        "checks": generation_checks,
        "verified": all(generation_checks.values()),
    }
    _write_json(result_dir / "generation_metrics.json", generation_report)
    predictions_path = result_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in final_eval["predictions"]),
        encoding="utf-8",
        newline="\n",
    )
    if not generation_report["verified"]:
        raise RuntimeError("P2-W03 causal generation stage failed")

    # Proper separately trained single-modality baselines.
    set_deterministic_seed(seed + 11, cpu_threads=threads)
    text_baseline = TextOnlyVQABaseline(
        vocabulary_size=len(tokenizer),
        answer_class_count=len(answer_classes),
        pad_id=tokenizer.id("<pad>"),
        d_model=int(config["model"]["d_model"]),
        num_heads=int(config["model"]["num_heads"]),
        ff_dim=int(config["model"]["ff_dim"]),
        max_question_length=int(config["model"]["max_question_length"]),
        dropout=float(config["model"]["dropout"]),
    )
    text_baseline, _ = _train_baseline(
        text_baseline, "text", result_dir / "text_only", config, train_loader, validation_loader
    )
    text_accuracy = _evaluate_baseline(text_baseline, test_loader, "text")
    set_deterministic_seed(seed + 12, cpu_threads=threads)
    image_baseline = ImageOnlyVQABaseline(
        answer_class_count=len(answer_classes),
        d_model=int(config["model"]["d_model"]),
        num_heads=int(config["model"]["num_heads"]),
        visual_channels=int(config["model"]["visual_channels"]),
        visual_height=int(config["model"]["visual_height"]),
        visual_width=int(config["model"]["visual_width"]),
    )
    image_baseline, _ = _train_baseline(
        image_baseline, "image", result_dir / "image_only", config, train_loader, validation_loader
    )
    image_accuracy = _evaluate_baseline(image_baseline, test_loader, "image")
    best_single = max(text_accuracy, image_accuracy)
    baseline_report = {
        "text_only_classification_accuracy": text_accuracy,
        "image_only_classification_accuracy": image_accuracy,
        "best_single_modality_accuracy": best_single,
        "multimodal_classification_accuracy": final_eval["classification_accuracy"],
        "advantage": final_eval["classification_accuracy"] - best_single,
    }
    baseline_report["verified"] = baseline_report["advantage"] >= thresholds["multimodal_advantage"]
    _write_json(result_dir / "single_modality_report.json", baseline_report)

    # Inference-time component ablations on the trained multimodal model.
    ablation_settings = {
        "image_masked": {"mask_image": True},
        "text_masked": {"mask_text": True},
        "fusion_disabled": {"disable_fusion": True},
        "visual_position_disabled": {"disable_visual_position": True},
    }
    ablation_metrics = {
        name: evaluate_vlm(
            model,
            test_loader,
            tokenizer=tokenizer,
            answer_classes=answer_classes,
            ablations=settings,
        )
        for name, settings in ablation_settings.items()
    }
    relation_kinds = {"horizontal_relation", "vertical_relation"}
    relation_rows = [
        row for row in final_eval["predictions"] if row["question_kind"] in relation_kinds
    ]
    position_rows = [
        row for row in ablation_metrics["visual_position_disabled"]["predictions"]
        if row["question_kind"] in relation_kinds
    ]
    relation_accuracy = sum(row["generation_correct"] for row in relation_rows) / len(relation_rows)
    no_position_relation_accuracy = sum(row["generation_correct"] for row in position_rows) / len(position_rows)
    ablation_report = {
        "full_generation_accuracy": final_eval["generation_accuracy"],
        "ablations": {
            name: {
                "classification_accuracy": report["classification_accuracy"],
                "generation_accuracy": report["generation_accuracy"],
                "drop": final_eval["generation_accuracy"] - report["generation_accuracy"],
            }
            for name, report in ablation_metrics.items()
        },
        "relation_generation_accuracy": relation_accuracy,
        "no_position_relation_generation_accuracy": no_position_relation_accuracy,
        "position_relation_drop": relation_accuracy - no_position_relation_accuracy,
    }
    ablation_checks = {
        "image": ablation_report["ablations"]["image_masked"]["drop"] >= thresholds["image_ablation_drop"],
        "text": ablation_report["ablations"]["text_masked"]["drop"] >= thresholds["text_ablation_drop"],
        "fusion": ablation_report["ablations"]["fusion_disabled"]["drop"] >= thresholds["fusion_ablation_drop"],
        "position": ablation_report["position_relation_drop"] >= thresholds["position_relation_drop"],
    }
    ablation_report["checks"] = ablation_checks
    ablation_report["verified"] = all(ablation_checks.values())
    _write_json(result_dir / "ablation_report.json", ablation_report)

    all_records = [json.loads(line) for line in (data_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    shortcut_report = _cue_baseline(all_records)
    shortcut_report["main_counterfactual_generation_accuracy"] = variant["counterfactual"]["generation_accuracy"]
    shortcut_report["verified"] = (
        shortcut_report["shortcut_iid_validation"]["accuracy"] >= thresholds["cue_iid_accuracy"]
        and shortcut_report["counterfactual_test"]["accuracy"] <= thresholds["cue_counterfactual_max_accuracy"]
        and shortcut_report["main_counterfactual_generation_accuracy"] >= thresholds["counterfactual_accuracy"]
    )
    _write_json(result_dir / "shortcut_report.json", shortcut_report)

    # Short deterministic repeat.
    repro_digests: list[str] = []
    repro_sequences: list[list[dict[str, float]]] = []
    for name in ("a", "b"):
        set_deterministic_seed(seed + 31, cpu_threads=threads)
        repro_model = _new_model(config, tokenizer, len(answer_classes))
        _, repro_trainer = _new_trainer(
            repro_model,
            result_dir / f"repro_{name}",
            config,
            epochs=2,
            classification_weight=0.5,
            generation_weight=1.0,
        )
        repro_loader = _loader(overfit_dataset, batch_size=24, shuffle=True, seed=seed + 31)
        history = repro_trainer.fit(repro_loader, repro_loader)
        repro_digests.append(_state_digest(repro_model))
        repro_sequences.append([
            {"loss": row["train"]["loss"], "sequence_accuracy": row["train"]["sequence_accuracy"]}
            for row in history
        ])
    reproducibility_report = {
        "state_sha256": repro_digests,
        "state_equal": repro_digests[0] == repro_digests[1],
        "metrics_equal": repro_sequences[0] == repro_sequences[1],
    }
    reproducibility_report["verified"] = reproducibility_report["state_equal"] and reproducibility_report["metrics_equal"]
    _write_json(result_dir / "reproducibility_report.json", reproducibility_report)

    failures = [row for row in final_eval["predictions"] if not row["generation_correct"]]
    _write_json(result_dir / "failure_examples.json", failures[:30])
    _write_json(
        result_dir / "generation_traces.json",
        [
            {
                "record_id": row["record_id"],
                "question": row["question_zh"],
                "target": row["answer_zh"],
                "prediction": row["generation_prediction"],
                "token_ids": row["generated_token_ids"],
                "probabilities": row["generated_token_probabilities"],
            }
            for row in final_eval["predictions"][:12]
        ],
    )
    _save_curves(result_dir / "training_curves.png", classification_history, generation_history)
    _save_confusion(result_dir / "classification_confusion.png", final_eval["predictions"], answer_classes)
    _save_attention(
        result_dir / "cross_attention_examples.png",
        root,
        config["source_index"],
        model,
        datasets["test"],
        tokenizer,
    )

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    vae_parameters = int(json.loads((root / "results/P2-W02_image_vae/vae_metrics.json").read_text(encoding="utf-8"))["parameter_count"])
    elapsed = perf_counter() - started
    memory = process.memory_info()
    peak_rss = int(getattr(memory, "peak_wset", memory.rss))
    resource_report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "cpu_threads": threads,
        "wall_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "trainable_parameters": trainable_parameters,
        "frozen_vae_parameters": vae_parameters,
        "total_parameters": trainable_parameters + vae_parameters,
        "limits": {
            "max_wall_seconds": thresholds["max_wall_seconds"],
            "max_peak_rss_bytes": thresholds["max_peak_rss_bytes"],
            "max_trainable_parameters": thresholds["max_trainable_parameters"],
            "max_total_parameters": thresholds["max_total_parameters"],
        },
    }
    resource_report["verified"] = (
        elapsed <= thresholds["max_wall_seconds"]
        and peak_rss <= thresholds["max_peak_rss_bytes"]
        and trainable_parameters < thresholds["max_trainable_parameters"]
        and trainable_parameters + vae_parameters < thresholds["max_total_parameters"]
    )
    _write_json(result_dir / "resource_benchmark.json", resource_report)

    summary = {
        "data_verified": data_report["verified"],
        "overfit_verified": overfit_report["verified"],
        "classification_verified": classification_report["verified"],
        "generation_verified": generation_report["verified"],
        "single_modality_verified": baseline_report["verified"],
        "ablation_verified": ablation_report["verified"],
        "shortcut_verified": shortcut_report["verified"],
        "reproducibility_verified": reproducibility_report["verified"],
        "resource_verified": resource_report["verified"],
    }
    summary["verified"] = all(summary.values())
    _write_json(result_dir / "experiment_summary.json", summary)
    if not summary["verified"]:
        raise RuntimeError("one or more P2-W03 experiment gates failed")
    return summary
