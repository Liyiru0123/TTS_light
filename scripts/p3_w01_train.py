"""Train and evaluate the preregistered P3-W01 I002 temporal model."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.training import Trainer, TrainerConfig, load_checkpoint  # noqa: E402
from mini_multimodal.video_mllm import (  # noqa: E402
    TinyVideoMLLM,
    TinyVideoMLLMTask,
    VideoQADataset,
    collate_video_qa,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def make_model(config: dict, dataset: VideoQADataset, answer_count: int) -> TinyVideoMLLM:
    tokenizer = dataset.tokenizer
    return TinyVideoMLLM(
        vocabulary_size=len(tokenizer),
        answer_class_count=answer_count,
        pad_id=tokenizer.id("<pad>"),
        bos_id=tokenizer.id("<bos>"),
        eos_id=tokenizer.id("<eos>"),
        mask_id=tokenizer.id("<mask>"),
        **config["model"],
    )


def make_loader(dataset: VideoQADataset, config: dict, *, shuffle: bool, seed: int, batch_size: int | None = None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size or int(config["training"]["batch_size"]),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=collate_video_qa,
    )


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def reproducibility_probe(config: dict, dataset: VideoQADataset, answer_count: int) -> dict[str, Any]:
    sequences = []
    digests = []
    subset = torch.utils.data.Subset(dataset, range(8))
    for _ in range(2):
        set_seed(int(config["seed"]) + 91, int(config["training"]["cpu_threads"]))
        model = make_model(config, dataset, answer_count)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        task = TinyVideoMLLMTask(pad_id=dataset.tokenizer.id("<pad>"))
        loader = make_loader(subset, config, shuffle=False, seed=int(config["seed"]), batch_size=8)
        losses = []
        for _step in range(3):
            batch = next(iter(loader))
            optimizer.zero_grad(set_to_none=True)
            output = task.step(model, batch, training=True)
            output.loss.backward()
            optimizer.step()
            losses.append(float(output.loss.detach()))
        sequences.append(losses)
        digests.append(state_digest(model))
    return {
        "loss_sequences": sequences,
        "parameter_sha256": digests,
        "identical": sequences[0] == sequences[1] and digests[0] == digests[1],
    }


def train_with_trainer(
    *,
    config: dict,
    model: TinyVideoMLLM,
    task: TinyVideoMLLMTask,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    output_dir: Path,
    epochs: int,
    monitor: str,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=task,
        config=TrainerConfig(
            epochs=epochs,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            monitor=monitor,
            maximize=True,
        ),
        output_dir=output_dir,
    )
    return trainer.fit(train_loader, validation_loader)


@torch.inference_mode()
def evaluate(
    model: TinyVideoMLLM,
    loader: DataLoader,
    tokenizer,
    answer_classes: list[str],
    *,
    mask_video: bool = False,
    single_frame_index: int | None = None,
    shuffle_frames: bool = False,
    disable_time_position: bool = False,
    disable_temporal_module: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions = []
    permutation = torch.tensor([2, 5, 0, 7, 4, 1, 6, 3])
    for batch in loader:
        video = batch["video"]
        if single_frame_index is not None:
            video = video[:, single_frame_index : single_frame_index + 1].expand(-1, 8, -1, -1, -1)
        if shuffle_frames:
            video = video[:, permutation]
        output = model(
            video,
            batch["question_ids"],
            mask_video=mask_video,
            disable_time_position=disable_time_position,
            disable_temporal_module=disable_temporal_module,
        )
        generated, probabilities = model.generate(
            video,
            batch["question_ids"],
            mask_video=mask_video,
            disable_time_position=disable_time_position,
            disable_temporal_module=disable_temporal_module,
        )
        class_ids = output.classification_logits.argmax(dim=-1)
        for index, metadata in enumerate(batch["metadata"]):
            classification_answer = answer_classes[int(class_ids[index])]
            generated_answer = tokenizer.decode_answer(generated[index].tolist())
            predictions.append(
                {
                    "record_id": metadata["record_id"],
                    "split": metadata["split"],
                    "question_kind": metadata["question_kind"],
                    "question_zh": metadata["question_zh"],
                    "reference_answer": metadata["answer_zh"],
                    "classification_answer": classification_answer,
                    "generated_answer": generated_answer,
                    "classification_correct": classification_answer == metadata["answer_zh"],
                    "generation_correct": generated_answer == metadata["answer_zh"],
                    "is_fully_correct": classification_answer == metadata["answer_zh"] and generated_answer == metadata["answer_zh"],
                    "classification_probabilities": output.classification_logits[index].softmax(dim=-1).tolist(),
                    "generated_token_ids": generated[index].tolist(),
                    "generated_token_probabilities": probabilities[index].tolist(),
                    "pair_id": metadata["pair_id"],
                    "partner_record_id": metadata["partner_record_id"],
                    "group_id": metadata["group_id"],
                }
            )
    total = len(predictions)
    by_kind = defaultdict(list)
    for item in predictions:
        by_kind[item["question_kind"]].append(item)
    kind_metrics = {
        kind: {
            "sample_count": len(items),
            "classification_accuracy": sum(item["classification_correct"] for item in items) / len(items),
            "generation_accuracy": sum(item["generation_correct"] for item in items) / len(items),
        }
        for kind, items in sorted(by_kind.items())
    }
    metrics = {
        "sample_count": total,
        "classification_accuracy": sum(item["classification_correct"] for item in predictions) / total,
        "generation_accuracy": sum(item["generation_correct"] for item in predictions) / total,
        "fully_correct_rate": sum(item["is_fully_correct"] for item in predictions) / total,
        "minimum_question_kind_classification_accuracy": min(
            item["classification_accuracy"] for item in kind_metrics.values()
        ),
        "question_kind": kind_metrics,
    }
    return metrics, predictions


def main() -> None:
    config = json.loads((ROOT / "configs/p3_w01_tiny_video_mllm.json").read_text(encoding="utf-8"))
    branch = ROOT / "results/P3-W01_tiny_video_mllm/iterations/I002/branches/temporal_core"
    metrics_dir = branch / "metrics"
    predictions_dir = branch / "predictions"
    logs_dir = branch / "logs"
    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    set_seed(int(config["seed"]), int(config["training"]["cpu_threads"]))

    train_data = VideoQADataset(ROOT, data_dir=config["data"]["output_dir"], splits="train")
    validation_data = VideoQADataset(ROOT, data_dir=config["data"]["output_dir"], splits="validation")
    test_data = VideoQADataset(ROOT, data_dir=config["data"]["output_dir"], splits="test")
    answer_classes = json.loads((ROOT / config["data"]["output_dir"] / "answer_classes.json").read_text(encoding="utf-8"))
    tokenizer = train_data.tokenizer
    train_loader = make_loader(train_data, config, shuffle=True, seed=int(config["seed"]))
    validation_loader = make_loader(validation_data, config, shuffle=False, seed=int(config["seed"]))
    test_loader = make_loader(test_data, config, shuffle=False, seed=int(config["seed"]))

    write_json(logs_dir / "reproducibility_report.json", reproducibility_probe(config, train_data, len(answer_classes)))

    overfit_ids = json.loads((ROOT / config["data"]["overfit_ids_path"]).read_text(encoding="utf-8"))
    overfit_data = VideoQADataset(ROOT, data_dir=config["data"]["output_dir"], record_ids=overfit_ids)
    overfit_loader = make_loader(overfit_data, config, shuffle=True, seed=int(config["seed"]) + 1, batch_size=16)
    set_seed(int(config["seed"]) + 1, int(config["training"]["cpu_threads"]))
    overfit_model = make_model(config, overfit_data, len(answer_classes))
    overfit_history = train_with_trainer(
        config=config,
        model=overfit_model,
        task=TinyVideoMLLMTask(pad_id=tokenizer.id("<pad>")),
        train_loader=overfit_loader,
        validation_loader=overfit_loader,
        output_dir=logs_dir / "overfit",
        epochs=int(config["training"]["overfit_epochs"]),
        monitor="joint_score",
    )
    load_checkpoint(logs_dir / "overfit/checkpoints/best.pt", model=overfit_model, restore_rng=False)
    overfit_metrics, _ = evaluate(overfit_model, overfit_loader, tokenizer, answer_classes)
    write_json(metrics_dir / "overfit_report.json", {**overfit_metrics, "epochs": len(overfit_history)})
    del overfit_model, overfit_data, overfit_loader

    set_seed(int(config["seed"]), int(config["training"]["cpu_threads"]))
    model = make_model(config, train_data, len(answer_classes))
    history = train_with_trainer(
        config=config,
        model=model,
        task=TinyVideoMLLMTask(
            pad_id=tokenizer.id("<pad>"),
            classification_weight=float(config["training"]["classification_loss_weight"]),
            generation_weight=float(config["training"]["generation_loss_weight"]),
        ),
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_dir=branch,
        epochs=int(config["training"]["epochs"]),
        monitor="joint_score",
    )
    load_checkpoint(branch / "checkpoints/best.pt", model=model, restore_rng=False)
    write_json(logs_dir / "training_history.json", history)
    evaluation, predictions = evaluate(model, test_loader, tokenizer, answer_classes)
    write_json(metrics_dir / "evaluation_metrics.json", evaluation)
    (predictions_dir / "test_predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in predictions),
        encoding="utf-8",
        newline="\n",
    )

    set_seed(int(config["seed"]) + 2, int(config["training"]["cpu_threads"]))
    text_model = make_model(config, train_data, len(answer_classes))
    train_with_trainer(
        config=config,
        model=text_model,
        task=TinyVideoMLLMTask(pad_id=tokenizer.id("<pad>"), generation_weight=0.0, mask_video=True),
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_dir=logs_dir / "text_baseline",
        epochs=int(config["training"]["baseline_epochs"]),
        monitor="classification_accuracy",
    )
    load_checkpoint(logs_dir / "text_baseline/checkpoints/best.pt", model=text_model, restore_rng=False)
    text_metrics, _ = evaluate(text_model, test_loader, tokenizer, answer_classes, mask_video=True)

    set_seed(int(config["seed"]) + 3, int(config["training"]["cpu_threads"]))
    single_model = make_model(config, train_data, len(answer_classes))
    train_with_trainer(
        config=config,
        model=single_model,
        task=TinyVideoMLLMTask(pad_id=tokenizer.id("<pad>"), generation_weight=0.0, single_frame_index=3),
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_dir=logs_dir / "single_frame_baseline",
        epochs=int(config["training"]["baseline_epochs"]),
        monitor="classification_accuracy",
    )
    load_checkpoint(logs_dir / "single_frame_baseline/checkpoints/best.pt", model=single_model, restore_rng=False)
    single_metrics, _ = evaluate(single_model, test_loader, tokenizer, answer_classes, single_frame_index=3)
    baseline = {
        "full_classification_accuracy": evaluation["classification_accuracy"],
        "text_only": text_metrics,
        "single_frame_index_3": single_metrics,
        "advantage_over_text": evaluation["classification_accuracy"] - text_metrics["classification_accuracy"],
        "advantage_over_single_frame": evaluation["classification_accuracy"] - single_metrics["classification_accuracy"],
        "baseline_training_epochs": int(config["training"]["baseline_epochs"]),
    }
    write_json(metrics_dir / "baseline_metrics.json", baseline)

    shuffled, _ = evaluate(model, test_loader, tokenizer, answer_classes, shuffle_frames=True)
    no_position, _ = evaluate(model, test_loader, tokenizer, answer_classes, disable_time_position=True)
    no_temporal, _ = evaluate(model, test_loader, tokenizer, answer_classes, disable_temporal_module=True)
    ablation = {
        "full_classification_accuracy": evaluation["classification_accuracy"],
        "frame_shuffled": {**shuffled, "classification_drop": evaluation["classification_accuracy"] - shuffled["classification_accuracy"]},
        "time_position_disabled": {**no_position, "classification_drop": evaluation["classification_accuracy"] - no_position["classification_accuracy"]},
        "temporal_module_disabled": {**no_temporal, "classification_drop": evaluation["classification_accuracy"] - no_temporal["classification_accuracy"]},
    }
    write_json(metrics_dir / "ablation_metrics.json", ablation)

    by_id = {item["record_id"]: item for item in predictions}
    pairs = defaultdict(list)
    for item in predictions:
        if item["pair_id"]:
            pairs[item["pair_id"]].append(item)
    pair_rows = []
    for pair_id, items in sorted(pairs.items()):
        if len(items) != 2:
            raise ValueError(f"reversal pair does not contain two records: {pair_id}")
        pair_rows.append(
            {
                "pair_id": pair_id,
                "record_ids": [item["record_id"] for item in items],
                "reference_answers": [item["reference_answer"] for item in items],
                "classification_answers": [item["classification_answer"] for item in items],
                "both_classification_correct": all(item["classification_correct"] for item in items),
                "predictions_are_different": items[0]["classification_answer"] != items[1]["classification_answer"],
            }
        )
    reversal = {
        "pair_count": len(pair_rows),
        "pair_consistency_accuracy": sum(item["both_classification_correct"] for item in pair_rows) / len(pair_rows),
        "prediction_sensitivity_rate": sum(item["predictions_are_different"] for item in pair_rows) / len(pair_rows),
        "pairs": pair_rows,
    }
    write_json(metrics_dir / "reversal_metrics.json", reversal)

    elapsed = time.perf_counter() - started
    memory = process.memory_info()
    resource = {
        "wall_seconds": elapsed,
        "peak_rss_bytes": int(getattr(memory, "peak_wset", memory.rss)),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "cpu_threads": int(config["training"]["cpu_threads"]),
    }
    write_json(metrics_dir / "resource_benchmark.json", resource)
    write_json(
        logs_dir / "run_summary.json",
        {
            "epochs_completed": len(history),
            "best_epoch": int(torch.load(branch / "checkpoints/best.pt", map_location="cpu", weights_only=False)["epoch"]),
            "checkpoint_sha256": {name: sha256(branch / "checkpoints" / name) for name in ("best.pt", "last.pt")},
            "answer_distribution": dict(sorted(Counter(item["classification_answer"] for item in predictions).items())),
        },
    )
    print(json.dumps({"evaluation": evaluation, "baseline": baseline, "ablation": ablation, "reversal": {k: v for k, v in reversal.items() if k != "pairs"}, "resource": resource}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
