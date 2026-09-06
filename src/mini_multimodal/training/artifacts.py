"""End-to-end evidence generation for the P2-W01 training framework."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import matplotlib
import psutil
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .baseline import (
    ClassificationTask,
    ManifestImageDataset,
    TinyConvClassifier,
    cue_only_accuracy,
    learn_cue_mapping,
)
from .checkpoint import load_checkpoint
from .core import Trainer, TrainerConfig
from .data import build_p2_w01_data_spec, export_p2_w01_data
from .randomness import seeded_generator, set_deterministic_seed


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _loader(
    root: Path,
    entries: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    return DataLoader(
        ManifestImageDataset(root, entries),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=seeded_generator(seed) if shuffle else None,
    )


def _new_trainer(
    output_dir: Path,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> tuple[TinyConvClassifier, torch.optim.Optimizer, Trainer]:
    model = TinyConvClassifier()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=ClassificationTask(),
        config=TrainerConfig(
            epochs=epochs,
            device="cpu",
            gradient_clip_norm=gradient_clip_norm,
            monitor="accuracy",
            maximize=True,
        ),
        output_dir=output_dir,
    )
    return model, optimizer, trainer


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_curve(path: Path, history: list[Mapping[str, Any]]) -> None:
    epochs = [row["epoch"] + 1 for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].plot(epochs, [row["train"]["loss"] for row in history], label="train")
    axes[0].plot(
        epochs, [row["validation"]["loss"] for row in history], label="validation"
    )
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(
        epochs, [row["train"]["accuracy"] for row in history], label="train"
    )
    axes[1].plot(
        epochs,
        [row["validation"]["accuracy"] for row in history],
        label="validation",
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylim=(-0.02, 1.02))
    axes[1].legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _maximum_optimizer_step(optimizer: torch.optim.Optimizer) -> int:
    steps = []
    for state in optimizer.state.values():
        value = state.get("step")
        if value is not None:
            steps.append(int(value.item()) if torch.is_tensor(value) else int(value))
    return max(steps, default=0)


def _peak_rss(process: psutil.Process) -> int:
    memory = process.memory_info()
    return int(getattr(memory, "peak_wset", memory.rss))


def run_p2_w01_experiments(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_data = root / config["output_data"]
    output_results = root / config["output_results"]
    output_results.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    started = perf_counter()
    start_rss = process.memory_info().rss
    seed = int(config["seed"])
    baseline_config = config["baseline"]
    thresholds = config["thresholds"]
    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))

    data_manifest = export_p2_w01_data(
        root / config["source_index"],
        output_data,
        overfit_examples_per_class=int(baseline_config["overfit_examples_per_class"]),
    )
    routing, baseline = build_p2_w01_data_spec(
        root / config["source_index"],
        overfit_examples_per_class=int(baseline_config["overfit_examples_per_class"]),
    )
    _write_json(output_results / "task_routing_report.json", routing)

    # Component gate: balanced tiny-set overfit.
    overfit_dir = output_results / "overfit_run"
    overfit_loader = _loader(
        root,
        baseline["overfit"],
        batch_size=len(baseline["overfit"]),
        shuffle=True,
        seed=seed,
    )
    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))
    _, _, overfit_trainer = _new_trainer(
        overfit_dir,
        epochs=int(baseline_config["overfit_epochs"]),
        learning_rate=float(baseline_config["learning_rate"]),
        weight_decay=0.0,
        gradient_clip_norm=float(baseline_config["gradient_clip_norm"]),
    )
    overfit_history = overfit_trainer.fit(overfit_loader, overfit_loader)
    first_passing_epoch = next(
        (
            row["epoch"]
            for row in overfit_history
            if row["train"]["accuracy"] >= thresholds["overfit_accuracy"]
        ),
        None,
    )
    overfit_report = {
        "example_count": len(baseline["overfit"]),
        "threshold": thresholds["overfit_accuracy"],
        "first_passing_epoch": first_passing_epoch,
        "initial": overfit_history[0]["train"],
        "final": overfit_history[-1]["train"],
        "maximum_accuracy": max(row["train"]["accuracy"] for row in overfit_history),
        "verified": first_passing_epoch is not None,
    }
    _write_json(output_results / "overfit_report.json", overfit_report)
    if not overfit_report["verified"]:
        raise RuntimeError("small-data overfit gate failed; formal training is blocked")

    # Formal baseline training.
    train_loader = _loader(
        root,
        baseline["groups"]["train"],
        batch_size=int(baseline_config["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        root,
        baseline["groups"]["validation"],
        batch_size=64,
    )
    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))
    model, optimizer, trainer = _new_trainer(
        output_results,
        epochs=int(baseline_config["epochs"]),
        learning_rate=float(baseline_config["learning_rate"]),
        weight_decay=float(baseline_config["weight_decay"]),
        gradient_clip_norm=float(baseline_config["gradient_clip_norm"]),
    )
    history = trainer.fit(train_loader, validation_loader)
    best_state = load_checkpoint(
        output_results / "checkpoints" / "best.pt",
        model=model,
        optimizer=optimizer,
        restore_rng=False,
    )
    groups = {
        "validation_all": baseline["groups"]["validation"],
        "validation_shortcut_iid": [
            item
            for item in baseline["groups"]["validation"]
            if item["variant"] == "shortcut_iid"
        ],
        "validation_compositional": [
            item
            for item in baseline["groups"]["validation"]
            if item["variant"] == "compositional"
        ],
        "test_counterfactual": [
            item
            for item in baseline["groups"]["test"]
            if item["variant"] == "counterfactual"
        ],
        "test_compositional": [
            item
            for item in baseline["groups"]["test"]
            if item["variant"] == "compositional"
        ],
    }
    group_metrics = {
        name: trainer.evaluate(_loader(root, entries, batch_size=64))
        for name, entries in groups.items()
    }
    model_metrics = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(best_state["epoch"]),
        "best_validation_metric": float(best_state["best_metric"]),
        "groups": group_metrics,
        "thresholds": {
            "validation_all": thresholds["validation_accuracy"],
            "test_counterfactual": thresholds["counterfactual_accuracy"],
            "test_compositional": thresholds["compositional_accuracy"],
        },
    }
    model_metrics["verified"] = all(
        group_metrics[name]["accuracy"] >= threshold
        for name, threshold in model_metrics["thresholds"].items()
    )
    _write_json(output_results / "baseline_metrics.json", model_metrics)
    _save_curve(output_results / "training_curve.png", history)

    # Deliberately weak shortcut-only comparator.
    cue_mapping = learn_cue_mapping(baseline["groups"]["train"])
    iid_entries = groups["validation_shortcut_iid"]
    counterfactual_entries = groups["test_counterfactual"]
    shortcut_report = {
        "learned_mapping": cue_mapping,
        "validation_shortcut_iid": cue_only_accuracy(iid_entries, cue_mapping),
        "test_counterfactual": cue_only_accuracy(counterfactual_entries, cue_mapping),
        "thresholds": {
            "iid_min": thresholds["cue_only_iid_accuracy"],
            "counterfactual_max": thresholds["cue_only_counterfactual_max_accuracy"],
        },
    }
    shortcut_report["verified"] = (
        shortcut_report["validation_shortcut_iid"]["accuracy"]
        >= thresholds["cue_only_iid_accuracy"]
        and shortcut_report["test_counterfactual"]["accuracy"]
        <= thresholds["cue_only_counterfactual_max_accuracy"]
    )
    _write_json(output_results / "shortcut_diagnostic.json", shortcut_report)

    # Same-seed reproducibility on the small fixture.
    digests: list[str] = []
    metric_sequences: list[list[dict[str, Any]]] = []
    for run_name in ("a", "b"):
        set_deterministic_seed(seed + 11, cpu_threads=int(config["cpu_threads"]))
        loader = _loader(
            root,
            baseline["overfit"],
            batch_size=10,
            shuffle=True,
            seed=seed + 11,
        )
        repro_model, _, repro_trainer = _new_trainer(
            output_results / f"repro_run_{run_name}",
            epochs=3,
            learning_rate=float(baseline_config["learning_rate"]),
            weight_decay=0.0,
            gradient_clip_norm=float(baseline_config["gradient_clip_norm"]),
        )
        repro_history = repro_trainer.fit(loader, loader)
        digests.append(_state_digest(repro_model))
        metric_sequences.append(
            [
                {"train": row["train"], "validation": row["validation"]}
                for row in repro_history
            ]
        )
    reproducibility = {
        "seed": seed + 11,
        "state_sha256": digests,
        "state_equal": digests[0] == digests[1],
        "metrics_equal": metric_sequences[0] == metric_sequences[1],
    }
    reproducibility["verified"] = reproducibility["state_equal"] and reproducibility["metrics_equal"]
    _write_json(output_results / "reproducibility_report.json", reproducibility)

    # One-epoch checkpoint followed by one resumed epoch.
    resume_dir = output_results / "resume_check"
    set_deterministic_seed(seed + 23, cpu_threads=int(config["cpu_threads"]))
    resume_loader = _loader(
        root,
        baseline["overfit"],
        batch_size=10,
        shuffle=True,
        seed=seed + 23,
    )
    _, first_optimizer, first_trainer = _new_trainer(
        resume_dir,
        epochs=1,
        learning_rate=float(baseline_config["learning_rate"]),
        weight_decay=0.0,
        gradient_clip_norm=float(baseline_config["gradient_clip_norm"]),
    )
    first_trainer.fit(resume_loader, resume_loader)
    step_before = _maximum_optimizer_step(first_optimizer)
    _, resumed_optimizer, resumed_trainer = _new_trainer(
        resume_dir,
        epochs=2,
        learning_rate=float(baseline_config["learning_rate"]),
        weight_decay=0.0,
        gradient_clip_norm=float(baseline_config["gradient_clip_norm"]),
    )
    resumed_history = resumed_trainer.fit(
        resume_loader,
        resume_loader,
        resume_from=resume_dir / "checkpoints" / "last.pt",
    )
    step_after = _maximum_optimizer_step(resumed_optimizer)
    resume_rows = [
        json.loads(line)
        for line in (resume_dir / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    resume_report = {
        "logged_epochs": [row["epoch"] for row in resume_rows],
        "resumed_epoch": resumed_history[0]["epoch"],
        "optimizer_step_before": step_before,
        "optimizer_step_after": step_after,
        "verified": [row["epoch"] for row in resume_rows] == [0, 1]
        and resumed_history[0]["epoch"] == 1
        and step_after > step_before,
    }
    _write_json(output_results / "checkpoint_resume_report.json", resume_report)

    elapsed = perf_counter() - started
    resource_report = {
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "wall_seconds": elapsed,
        "start_rss_bytes": start_rss,
        "peak_rss_bytes": _peak_rss(process),
        "limits": {
            "max_wall_seconds": thresholds["max_wall_seconds"],
            "max_peak_rss_bytes": thresholds["max_peak_rss_bytes"],
        },
    }
    resource_report["verified"] = (
        elapsed <= thresholds["max_wall_seconds"]
        and resource_report["peak_rss_bytes"] <= thresholds["max_peak_rss_bytes"]
    )
    _write_json(output_results / "resource_benchmark.json", resource_report)

    summary = {
        "data_manifest": data_manifest,
        "routing_verified": routing["unrouted_count"] == 0
        and routing["image_qa"]["count"] == 8792
        and routing["video_qa"]["count"] == 11163,
        "overfit_verified": overfit_report["verified"],
        "baseline_verified": model_metrics["verified"],
        "shortcut_verified": shortcut_report["verified"],
        "reproducibility_verified": reproducibility["verified"],
        "resume_verified": resume_report["verified"],
        "resource_verified": resource_report["verified"],
    }
    summary["verified"] = all(
        value for key, value in summary.items() if key.endswith("_verified")
    )
    _write_json(output_results / "experiment_summary.json", summary)
    if not summary["verified"]:
        raise RuntimeError("one or more P2-W01 experiment gates failed")
    return summary

