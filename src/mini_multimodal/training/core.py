"""Task-agnostic CPU trainer with weighted metrics and resumable checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Protocol

import torch

from .checkpoint import load_checkpoint, save_checkpoint


@dataclass(slots=True)
class StepOutput:
    loss: torch.Tensor
    metrics: Mapping[str, float]
    weight: int

    def __post_init__(self) -> None:
        if self.loss.ndim != 0:
            raise ValueError("step loss must be a scalar tensor")
        if self.weight <= 0:
            raise ValueError("step weight must be positive")


class TrainTask(Protocol):
    def step(
        self, model: torch.nn.Module, batch: Any, *, training: bool
    ) -> StepOutput: ...


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    epochs: int
    device: str = "cpu"
    gradient_clip_norm: float | None = None
    monitor: str = "accuracy"
    maximize: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.device != "cpu":
            raise ValueError("P2-W01 trainer currently permits CPU only")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


def _aggregate(outputs: list[StepOutput]) -> dict[str, float]:
    if not outputs:
        raise ValueError("loader yielded no batches")
    totals: dict[str, float] = {"loss": 0.0}
    total_weight = sum(output.weight for output in outputs)
    for output in outputs:
        totals["loss"] += float(output.loss.detach()) * output.weight
        for key, value in output.metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite metric: {key}")
            totals[key] = totals.get(key, 0.0) + float(value) * output.weight
    return {key: value / total_weight for key, value in totals.items()}


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        task: TrainTask,
        config: TrainerConfig,
        output_dir: str | Path,
    ) -> None:
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.task = task
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_path = self.output_dir / "training_log.jsonl"

    def run_epoch(self, loader: Iterable[Any], *, training: bool) -> dict[str, float]:
        self.model.train(training)
        outputs: list[StepOutput] = []
        for batch in loader:
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                output = self.task.step(self.model, batch, training=True)
                if not torch.isfinite(output.loss):
                    raise ValueError("non-finite training loss")
                output.loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                ]
                if not gradients:
                    raise ValueError("training step produced no gradients")
                if not all(torch.isfinite(gradient).all() for gradient in gradients):
                    raise ValueError("training step produced non-finite gradients")
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output = self.task.step(self.model, batch, training=False)
            outputs.append(output)
        return _aggregate(outputs)

    def evaluate(self, loader: Iterable[Any]) -> dict[str, float]:
        return self.run_epoch(loader, training=False)

    def fit(
        self,
        train_loader: Iterable[Any],
        validation_loader: Iterable[Any],
        *,
        resume_from: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        start_epoch = 0
        best = -math.inf if self.config.maximize else math.inf
        if resume_from is not None:
            state = load_checkpoint(
                resume_from,
                model=self.model,
                optimizer=self.optimizer,
                restore_rng=True,
            )
            start_epoch = int(state["epoch"]) + 1
            best = float(state["best_metric"])
        mode = "a" if resume_from is not None else "w"
        history: list[dict[str, Any]] = []
        with self.log_path.open(mode, encoding="utf-8", newline="\n") as log:
            for epoch in range(start_epoch, self.config.epochs):
                started = perf_counter()
                train_metrics = self.run_epoch(train_loader, training=True)
                validation_metrics = self.evaluate(validation_loader)
                monitored = validation_metrics.get(self.config.monitor)
                if monitored is None:
                    raise KeyError(f"monitor metric absent: {self.config.monitor}")
                improved = monitored > best if self.config.maximize else monitored < best
                if improved:
                    best = monitored
                row = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "monitor": self.config.monitor,
                    "best_metric": best,
                    "improved": improved,
                    "seconds": perf_counter() - started,
                }
                log.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                log.flush()
                checkpoint_config = asdict(self.config)
                save_checkpoint(
                    self.checkpoint_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_metric=best,
                    config=checkpoint_config,
                )
                if improved:
                    save_checkpoint(
                        self.checkpoint_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        best_metric=best,
                        config=checkpoint_config,
                    )
                history.append(row)
        return history

