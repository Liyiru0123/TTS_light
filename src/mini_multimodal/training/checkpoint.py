"""Atomic, resumable PyTorch checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .randomness import capture_random_state, restore_random_state


CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "config": dict(config),
            "random_state": capture_random_state(),
        },
        temporary,
    )
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    model.load_state_dict(state["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if restore_rng:
        restore_random_state(state["random_state"])
    return state

