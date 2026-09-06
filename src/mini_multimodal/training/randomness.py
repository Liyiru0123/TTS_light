"""Deterministic CPU execution and checkpointable random states."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_deterministic_seed(seed: int, *, cpu_threads: int | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cpu_threads is not None:
        if cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)


def capture_random_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator

