"""Conditional Flow Matching (linear CFM) training step and ODE inference.

Linear path: x_t = (1-t) x_0 + t x_1, x_0 ~ N(0, I), x_1 = posterior mean.
Target velocity: v = x_1 - x_0. Model predicts v_theta(x_t, t, c).

The condition c is the frame-wise condition built from text + (ground-truth or
predicted) duration. For Gate B we use ground-truth duration to isolate the CFM.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


def scale_durations(durations: torch.Tensor, factor: float, *, t_max: int = 64) -> torch.Tensor:
    """Scale durations with an exact global total while preserving zero tokens.

    Positive-duration tokens keep at least one frame. Remaining frames are
    distributed by largest remainder, avoiding weak/noisy per-token rounding.
    """
    if durations.ndim != 1 or factor <= 0:
        raise ValueError("durations must be 1-D and factor must be positive")
    source = durations.to(dtype=torch.long)
    positive = source > 0
    count = int(positive.sum().item())
    total = int(source.sum().item())
    if count == 0 or total == 0:
        raise ValueError("at least one positive duration is required")
    target = max(count, int(round(total * float(factor))))
    if target > t_max:
        raise ValueError(f"scaled duration {target} exceeds t_max {t_max}")
    raw = source.float() * (target / total)
    result = torch.floor(raw).long()
    result[positive] = result[positive].clamp_min(1)
    result[~positive] = 0
    while int(result.sum()) < target:
        candidates = torch.where(positive, raw - result.float(), torch.full_like(raw, -float("inf")))
        result[int(candidates.argmax())] += 1
    while int(result.sum()) > target:
        removable = positive & (result > 1)
        if not removable.any():
            raise ValueError("cannot satisfy scaled duration target")
        candidates = torch.where(removable, raw - result.float(), torch.full_like(raw, float("inf")))
        result[int(candidates.argmin())] -= 1
    return result


def text_to_token_ids(text_chars: Sequence[str], vocab: dict) -> list[int]:
    ids = []
    for ch in text_chars:
        if ch in vocab:
            ids.append(vocab[ch])
    return ids


def build_condition(model: nn.Module, token_ids: list[int], durations: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return (C [80, 64], T_valid) from token_ids and (ground-truth) durations."""
    h = model.encode_text(torch.tensor(token_ids, dtype=torch.long))  # [N, 64]
    H = model.regulator(h, durations)  # [T_valid, 64]
    C = model.condition_frames(H)  # [80, 64]
    T_valid = int(durations.sum().item())
    return C, T_valid


def cfm_loss(
    model: nn.Module,
    x1: torch.Tensor,
    C: torch.Tensor,
    t_valid: int,
    t_max: int = 64,
) -> tuple[torch.Tensor, float]:
    """Single-sample CFM loss. x1: [80, t_max] (valid region filled, rest 0)."""
    B = 1
    mask = torch.zeros(t_max)
    mask[:t_valid] = 1.0
    x0 = torch.randn_like(x1) * mask[None, :]
    t = torch.rand(1) * 0.999 + 0.001  # avoid exact 0/1
    t_scalar = t.item()
    x_t = (1.0 - t_scalar) * x0 + t_scalar * x1
    v_pred = model.velocity(x_t[None], t, C[None], mask[None])  # [1, 80, t_max]
    v_target = x1 - x0
    # masked MSE over valid region
    diff = (v_pred[0] - v_target) ** 2 * mask[None, :]
    loss = diff.sum() / (x1.shape[0] * t_valid + 1e-9)
    return loss, t_scalar


def ode_infer(
    model: nn.Module,
    C: torch.Tensor,
    t_valid: int,
    t_max: int = 64,
    ode_steps: int = 20,
    seed: int | None = None,
) -> torch.Tensor:
    """Integrate the ODE from t=0 (noise) to t=1 (data). Returns x1 [80, t_max]."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    mask = torch.zeros(t_max)
    mask[:t_valid] = 1.0
    x = torch.randn(t_max, 80, generator=gen).transpose(0, 1) * mask[None, :]  # [80, t_max]
    ts = [i / ode_steps for i in range(1, ode_steps + 1)]
    for i, t in enumerate(ts):
        dt = 1.0 / ode_steps
        v = model.velocity(x[None], torch.tensor([t]), C[None], mask[None])[0]
        x = (x + dt * v) * mask[None, :]
    return x
