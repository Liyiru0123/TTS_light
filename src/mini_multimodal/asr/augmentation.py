"""Deterministic waveform noise and lightweight SpecAugment for I002."""

from __future__ import annotations

import hashlib
import numpy as np


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def add_white_noise_at_snr(waveform: np.ndarray, snr_db: float, *, seed: int) -> tuple[np.ndarray, float]:
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 1 or len(wave) == 0 or not np.isfinite(wave).all():
        raise ValueError("waveform must be a finite non-empty mono array")
    signal_power = float(np.mean(wave.astype(np.float64) ** 2))
    if signal_power <= 0:
        raise ValueError("waveform has no signal power")
    generator = np.random.default_rng(seed)
    noise = generator.normal(0.0, 1.0, len(wave)).astype(np.float64)
    noise -= noise.mean()
    target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise *= np.sqrt(target_noise_power / np.mean(noise ** 2))
    mixed = wave.astype(np.float64) + noise
    measured = 10.0 * np.log10(signal_power / float(np.mean(noise ** 2)))
    return mixed.astype(np.float32), float(measured)


def spec_augment(
    feature: np.ndarray,
    *,
    record_id: str,
    epoch: int,
    seed: int,
    probability: float,
    frequency_mask_count: int,
    frequency_mask_max_width: int,
    time_mask_count: int,
    time_mask_max_width: int,
) -> np.ndarray:
    """Mask time/frequency blocks after normalization; shape and length are preserved."""
    value = np.asarray(feature, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("feature must have shape [time,frequency]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    generator = np.random.default_rng(stable_seed(seed, record_id, epoch))
    output = value.copy()
    if generator.random() >= probability:
        return output
    time_steps, frequencies = output.shape
    for _ in range(frequency_mask_count):
        width = int(generator.integers(1, min(frequency_mask_max_width, frequencies) + 1))
        start = int(generator.integers(0, frequencies - width + 1))
        output[:, start:start + width] = 0.0
    for _ in range(time_mask_count):
        width = int(generator.integers(1, min(time_mask_max_width, time_steps) + 1))
        start = int(generator.integers(0, time_steps - width + 1))
        output[start:start + width, :] = 0.0
    return output
