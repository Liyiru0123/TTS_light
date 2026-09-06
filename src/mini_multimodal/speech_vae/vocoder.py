"""Explicit, deterministic Log-Mel to waveform reconstruction."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

from mini_multimodal.speech.features import mel_filterbank


def logmel_to_waveform(
    logmel: np.ndarray | torch.Tensor,
    *,
    sample_rate: int = 22050,
    n_fft: int = 512,
    hop_length: int = 256,
    fmin: float = 40.0,
    fmax: float = 8000.0,
    n_iter: int = 24,
    momentum: float = 0.99,
) -> torch.Tensor:
    """Invert the project's natural-log Mel power with a fixed Griffin-Lim chain."""

    value = torch.as_tensor(logmel, dtype=torch.float32, device="cpu")
    if value.ndim != 2 or value.shape[0] <= 1 or value.shape[1] <= 1:
        raise ValueError("logmel must have shape [n_mels,time] with both dimensions > 1")
    if not torch.isfinite(value).all():
        raise ValueError("logmel must be finite")
    bank = mel_filterbank(sample_rate, n_fft, value.shape[0], fmin, fmax)
    inverse = torch.from_numpy(np.linalg.pinv(bank, rcond=1e-5).astype(np.float32))
    mel_power = value.clamp(-23.025850296020508, 12.0).exp()
    linear_power = (inverse @ mel_power).clamp_min(1e-10)
    magnitude = linear_power.sqrt()
    # torchaudio's centered ISTFT requires a COLA-compatible periodic Hann window.
    # P1 used NumPy's symmetric Hann for analysis; the small endpoint difference is
    # explicitly included in the measured waveform-cycle error.
    window = torch.hann_window(n_fft, periodic=True)
    waveform = torchaudio.functional.griffinlim(
        magnitude,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=1.0,
        n_iter=int(n_iter),
        momentum=float(momentum),
        length=None,
        rand_init=False,
    )
    waveform = waveform / waveform.abs().max().clamp_min(1.0)
    if not torch.isfinite(waveform).all():
        raise RuntimeError("waveform reconstruction produced non-finite values")
    return waveform
