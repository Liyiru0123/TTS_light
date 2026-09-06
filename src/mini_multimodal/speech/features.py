"""Transparent NumPy Log-Mel extraction."""

from __future__ import annotations
import numpy as np


def _hz_to_mel(hz): return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)
def _mel_to_hz(mel): return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    if not 0 <= fmin < fmax <= sample_rate / 2: raise ValueError("Invalid Mel frequency bounds")
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_points) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bins[m:m+3]
        if center <= left: center = left + 1
        if right <= center: right = center + 1
        for k in range(left, min(center, bank.shape[1])): bank[m,k] = (k-left)/(center-left)
        for k in range(center, min(right, bank.shape[1])): bank[m,k] = (right-k)/(right-center)
    return bank


def log_mel(waveform: np.ndarray, sample_rate: int, *, n_fft: int=512, hop_length: int=256, n_mels: int=64, fmin: float=40.0, fmax: float=8000.0) -> np.ndarray:
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 1 or not np.isfinite(wave).all(): raise ValueError("Waveform must be finite mono audio")
    if len(wave) < n_fft: wave = np.pad(wave, (0, n_fft-len(wave)))
    frame_count = 1 + (len(wave)-n_fft)//hop_length
    frames = np.lib.stride_tricks.sliding_window_view(wave, n_fft)[::hop_length][:frame_count]
    spectrum = np.fft.rfft(frames * np.hanning(n_fft), axis=1)
    power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    mel = mel_filterbank(sample_rate,n_fft,n_mels,fmin,fmax) @ power.T
    return np.log(np.maximum(mel, 1e-10)).astype(np.float32)
