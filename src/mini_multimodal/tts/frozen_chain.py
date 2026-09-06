"""Frozen P3-W02 speech chain: 80-mel window <-> latent <-> mel <-> waveform.

Wraps the frozen TinySpeechVAE (P3-W02 inference package) and the frozen
HiFi-GAN V3 generator. Used by TinyTTS to:
- encode a unit's mel window to the posterior mean (training target x_1)
- decode a generated latent to mel and then to waveform (inference/eval)

All parameters are frozen; nothing here is trained in P3-W03.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mini_multimodal.speech_vae.hifigan import load_frozen_hifigan_v3
from mini_multimodal.speech_vae.model import TinySpeechVAE


class FrozenSpeechChain:
    """VAE (encode/decode) + HiFi-GAN (mel->waveform), all frozen."""

    def __init__(
        self,
        package_path: str,
        hifigan_weight_path: str,
        padding_value: float = -11.512925148010254,
        minimum_logmel: float = -11.512925148010254,
        maximum_logmel: float = 2.5,
        window_frames: int = 256,
        n_mels: int = 80,
        latent_channels: int = 4,
        latent_freq: int = 20,
        t_max: int = 64,
        sample_rate: int = 22050,
    ):
        self.padding_value = padding_value
        self.minimum_logmel = minimum_logmel
        self.maximum_logmel = maximum_logmel
        self.window_frames = window_frames
        self.n_mels = n_mels
        self.latent_channels = latent_channels
        self.latent_freq = latent_freq
        self.t_max = t_max
        self.sample_rate = sample_rate

        pkg = torch.load(package_path, map_location="cpu", weights_only=False)
        cfg = pkg["model_config"]
        self.vae = TinySpeechVAE(
            base_channels=int(cfg["base_channels"]),
            latent_channels=int(cfg["latent_channels"]),
            logvar_min=float(cfg["logvar_min"]),
            logvar_max=float(cfg["logvar_max"]),
            n_mels=int(cfg["n_mels"]),
            window_frames=int(cfg["window_frames"]),
        )
        self.vae.load_state_dict(pkg["model_state"], strict=True)
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.hifigan = load_frozen_hifigan_v3(hifigan_weight_path)

    def _normalize(self, mel: torch.Tensor) -> torch.Tensor:
        """raw log-mel -> [-1, 1] (P3-W02 convention)."""
        low, high = self.minimum_logmel, self.maximum_logmel
        return torch.clip((mel - low) / (high - low) * 2.0 - 1.0, -1.0, 1.0)

    def _denormalize(self, mel: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> raw log-mel (P3-W02 convention)."""
        low, high = self.minimum_logmel, self.maximum_logmel
        return (mel + 1.0) / 2.0 * (high - low) + low

    def prepare_window(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [n_mels, L] raw log-mel (L <= window_frames) -> [1,1,n_mels,window_frames] normalized+padding."""
        if mel.ndim != 2 or mel.shape[0] != self.n_mels:
            raise ValueError(f"mel must be [{self.n_mels}, L], got {tuple(mel.shape)}")
        L = mel.shape[1]
        if L > self.window_frames:
            raise ValueError(f"mel length {L} exceeds window_frames {self.window_frames}")
        window = torch.full((1, 1, self.n_mels, self.window_frames), self.padding_value, dtype=torch.float32)
        window[:, :, :, :L] = mel.to(torch.float32)
        return self._normalize(window)

    @torch.inference_mode()
    def posterior_mean(self, mel_window: torch.Tensor) -> torch.Tensor:
        """mel_window: [1, n_mels, window_frames] -> mean [latent_channels, latent_freq, t_max]."""
        mean, _ = self.vae.encode(mel_window)
        return mean[0]  # [4, 20, 64]

    @torch.inference_mode()
    def encode_unit(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [n_mels, L] -> posterior mean [4, 20, 64]."""
        return self.posterior_mean(self.prepare_window(mel))

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [4, 20, 64] -> raw log-mel [n_mels, window_frames]."""
        if latent.ndim != 3 or latent.shape != (self.latent_channels, self.latent_freq, self.t_max):
            raise ValueError(f"latent must be [{self.latent_channels},{self.latent_freq},{self.t_max}], got {tuple(latent.shape)}")
        out = self.vae.decode(latent[None])  # [1, 1, n_mels, window_frames] (normalized)
        return self._denormalize(out[0, 0])  # [n_mels, window_frames] (raw log-mel)

    @torch.inference_mode()
    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [n_mels, L] -> waveform [1, L*256]."""
        if mel.ndim != 2 or mel.shape[0] != self.n_mels:
            raise ValueError(f"mel must be [{self.n_mels}, L], got {tuple(mel.shape)}")
        return self.hifigan(mel[None])  # [1, n_mels, L] -> [1, 1, L*256]

    @torch.inference_mode()
    def latent_to_waveform(self, latent: torch.Tensor, valid_mel_frames: int) -> torch.Tensor:
        """latent: [4, 20, 64] -> waveform cropped to valid_mel_frames*256 samples."""
        mel = self.decode_latent(latent)  # [n_mels, 256]
        mel = mel[:, :valid_mel_frames]
        return self.synthesize(mel)  # [1, valid_mel_frames*256]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.vae.parameters()) + sum(p.numel() for p in self.hifigan.parameters())


def load_frozen_chain(cfg: dict, root: Path) -> FrozenSpeechChain:
    return FrozenSpeechChain(
        package_path=str(root / cfg["speech_vae_package"]),
        hifigan_weight_path=str(root / cfg["hifigan_weight"]),
        padding_value=float(cfg["feature"]["padding_value"]),
        minimum_logmel=float(cfg["feature"]["minimum_logmel"]),
        maximum_logmel=float(cfg["feature"]["maximum_logmel"]),
        window_frames=int(cfg["container"]["mel_window"]),
        n_mels=int(cfg["feature"]["n_mels"]),
        latent_channels=int(cfg["container"]["latent_channels"]),
        latent_freq=int(cfg["container"]["latent_freq"]),
        t_max=int(cfg["container"]["t_max"]),
        sample_rate=int(cfg["feature"]["sample_rate"]),
    )
