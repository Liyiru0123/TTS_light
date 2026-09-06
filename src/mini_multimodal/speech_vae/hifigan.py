"""Frozen HiFi-GAN V3 generator and its feature-matched Mel transform.

The generator architecture is a dependency-light port of TorchAudio 2.8's
HiFi-GAN implementation, which in turn follows the original HiFi-GAN source.
The original model code is released under the MIT License; see the bundled
license and provenance files in ``models/external/hifigan_v3_ljspeech``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock2(nn.Module):
    """The compact two-convolution residual block used by HiFi-GAN V3."""

    def __init__(self, channels: int, kernel_size: int, dilations: Sequence[int], slope: float) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                padding=_same_padding(kernel_size, dilation),
            )
            for dilation in dilations
        )
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(F.leaky_relu(x, self.slope))
        return x


class HiFiGANV3(nn.Module):
    """Small HiFi-GAN V3 waveform generator compatible with official weights."""

    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: Sequence[int] = (8, 8, 4),
        upsample_kernel_sizes: Sequence[int] = (16, 16, 8),
        upsample_initial_channel: int = 256,
        resblock_kernel_sizes: Sequence[int] = (3, 5, 7),
        resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 2), (2, 6), (3, 12)),
        lrelu_slope: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(
                upsample_initial_channel // (2**index),
                upsample_initial_channel // (2 ** (index + 1)),
                kernel_size,
                stride=rate,
                padding=(kernel_size - rate) // 2,
            )
            for index, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        )
        blocks: list[nn.Module] = []
        for index in range(len(self.ups)):
            channels = upsample_initial_channel // (2 ** (index + 1))
            for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                blocks.append(ResBlock2(channels, kernel_size, dilations, lrelu_slope))
        self.resblocks = nn.ModuleList(blocks)
        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3)
        self.lrelu_slope = lrelu_slope

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(features)
        for index, upsample in enumerate(self.ups):
            x = upsample(F.leaky_relu(x, self.lrelu_slope))
            residual_sum = torch.zeros_like(x)
            for kernel_index in range(self.num_kernels):
                residual_sum = residual_sum + self.resblocks[index * self.num_kernels + kernel_index](x)
            x = residual_sum / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)


class HiFiGANMelTransform(nn.Module):
    """Waveform-to-Log-Mel transform matched to the pretrained V3 bundle."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        log_floor: float = 1e-5,
    ) -> None:
        super().__init__()
        self.reflect_padding = (n_fft - hop_length) // 2
        self.log_floor = log_floor
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            normalized=False,
            pad=0,
            mel_scale="slaney",
            norm="slaney",
            center=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"Expected [batch, time], got {tuple(waveform.shape)}")
        padded = F.pad(waveform.unsqueeze(1), (self.reflect_padding, self.reflect_padding), mode="reflect")
        padded = padded.squeeze(1)
        magnitude = (self.mel.spectrogram(padded) + 1e-9).sqrt()
        mel = self.mel.mel_scale(magnitude)
        return torch.log(torch.clamp(mel, min=self.log_floor))


def load_frozen_hifigan_v3(weight_path: str) -> HiFiGANV3:
    """Strictly load the published generator and freeze all parameters."""

    model = HiFiGANV3()
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model

