"""Padding and layout conversion for model-ready multimodal batches."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _copy_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(array, copy=True))


def collate_multimodal(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one sample is required")
    result: dict[str, Any] = {
        "metadata": [sample["metadata"] for sample in samples],
        "scene": [sample["scene"] for sample in samples],
    }
    for key in ("text", "task", "annotations"):
        if key in samples[0]:
            if any(key not in sample for sample in samples):
                raise ValueError(f"all samples must consistently contain {key!r}")
            result[key] = [sample[key] for sample in samples]

    if "image" in samples[0]:
        images = np.stack([sample["image"] for sample in samples])
        result["image"] = _copy_tensor(images).permute(0, 3, 1, 2).float() / 255.0

    if "video" in samples[0]:
        videos = np.stack([sample["video"] for sample in samples])
        result["video"] = (
            _copy_tensor(videos).permute(0, 1, 4, 2, 3).float() / 255.0
        )

    if "waveform" in samples[0]:
        waveforms = [np.asarray(sample["waveform"], dtype=np.float32) for sample in samples]
        lengths = torch.tensor([len(item) for item in waveforms], dtype=torch.long)
        maximum = int(lengths.max())
        padded = torch.zeros((len(waveforms), maximum), dtype=torch.float32)
        mask = torch.zeros((len(waveforms), maximum), dtype=torch.bool)
        for index, waveform in enumerate(waveforms):
            length = len(waveform)
            padded[index, :length] = _copy_tensor(waveform)
            mask[index, :length] = True
        result.update(
            waveform=padded,
            waveform_lengths=lengths,
            waveform_mask=mask,
        )

    if "logmel" in samples[0]:
        features = [np.asarray(sample["logmel"], dtype=np.float32) for sample in samples]
        frequencies = {item.shape[0] for item in features}
        if len(frequencies) != 1 or any(item.ndim != 2 for item in features):
            raise ValueError("all log-mel arrays must share [frequency, time] layout")
        lengths = torch.tensor([item.shape[1] for item in features], dtype=torch.long)
        maximum = int(lengths.max())
        padded = torch.zeros(
            (len(features), next(iter(frequencies)), maximum), dtype=torch.float32
        )
        mask = torch.zeros((len(features), maximum), dtype=torch.bool)
        for index, feature in enumerate(features):
            length = feature.shape[1]
            padded[index, :, :length] = _copy_tensor(feature)
            mask[index, :length] = True
        result.update(logmel=padded, logmel_lengths=lengths, logmel_mask=mask)

    return result
