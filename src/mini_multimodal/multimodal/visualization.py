"""Compact visual evidence for one four-modality sample."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_sample_overview(sample: Mapping[str, Any]) -> plt.Figure:
    required = {"image", "logmel", "video"}
    if not required.issubset(sample):
        raise ValueError("overview requires image, logmel, and video modalities")
    image = np.asarray(sample["image"])
    logmel = np.asarray(sample["logmel"])
    video = np.asarray(sample["video"])
    strip = np.concatenate(list(video), axis=1)

    figure, axes = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Image at frame 0")
    axes[1].imshow(logmel, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("Log-Mel spectrogram")
    axes[2].imshow(strip)
    axes[2].set_title("Eight video frames")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    return figure


def save_sample_overview(sample: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = make_sample_overview(sample)
    try:
        figure.savefig(destination, dpi=120)
    finally:
        plt.close(figure)
