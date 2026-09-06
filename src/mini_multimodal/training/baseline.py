"""Single-object shape dataset and diagnostic convolutional baseline."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from .core import StepOutput


class ManifestImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, project_root: str | Path, entries: Sequence[Mapping[str, Any]]) -> None:
        self.project_root = Path(project_root).resolve()
        self.entries = [dict(entry) for entry in entries]
        if not self.entries:
            raise ValueError("image dataset requires entries")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        path = (self.project_root / entry["image_path"]).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as error:
            raise ValueError("image path escapes project root") from error
        with Image.open(path) as source:
            image = source.convert("RGB")
            x, y, width, height = (int(value) for value in entry["object_bbox_xywh"])
            padding = max(2, round(max(width, height) * 0.2))
            left = max(0, x - padding)
            top = max(0, y - padding)
            right = min(image.width, x + width + padding)
            bottom = min(image.height, y + height + padding)
            image = image.crop((left, top, right, bottom)).resize(
                (64, 64), Image.Resampling.BILINEAR
            )
            array = np.array(image, dtype=np.uint8, copy=True)
        image = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        return {
            "image": image,
            "target": torch.tensor(entry["label_index"], dtype=torch.long),
            "record_id": entry["record_id"],
            "variant": entry["variant"],
        }


class TinyConvClassifier(nn.Module):
    def __init__(self, class_count: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, class_count),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1:] != (3, 64, 64):
            raise ValueError("expected image tensor [B,3,64,64]")
        return self.classifier(self.features(image))


class ClassificationTask:
    def step(
        self, model: nn.Module, batch: Mapping[str, Any], *, training: bool
    ) -> StepOutput:
        logits = model(batch["image"])
        target = batch["target"]
        loss = F.cross_entropy(logits, target)
        accuracy = float((logits.argmax(dim=1) == target).float().mean())
        return StepOutput(loss=loss, metrics={"accuracy": accuracy}, weight=len(target))


def learn_cue_mapping(entries: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        if entry.get("cue_token") is not None:
            counts[str(entry["cue_token"])][str(entry["label"])] += 1
    return {
        cue: labels.most_common(1)[0][0] for cue, labels in sorted(counts.items())
    }


def cue_only_accuracy(
    entries: Iterable[Mapping[str, Any]], mapping: Mapping[str, str]
) -> dict[str, float | int]:
    usable = [entry for entry in entries if entry.get("cue_token") in mapping]
    correct = sum(mapping[str(entry["cue_token"])] == entry["label"] for entry in usable)
    return {
        "correct": correct,
        "total": len(usable),
        "accuracy": 0.0 if not usable else correct / len(usable),
    }
