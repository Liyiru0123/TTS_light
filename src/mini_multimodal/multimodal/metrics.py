"""Model-independent metric primitives used by later teaching models."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def normalize_exact_match(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return " ".join(text.split())


@dataclass(slots=True)
class ExactMatchMetric:
    correct: int = 0
    total: int = 0

    def update(self, predictions: Iterable[object], targets: Iterable[object]) -> None:
        prediction_list = list(predictions)
        target_list = list(targets)
        if len(prediction_list) != len(target_list):
            raise ValueError("predictions and targets must have equal length")
        for prediction, target in zip(prediction_list, target_list, strict=True):
            self.correct += int(
                normalize_exact_match(prediction) == normalize_exact_match(target)
            )
            self.total += 1

    def compute(self) -> dict[str, float | int]:
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": 0.0 if self.total == 0 else self.correct / self.total,
        }


@dataclass(slots=True)
class MaskedRegressionMetric:
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    element_count: int = 0

    def update(
        self,
        prediction: object,
        target: object,
        mask: object | None = None,
    ) -> None:
        predicted = np.asarray(prediction, dtype=np.float64)
        expected = np.asarray(target, dtype=np.float64)
        if predicted.shape != expected.shape:
            raise ValueError("prediction and target shapes must match")
        difference = predicted - expected
        if mask is not None:
            selected = np.asarray(mask, dtype=bool)
            try:
                selected = np.broadcast_to(selected, difference.shape)
            except ValueError as error:
                raise ValueError("mask cannot be broadcast to the value shape") from error
            difference = difference[selected]
        else:
            difference = difference.reshape(-1)
        self.absolute_error_sum += float(np.abs(difference).sum())
        self.squared_error_sum += float(np.square(difference).sum())
        self.element_count += int(difference.size)

    def compute(self) -> dict[str, float | int]:
        if self.element_count == 0:
            return {"element_count": 0, "mae": 0.0, "mse": 0.0}
        return {
            "element_count": self.element_count,
            "mae": self.absolute_error_sum / self.element_count,
            "mse": self.squared_error_sum / self.element_count,
        }
