"""P2-W01 TrainTask adapter for TinyImageVAE."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from mini_multimodal.training.core import StepOutput

from .losses import VAELossWeights, compute_vae_loss


class ImageVAETask:
    def __init__(self, weights: VAELossWeights) -> None:
        self.weights = weights

    def step(
        self, model: nn.Module, batch: Mapping[str, Any], *, training: bool
    ) -> StepOutput:
        output = model(batch["image"], sample_posterior=training)
        components = compute_vae_loss(
            output, batch["image"], batch["foreground_mask"], self.weights
        )
        metrics = {
            key: float(value.detach())
            for key, value in components.items()
            if key != "loss"
        }
        return StepOutput(
            loss=components["loss"], metrics=metrics, weight=len(batch["image"])
        )

