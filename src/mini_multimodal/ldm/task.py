"""Noise-prediction training adapter for the shared CPU trainer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_multimodal.training.core import StepOutput

from .model import TinyImageDenoiser
from .schedule import DiffusionSchedule


class LatentDiffusionTask:
    def __init__(
        self,
        schedule: DiffusionSchedule,
        *,
        condition_dropout: float,
        high_noise_training_fraction: float = 0.0,
        foreground_noise_weight: float = 0.0,
        null_id: int = 1,
    ) -> None:
        if not 0 <= condition_dropout < 1:
            raise ValueError("condition dropout must be within [0,1)")
        self.schedule = schedule
        self.condition_dropout = float(condition_dropout)
        if not 0 <= high_noise_training_fraction <= 1:
            raise ValueError("high-noise training fraction must be within [0,1]")
        self.high_noise_training_fraction = float(high_noise_training_fraction)
        if foreground_noise_weight < 0:
            raise ValueError("foreground noise weight must be non-negative")
        self.foreground_noise_weight = float(foreground_noise_weight)
        self.null_id = int(null_id)

    def step(self, model: TinyImageDenoiser, batch: dict[str, torch.Tensor], *, training: bool) -> StepOutput:
        clean = batch["latent"]
        batch_size = clean.shape[0]
        timestep = torch.randint(0, self.schedule.timesteps, (batch_size,), device=clean.device)
        high_noise = torch.zeros(batch_size, dtype=torch.bool, device=clean.device)
        if training and self.high_noise_training_fraction:
            high_noise = torch.rand(batch_size, device=clean.device) < self.high_noise_training_fraction
            timestep[high_noise] = torch.randint(
                self.schedule.timesteps // 2,
                self.schedule.timesteps,
                (int(high_noise.sum()),),
                device=clean.device,
            )
        noise = torch.randn_like(clean)
        noisy = self.schedule.q_sample(clean, timestep, noise)
        text_ids = batch["text_ids"].clone()
        text_mask = batch["text_mask"].clone()
        dropped = torch.zeros(batch_size, dtype=torch.bool, device=clean.device)
        if training and self.condition_dropout:
            dropped = torch.rand(batch_size, device=clean.device) < self.condition_dropout
            text_ids[dropped] = 0
            text_mask[dropped] = False
            text_ids[dropped, 0] = self.null_id
            text_mask[dropped, 0] = True
        output = model(noisy, timestep, text_ids, text_mask)
        squared_error = (output.predicted_noise - noise).square()
        spatial_weight = 1.0 + self.foreground_noise_weight * batch["foreground_latent_mask"]
        loss = (squared_error * spatial_weight).sum() / (
            spatial_weight.sum() * squared_error.shape[1]
        )
        unweighted_loss = F.mse_loss(output.predicted_noise, noise)
        baseline = noise.square().mean()
        return StepOutput(
            loss=loss,
            metrics={
                "noise_mse": float(loss.detach()),
                "unweighted_noise_mse": float(unweighted_loss.detach()),
                "zero_prediction_mse": float(baseline.detach()),
                "condition_drop_fraction": float(dropped.float().mean()),
                "high_noise_fraction": float(high_noise.float().mean()),
            },
            weight=batch_size,
        )
