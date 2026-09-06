"""A lightweight clean-latent refinement path for P2-W04 I005.

The frozen denoiser keeps the standard epsilon-prediction teaching path.  The
new module acts on an x0 estimate and converts the refined estimate back to
epsilon for the unchanged DDIM update.  Target masks are never model inputs.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .model import SinusoidalTimeEmbedding
from .model_grounded import TinyImageDenoiserGrounded
from .model_v2 import coordinate_grid
from .schedule import DiffusionSchedule


class RefinerResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value + condition)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return value + hidden


class CleanLatentRefiner(nn.Module):
    """Predict a bounded residual in the 4x16x16 clean latent space."""

    def __init__(
        self,
        *,
        latent_channels: int,
        text_dim: int,
        time_dim: int,
        grounding_channels: int,
        hidden_channels: int = 32,
        residual_blocks: int = 2,
        max_residual: float = 0.25,
    ) -> None:
        super().__init__()
        if hidden_channels % 8 or residual_blocks < 1 or max_residual <= 0:
            raise ValueError("invalid clean-refiner dimensions")
        self.latent_channels = int(latent_channels)
        self.max_residual = float(max_residual)
        self.clean_projection = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)
        self.coordinate_projection = nn.Conv2d(2, hidden_channels, 1, bias=False)
        self.spatial_text_projection = nn.Conv2d(1, hidden_channels, 3, padding=1, bias=False)
        self.grounding_projection = nn.Conv2d(grounding_channels, hidden_channels, 1, bias=False)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.blocks = nn.ModuleList(
            RefinerResidualBlock(hidden_channels) for _ in range(residual_blocks)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, 3, padding=1),
        )
        nn.init.normal_(self.output[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        clean_estimate: torch.Tensor,
        timestep: torch.Tensor,
        pooled_text: torch.Tensor,
        spatial_text: torch.Tensor,
        object_grounding: torch.Tensor,
    ) -> torch.Tensor:
        if clean_estimate.ndim != 4 or clean_estimate.shape[1:] != (
            self.latent_channels,
            16,
            16,
        ):
            raise ValueError("clean refiner expects [B,C,16,16]")
        batch = clean_estimate.shape[0]
        if timestep.shape != (batch,) or pooled_text.shape[0] != batch:
            raise ValueError("clean-refiner condition shape mismatch")
        if spatial_text.shape != (batch, 1, 16, 16):
            raise ValueError("spatial text must have shape [B,1,16,16]")
        grid = coordinate_grid(
            16, 16, device=clean_estimate.device, dtype=clean_estimate.dtype
        )
        value = self.clean_projection(clean_estimate)
        value = value + self.coordinate_projection(grid)
        value = value + self.spatial_text_projection(spatial_text.to(value.dtype))
        value = value + self.grounding_projection(object_grounding.to(value.dtype))
        condition = self.time_embedding(timestep) + self.text_projection(pooled_text)
        condition = condition[:, :, None, None].to(value.dtype)
        for block in self.blocks:
            value = block(value, condition)
        return self.max_residual * torch.tanh(self.output(value))


@dataclass(slots=True)
class RefinedDenoiserOutput:
    predicted_noise: torch.Tensor
    base_predicted_noise: torch.Tensor
    base_clean: torch.Tensor
    refined_clean: torch.Tensor
    clean_residual: torch.Tensor
    active_mask: torch.Tensor
    cross_attention_16: torch.Tensor
    cross_attention_8: torch.Tensor


class TinyImageLDMWithCleanRefiner(nn.Module):
    """Frozen grounded epsilon predictor plus an independently trainable x0 refiner."""

    model_family = "TinyImageLDM-clean-refiner-I005"

    def __init__(
        self,
        denoiser: TinyImageDenoiserGrounded,
        schedule: DiffusionSchedule,
        *,
        hidden_channels: int = 32,
        residual_blocks: int = 2,
        max_residual: float = 0.25,
        max_active_timestep: int = 35,
    ) -> None:
        super().__init__()
        if not 0 <= max_active_timestep < schedule.timesteps:
            raise ValueError("max active timestep lies outside the diffusion schedule")
        self.denoiser = denoiser
        self.schedule = schedule
        self.max_active_timestep = int(max_active_timestep)
        self.refiner = CleanLatentRefiner(
            latent_channels=denoiser.latent_channels,
            text_dim=denoiser.text_embedding.embedding_dim,
            time_dim=denoiser.time_embedding[0].dimension,
            grounding_channels=denoiser.input.out_channels,
            hidden_channels=hidden_channels,
            residual_blocks=residual_blocks,
            max_residual=max_residual,
        )
        self.freeze_denoiser()

    @property
    def latent_channels(self) -> int:
        return self.denoiser.latent_channels

    @property
    def max_text_length(self) -> int:
        return self.denoiser.max_text_length

    def freeze_denoiser(self) -> None:
        self.denoiser.requires_grad_(False)
        self.denoiser.eval()

    def train(self, mode: bool = True) -> TinyImageLDMWithCleanRefiner:
        super().train(mode)
        self.denoiser.eval()
        return self

    def _text_conditions(
        self, text_ids: torch.Tensor, text_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.denoiser.encode_text(text_ids, text_mask)
        weights = text_mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        spatial = self.denoiser.text_spatial_map(text_ids, 16, 16).to(encoded.dtype)
        grounding = self.denoiser.object_grounding_feature(
            text_ids, encoded, 16, 16, scale="16x16"
        )
        return pooled, spatial, grounding

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        enable_refiner: bool = True,
        detach_denoiser: bool = True,
        denoiser_options: dict[str, bool] | None = None,
    ) -> RefinedDenoiserOutput:
        context = torch.no_grad() if detach_denoiser else nullcontext()
        with context:
            base = self.denoiser(
                noisy_latent,
                timestep,
                text_ids,
                text_mask,
                **dict(denoiser_options or {}),
            )
            base_clean = self.schedule.predict_clean(
                noisy_latent, timestep, base.predicted_noise
            )
            pooled, spatial, grounding = self._text_conditions(text_ids, text_mask)
        active = (timestep <= self.max_active_timestep).to(base_clean.dtype)
        active = active.reshape(active.shape[0], 1, 1, 1)
        if enable_refiner:
            residual = self.refiner(
                base_clean.detach() if detach_denoiser else base_clean,
                timestep,
                pooled.detach() if detach_denoiser else pooled,
                spatial.detach() if detach_denoiser else spatial,
                grounding.detach() if detach_denoiser else grounding,
            ) * active
        else:
            residual = torch.zeros_like(base_clean)
        refined_clean = base_clean + residual
        if enable_refiner:
            refined_noise = self.schedule.predict_noise(
                noisy_latent, timestep, refined_clean
            )
            predicted_noise = torch.where(
                active.bool(), refined_noise, base.predicted_noise
            )
        else:
            predicted_noise = base.predicted_noise
        return RefinedDenoiserOutput(
            predicted_noise=predicted_noise,
            base_predicted_noise=base.predicted_noise,
            base_clean=base_clean,
            refined_clean=refined_clean,
            clean_residual=residual,
            active_mask=active,
            cross_attention_16=base.cross_attention_16,
            cross_attention_8=base.cross_attention_8,
        )

    def parameter_report(self) -> dict[str, int]:
        return {
            "denoiser": sum(parameter.numel() for parameter in self.denoiser.parameters()),
            "refiner": sum(parameter.numel() for parameter in self.refiner.parameters()),
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }
