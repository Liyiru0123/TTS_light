"""Training task and factory for the I005 clean-latent refiner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from mini_multimodal.training import load_checkpoint
from mini_multimodal.training.core import StepOutput

from .data import CharacterTokenizer
from .geometry_data import soft_foreground_probability
from .geometry_training import _boundary_loss, _region_loss
from .grounded_training import grounded_model_from_config
from .model_refiner import TinyImageLDMWithCleanRefiner
from .revision_training import schedule_from_config


def load_i005_config(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    return json.loads((root / "configs/p2_w04_i005.json").read_text(encoding="utf-8"))


def refiner_model_from_config(
    project_root: str | Path,
    base_config: Mapping[str, Any],
    i005_config: Mapping[str, Any],
    tokenizer: CharacterTokenizer,
    *,
    load_warmstart: bool = True,
) -> TinyImageLDMWithCleanRefiner:
    root = Path(project_root).resolve()
    denoiser = grounded_model_from_config(base_config, tokenizer)
    if load_warmstart:
        load_checkpoint(
            root / i005_config["frozen_assets"]["warmstart"],
            model=denoiser,
            optimizer=None,
            restore_rng=False,
        )
    refiner = i005_config["clean_refiner"]
    return TinyImageLDMWithCleanRefiner(
        denoiser,
        schedule_from_config(base_config),
        hidden_channels=int(refiner["hidden_channels"]),
        residual_blocks=int(refiner["residual_blocks"]),
        max_residual=float(refiner["max_residual"]),
        max_active_timestep=int(refiner["max_active_timestep"]),
    )


class CleanRefinerTask:
    """Optimize clean-latent/image metrics without updating the epsilon predictor."""

    def __init__(
        self,
        vae: torch.nn.Module,
        *,
        latent_scaling_factor: float,
        null_id: int,
        condition_dropout: float,
        max_geometry_samples: int,
        latent_weight: float,
        foreground_weight: float,
        boundary_weight: float,
        residual_regularization_weight: float,
        soft_chroma_threshold: float,
        soft_chroma_temperature: float,
    ) -> None:
        self.vae = vae.eval()
        self.vae.requires_grad_(False)
        self.latent_scaling_factor = float(latent_scaling_factor)
        self.null_id = int(null_id)
        self.condition_dropout = float(condition_dropout)
        self.max_geometry_samples = int(max_geometry_samples)
        self.latent_weight = float(latent_weight)
        self.foreground_weight = float(foreground_weight)
        self.boundary_weight = float(boundary_weight)
        self.residual_regularization_weight = float(residual_regularization_weight)
        self.soft_chroma_threshold = float(soft_chroma_threshold)
        self.soft_chroma_temperature = float(soft_chroma_temperature)
        if not 0 <= self.condition_dropout < 1 or self.max_geometry_samples < 1:
            raise ValueError("invalid clean-refiner task settings")
        if min(
            self.latent_weight,
            self.foreground_weight,
            self.boundary_weight,
            self.residual_regularization_weight,
        ) < 0:
            raise ValueError("clean-refiner loss weights must be non-negative")

    def step(
        self,
        model: TinyImageLDMWithCleanRefiner,
        batch: dict[str, Any],
        *,
        training: bool,
    ) -> StepOutput:
        if any(parameter.requires_grad for parameter in model.denoiser.parameters()):
            raise RuntimeError("clean-refiner task requires a frozen epsilon predictor")
        clean = batch["latent"]
        batch_size = clean.shape[0]
        timestep = torch.randint(
            0,
            model.max_active_timestep + 1,
            (batch_size,),
            device=clean.device,
        )
        noise = torch.randn_like(clean)
        noisy = model.schedule.q_sample(clean, timestep, noise)
        text_ids = batch["text_ids"].clone()
        text_mask = batch["text_mask"].clone()
        dropped = torch.zeros(batch_size, dtype=torch.bool, device=clean.device)
        if training and self.condition_dropout:
            dropped = torch.rand(batch_size, device=clean.device) < self.condition_dropout
            text_ids[dropped] = 0
            text_mask[dropped] = False
            text_ids[dropped, 0] = self.null_id
            text_mask[dropped, 0] = True
        output = model(
            noisy,
            timestep,
            text_ids,
            text_mask,
            enable_refiner=True,
            detach_denoiser=True,
        )
        latent_loss = F.smooth_l1_loss(output.refined_clean, clean)
        base_latent_loss = F.smooth_l1_loss(output.base_clean, clean)
        residual_regularization = output.clean_residual.square().mean()
        selected = (~dropped).nonzero(as_tuple=False).flatten()[: self.max_geometry_samples]
        foreground_loss = latent_loss.new_zeros(())
        foreground_bce = latent_loss.new_zeros(())
        foreground_dice = latent_loss.new_zeros(())
        boundary_loss = latent_loss.new_zeros(())
        if selected.numel():
            decoded = self.vae.decode(
                output.refined_clean[selected] / self.latent_scaling_factor
            )
            probability = soft_foreground_probability(
                decoded,
                threshold=self.soft_chroma_threshold,
                temperature=self.soft_chroma_temperature,
            )
            target = batch["geometry_mask"][selected]
            foreground_loss, foreground_bce, foreground_dice = _region_loss(
                probability, target
            )
            boundary_loss = _boundary_loss(probability, target)
        loss = (
            self.latent_weight * latent_loss
            + self.foreground_weight * foreground_loss
            + self.boundary_weight * boundary_loss
            + self.residual_regularization_weight * residual_regularization
        )
        return StepOutput(
            loss=loss,
            metrics={
                "base_clean_latent_loss": float(base_latent_loss.detach()),
                "refined_clean_latent_loss": float(latent_loss.detach()),
                "foreground_loss": float(foreground_loss.detach()),
                "foreground_bce": float(foreground_bce.detach()),
                "foreground_dice_loss": float(foreground_dice.detach()),
                "boundary_loss": float(boundary_loss.detach()),
                "residual_regularization": float(residual_regularization.detach()),
                "residual_rms": float(output.clean_residual.square().mean().sqrt().detach()),
                "geometry_selected_fraction": int(selected.numel()) / batch_size,
                "condition_drop_fraction": float(dropped.float().mean()),
            },
            weight=batch_size,
        )
