"""Reconstruction, foreground-analysis, and KL terms for TinyImageVAE."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import VAEOutput


@dataclass(frozen=True, slots=True)
class VAELossWeights:
    global_weight: float = 0.0
    global_mse_weight: float = 1.0
    foreground_weight: float = 0.0
    background_weight: float = 0.0
    edge_weight: float = 0.0
    kl_weight: float = 1e-4

    def __post_init__(self) -> None:
        values = (
            self.global_weight,
            self.global_mse_weight,
            self.foreground_weight,
            self.background_weight,
            self.edge_weight,
            self.kl_weight,
        )
        if any(value < 0 for value in values) or sum(values[:5]) <= 0:
            raise ValueError("VAE loss weights must be non-negative with reconstruction weight")


def _masked_mae(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(error.dtype).expand_as(error)
    return (error * expanded).sum() / expanded.sum().clamp_min(1.0)


def compute_vae_loss(
    output: VAEOutput,
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
    weights: VAELossWeights,
) -> dict[str, torch.Tensor]:
    if target.shape != output.reconstruction.shape:
        raise ValueError("target and reconstruction shapes differ")
    if foreground_mask.shape != (target.shape[0], 1, target.shape[2], target.shape[3]):
        raise ValueError("foreground mask must be [B,1,H,W]")
    error = (output.reconstruction - target).abs()
    global_mae = error.mean()
    global_mse = (output.reconstruction - target).square().mean()
    foreground_mae = _masked_mae(error, foreground_mask)
    background_mae = _masked_mae(error, ~foreground_mask.bool())
    boundary_source = foreground_mask.float()
    dilated = torch.nn.functional.max_pool2d(boundary_source, 3, stride=1, padding=1)
    eroded = -torch.nn.functional.max_pool2d(-boundary_source, 3, stride=1, padding=1)
    boundary = (dilated - eroded) > 0
    horizontal_error = (
        (output.reconstruction[:, :, :, 1:] - output.reconstruction[:, :, :, :-1])
        - (target[:, :, :, 1:] - target[:, :, :, :-1])
    ).abs()
    vertical_error = (
        (output.reconstruction[:, :, 1:, :] - output.reconstruction[:, :, :-1, :])
        - (target[:, :, 1:, :] - target[:, :, :-1, :])
    ).abs()
    horizontal_mask = boundary[:, :, :, 1:] | boundary[:, :, :, :-1]
    vertical_mask = boundary[:, :, 1:, :] | boundary[:, :, :-1, :]
    edge_mae = 0.5 * (
        _masked_mae(horizontal_error, horizontal_mask)
        + _masked_mae(vertical_error, vertical_mask)
    )
    kl = -0.5 * (1.0 + output.logvar - output.mean.square() - output.logvar.exp()).mean()
    reconstruction_loss = (
        weights.global_weight * global_mae
        + weights.global_mse_weight * global_mse
        + weights.foreground_weight * foreground_mae
        + weights.background_weight * background_mae
        + weights.edge_weight * edge_mae
    )
    total = reconstruction_loss + weights.kl_weight * kl
    return {
        "loss": total,
        "reconstruction_loss": reconstruction_loss,
        "global_mae": global_mae,
        "global_mse": global_mse,
        "foreground_mae": foreground_mae,
        "background_mae": background_mae,
        "edge_mae": edge_mae,
        "kl": kl,
    }
