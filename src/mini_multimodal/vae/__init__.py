"""Tiny KL-regularized image autoencoder for the latent-diffusion spine."""

from .data import (
    VAEImageDataset,
    build_p2_w02_data_spec,
    collate_vae_images,
    export_p2_w02_data,
    verify_p2_w02_data,
)
from .losses import VAELossWeights, compute_vae_loss
from .model import TinyImageVAE, VAEOutput
from .task import ImageVAETask

__all__ = [
    "ImageVAETask",
    "TinyImageVAE",
    "VAEImageDataset",
    "VAELossWeights",
    "VAEOutput",
    "build_p2_w02_data_spec",
    "collate_vae_images",
    "compute_vae_loss",
    "export_p2_w02_data",
    "verify_p2_w02_data",
]

