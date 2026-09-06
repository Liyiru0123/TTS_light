from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.geometry_data import (
    GeometryLatentTextDataset,
    collate_geometry_text,
    soft_foreground_probability,
)
from mini_multimodal.ldm.geometry_training import GeometryAuxiliaryDiffusionTask
from mini_multimodal.ldm.grounded_training import grounded_model_from_config
from mini_multimodal.ldm.inference import load_vae_decoder
from mini_multimodal.ldm.revision_training import load_config, schedule_from_config


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_geometry_mask_asset_is_complete_and_source_preserving() -> None:
    manifest = _json(ROOT / "data/ldm/p2_w04_geometry_v3/manifest.json")
    assert manifest["verified"] is True
    assert manifest["record_count"] == 6520
    assert manifest["unique_record_count"] == 6520
    assert manifest["source_records_sha256"] == _sha(ROOT / manifest["source_records"])
    assert manifest["source_manifest_sha256"] == _sha(ROOT / "data/ldm/p2_w04_v2/manifest.json")
    assert manifest["asset_sha256"] == _sha(ROOT / manifest["asset"])


def test_soft_foreground_calibration_reaches_frozen_training_gate() -> None:
    report = _json(ROOT / "results/P2-W04_tiny_image_ldm/revision_v2/attempt3_geometry_calibration.json")
    assert report["verified_for_training"] is True
    assert report["sample_count"] == 420
    assert report["vae_mask_iou"] >= 0.90
    assert report["background_false_activation_rate"] <= 0.01
    assert report["foreground_recall"] >= 0.90


def test_geometry_dataset_collates_full_resolution_masks() -> None:
    config = load_config(ROOT)
    dataset = GeometryLatentTextDataset(
        ROOT,
        data_dir=config["output_data"],
        geometry_data_dir=config["geometry_auxiliary"]["data_dir"],
        split="test",
        max_text_length=config["model"]["max_text_length"],
    )
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_geometry_text)))
    assert batch["latent"].shape == (3, 4, 16, 16)
    assert batch["geometry_mask"].shape == (3, 1, 64, 64)
    assert set(batch["geometry_mask"].unique().tolist()) <= {0.0, 1.0}
    assert (batch["geometry_mask"].flatten(1).sum(1) > 0).all()


def test_soft_foreground_probability_is_shape_preserving_and_differentiable() -> None:
    image = torch.full((1, 3, 8, 8), 0.97, requires_grad=True)
    colored = image.clone()
    colored[:, 0, 2:6, 2:6] = 0.1
    probability = soft_foreground_probability(colored, threshold=0.15, temperature=0.04)
    assert float(probability[0, 0, 3, 3].detach()) > 0.99
    assert float(probability[0, 0, 0, 0].detach()) < 0.03
    probability.sum().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_geometry_loss_backpropagates_through_frozen_vae_to_denoiser() -> None:
    config = load_config(ROOT)
    dataset = GeometryLatentTextDataset(
        ROOT,
        data_dir=config["output_data"],
        geometry_data_dir=config["geometry_auxiliary"]["data_dir"],
        split="test",
        max_text_length=config["model"]["max_text_length"],
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_geometry_text)))
    model = grounded_model_from_config(config, dataset.tokenizer)
    vae = load_vae_decoder(ROOT, config["source_vae_checkpoint"])
    task = GeometryAuxiliaryDiffusionTask(
        schedule_from_config(config),
        vae,
        latent_scaling_factor=dataset.scaling_factor,
        condition_dropout=0.0,
        high_noise_training_fraction=0.0,
        foreground_noise_weight=config["diffusion"]["foreground_noise_weight"],
        null_id=dataset.tokenizer.null_id,
        soft_chroma_threshold=config["geometry_auxiliary"]["soft_chroma_threshold"],
        soft_chroma_temperature=config["geometry_auxiliary"]["soft_chroma_temperature"],
        geometry_max_timestep=49,
        max_geometry_samples=2,
        geometry_foreground_weight=0.2,
        geometry_boundary_weight=0.05,
    )
    output = task.step(model, batch, training=False)
    assert torch.isfinite(output.loss)
    assert output.metrics["geometry_selected_fraction"] == 1.0
    assert output.metrics["geometry_region_loss"] > 0
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(parameter.grad is None for parameter in vae.parameters())
    assert all(not parameter.requires_grad for parameter in vae.parameters())
