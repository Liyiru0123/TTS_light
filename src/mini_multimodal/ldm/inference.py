"""Classifier-free guided deterministic latent sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .data import CharacterTokenizer
from .model import TinyImageDenoiser
from .schedule import DiffusionSchedule


def load_inference_package(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("model_family") != "TinyImageLDM":
        raise ValueError("unsupported TinyImageLDM inference package")
    tokenizer = CharacterTokenizer(payload["vocabulary"])
    model = TinyImageDenoiser(vocab_size=len(tokenizer), **payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    schedule = DiffusionSchedule(**payload["schedule_config"])
    return {"payload": payload, "tokenizer": tokenizer, "model": model, "schedule": schedule}


def load_vae_decoder(project_root: str | Path, checkpoint: str | Path) -> Any:
    from mini_multimodal.training import load_checkpoint
    from mini_multimodal.vae.model import TinyImageVAE

    root = Path(project_root).resolve()
    config = json.loads((root / "configs" / "p2_w02_image_vae.json").read_text(encoding="utf-8"))
    model = TinyImageVAE(**{key: config["model"][key] for key in (
        "base_channels", "latent_channels", "logvar_min", "logvar_max"
    )})
    load_checkpoint(root / checkpoint, model=model, optimizer=None, restore_rng=False)
    model.eval()
    return model


@torch.no_grad()
def generate_images_from_package(
    package: Mapping[str, Any],
    vae: Any,
    prompts: Sequence[str],
    *,
    seed: int,
    guidance_scale: float | None = None,
    sample_steps: int | None = None,
) -> dict[str, Any]:
    payload = package["payload"]
    sampled = sample_latents(
        package["model"],
        package["schedule"],
        package["tokenizer"],
        prompts,
        sample_steps=int(sample_steps or payload["default_sampling"]["sample_steps"]),
        guidance_scale=float(
            payload["default_sampling"]["guidance_scale"]
            if guidance_scale is None
            else guidance_scale
        ),
        seed=seed,
    )
    sampled["images"] = vae.decode(
        sampled["latent"] / float(payload["latent_scaling_factor"])
    ).clamp(0, 1)
    return sampled


def _text_batch(
    prompts: Sequence[str], tokenizer: CharacterTokenizer, max_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [tokenizer.encode(prompt, max_length) for prompt in prompts]
    width = max(max(len(value) for value in encoded), 1)
    ids = torch.zeros((len(prompts), width), dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.bool)
    null_ids = torch.zeros_like(ids)
    null_mask = torch.zeros_like(mask)
    for index, values in enumerate(encoded):
        ids[index, : len(values)] = torch.tensor(values)
        mask[index, : len(values)] = True
        null_ids[index, 0] = tokenizer.null_id
        null_mask[index, 0] = True
    return ids, mask, null_ids, null_mask


@torch.no_grad()
def sample_latents(
    model: TinyImageDenoiser,
    schedule: DiffusionSchedule,
    tokenizer: CharacterTokenizer,
    prompts: Sequence[str],
    *,
    sample_steps: int,
    guidance_scale: float,
    seed: int,
    initial_latent: torch.Tensor | None = None,
) -> dict[str, Any]:
    if not prompts:
        raise ValueError("sampling requires at least one prompt")
    ids, mask, null_ids, null_mask = _text_batch(prompts, tokenizer, model.max_text_length)
    generator = torch.Generator().manual_seed(int(seed))
    latent = (
        torch.randn((len(prompts), model.latent_channels, 16, 16), generator=generator)
        if initial_latent is None
        else initial_latent.clone()
    )
    if latent.shape != (len(prompts), model.latent_channels, 16, 16):
        raise ValueError("initial latent shape mismatch")
    trajectory = [latent.clone()]
    attention_trajectory: list[torch.Tensor] = []
    timesteps = schedule.sampling_timesteps(sample_steps)
    model.eval()
    for index, current in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        timestep = torch.full((len(prompts),), current, dtype=torch.long)
        conditional = model(latent, timestep, ids, mask)
        unconditional = model(latent, timestep, null_ids, null_mask)
        predicted = unconditional.predicted_noise + guidance_scale * (
            conditional.predicted_noise - unconditional.predicted_noise
        )
        latent = schedule.ddim_step(
            latent,
            timestep,
            torch.full_like(timestep, previous),
            predicted,
        )
        trajectory.append(latent.clone())
        attention_trajectory.append(conditional.cross_attention.clone())
    return {
        "latent": latent,
        "latent_trajectory": trajectory,
        "attention_trajectory": attention_trajectory,
        "timesteps": timesteps,
    }
