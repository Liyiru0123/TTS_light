"""Pure-text sampling for the I005 clean-latent refiner."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .data import CharacterTokenizer
from .inference import _text_batch
from .model_refiner import TinyImageLDMWithCleanRefiner


@torch.no_grad()
def sample_latents_refined(
    model: TinyImageLDMWithCleanRefiner,
    tokenizer: CharacterTokenizer,
    prompts: Sequence[str],
    *,
    sample_steps: int,
    guidance_scale: float,
    seed: int,
    enable_refiner: bool = True,
    initial_latent: torch.Tensor | None = None,
    keep_trajectory: bool = False,
) -> dict[str, Any]:
    """Generate latents from text only; no mask, box, class, or target image is accepted."""
    ids, mask, null_ids, null_mask = _text_batch(
        prompts, tokenizer, model.max_text_length
    )
    generator = torch.Generator().manual_seed(int(seed))
    latent = (
        torch.randn(
            (len(prompts), model.latent_channels, 16, 16), generator=generator
        )
        if initial_latent is None
        else initial_latent.clone()
    )
    if latent.shape != (len(prompts), model.latent_channels, 16, 16):
        raise ValueError("initial latent shape mismatch")
    trajectory = [latent.clone()] if keep_trajectory else []
    residual_norms: list[float] = []
    timesteps = model.schedule.sampling_timesteps(sample_steps)
    model.eval()
    for index, current in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        step = torch.full((len(prompts),), current, dtype=torch.long)
        conditional = model(
            latent,
            step,
            ids,
            mask,
            enable_refiner=enable_refiner,
        )
        unconditional = model(
            latent,
            step,
            null_ids,
            null_mask,
            enable_refiner=enable_refiner,
        )
        predicted = unconditional.predicted_noise + guidance_scale * (
            conditional.predicted_noise - unconditional.predicted_noise
        )
        latent = model.schedule.ddim_step(
            latent, step, torch.full_like(step, previous), predicted
        )
        residual_norms.append(float(conditional.clean_residual.square().mean().sqrt()))
        if keep_trajectory:
            trajectory.append(latent.clone())
    return {
        "latent": latent,
        "latent_trajectory": trajectory,
        "timesteps": timesteps,
        "conditional_residual_rms": residual_norms,
        "refiner_enabled": bool(enable_refiner),
    }
