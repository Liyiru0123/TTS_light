"""TinyT2V DDIM video sampling with classifier-free guidance + motion predictor.

Video latent [B, 4, F, 16, 16]; text -> MotionPredictor -> trajectory ->
per-frame position conditioning -> denoiser. Temporal modules can be toggled.
"""

from __future__ import annotations

import torch

from mini_multimodal.ldm.schedule import DiffusionSchedule

from .model import MotionPredictor, TinyVideoDenoiser


def _pool_text(model: TinyVideoDenoiser, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    text = model.encode_text(ids, mask)
    weights = mask.to(text.dtype).unsqueeze(-1)
    return (text * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def sample_video_latents(
    model: TinyVideoDenoiser,
    predictor: MotionPredictor,
    schedule: DiffusionSchedule,
    tokenizer,
    prompts: list[str],
    *,
    sample_steps: int,
    guidance_scale: float,
    seed: int,
    use_temporal: bool = True,
    disable_cross_attention: bool = False,
    trajectory_override: torch.Tensor | None = None,
) -> dict:
    if not prompts:
        raise ValueError("sampling requires at least one prompt")
    encoded = [tokenizer.encode(prompt, model.max_text_length) for prompt in prompts]
    width = max((len(value) for value in encoded), default=1)
    ids = torch.zeros((len(prompts), width), dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.bool)
    null_ids = torch.zeros_like(ids)
    null_mask = torch.zeros_like(mask)
    for index, values in enumerate(encoded):
        ids[index, : len(values)] = torch.tensor(values)
        mask[index, : len(values)] = True
        null_ids[index, 0] = tokenizer.null_id
        null_mask[index, 0] = True
    if trajectory_override is None:
        trajectory = predictor(_pool_text(model, ids, mask))  # [B, 8, 2]
    else:
        trajectory = trajectory_override.to(dtype=torch.float32)
        if trajectory.shape != (len(prompts), 8, 2):
            raise ValueError("trajectory_override must have shape [B,8,2]")
        if not torch.isfinite(trajectory).all() or not ((0 <= trajectory) & (trajectory <= 1)).all():
            raise ValueError("trajectory_override must be finite and normalized to [0,1]")
    null_trajectory = torch.zeros_like(trajectory)
    generator = torch.Generator().manual_seed(int(seed))
    latent = torch.randn((len(prompts), model.latent_channels, 8, 16, 16), generator=generator)
    trajectory_out = [latent.clone()]
    timesteps = schedule.sampling_timesteps(sample_steps)
    model.eval()
    for index, current in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        timestep = torch.full((len(prompts),), current, dtype=torch.long)
        conditional = model(latent, timestep, ids, mask, use_temporal=use_temporal,
                            disable_cross_attention=disable_cross_attention, trajectory=trajectory)
        unconditional = model(latent, timestep, null_ids, null_mask, use_temporal=use_temporal,
                              disable_cross_attention=disable_cross_attention,
                              trajectory=null_trajectory)
        predicted = unconditional.predicted_noise + guidance_scale * (
            conditional.predicted_noise - unconditional.predicted_noise
        )
        latent = schedule.ddim_step(latent, timestep, torch.full_like(timestep, previous), predicted)
        trajectory_out.append(latent.clone())
    return {"latent": latent, "latent_trajectory": trajectory_out, "timesteps": timesteps,
            "trajectory": trajectory}
