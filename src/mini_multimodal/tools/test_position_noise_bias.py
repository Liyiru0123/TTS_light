"""Test position-biased initial noise for spatial control during sampling.

Instead of modifying the model, modify the INITIAL NOISE during sampling:
1. Parse position region from text
2. Create a spatial Gaussian centered at the target region
3. Add the Gaussian to the initial noise
4. Run normal DDIM sampling

This doesn't require retraining — test with the existing checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mini_multimodal.ldm.artifacts import _load_shape_model, _load_vae, probe_image
from mini_multimodal.ldm.data import CharacterTokenizer
from mini_multimodal.ldm.inference import load_vae_decoder, sample_latents
from mini_multimodal.ldm.model import TinyImageDenoiser, build_position_token_ids
from mini_multimodal.ldm.schedule import DiffusionSchedule
from mini_multimodal.tasks.reasoning import position_region
from mini_multimodal.training import load_checkpoint
from mini_multimodal.world.schema import Position


# Region centers in normalized [0,1] coordinates (x=right, y=down)
REGION_CENTERS = {
    "upper_left": (1/6, 1/6), "upper_center": (3/6, 1/6), "upper_right": (5/6, 1/6),
    "center_left": (1/6, 3/6), "center": (3/6, 3/6), "center_right": (5/6, 3/6),
    "lower_left": (1/6, 5/6), "lower_center": (3/6, 5/6), "lower_right": (5/6, 5/6),
}


def make_position_noise_bias(region: str, channels: int = 4, size: int = 16, sigma: float = 0.18) -> torch.Tensor:
    """Create a spatial Gaussian centered at the target region."""
    cx, cy = REGION_CENTERS[region]
    coords = torch.linspace(0, 1, size)
    gy = coords.view(-1, 1).expand(size, size)
    gx = coords.view(1, -1).expand(size, size)
    gaussian = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
    return gaussian.unsqueeze(0).expand(channels, size, size)


def sample_with_position_bias(
    model, schedule, tokenizer, prompts, *, sample_steps, guidance_scale, seed,
    bias_strength: float = 0.0,
):
    """Sample with optional position-biased initial noise."""
    from mini_multimodal.ldm.inference import _text_batch
    ids, mask, null_ids, null_mask = _text_batch(prompts, tokenizer, model.max_text_length)
    generator = torch.Generator().manual_seed(int(seed))
    latent = torch.randn((len(prompts), model.latent_channels, 16, 16), generator=generator)

    # Apply position bias to initial noise
    if bias_strength > 0:
        for i, prompt in enumerate(prompts):
            # Extract position region from text by tokenizing and checking
            encoded = tokenizer.encode(prompt, model.max_text_length)
            region = None
            for token_id in encoded:
                token = tokenizer.tokens[token_id] if token_id < len(tokenizer.tokens) else ""
                for zh_name, region_name in [
                    ("左上区域", "upper_left"), ("上方区域", "upper_center"), ("右上区域", "upper_right"),
                    ("左侧区域", "center_left"), ("中央区域", "center"), ("右侧区域", "center_right"),
                    ("左下区域", "lower_left"), ("下方区域", "lower_center"), ("右下区域", "lower_right"),
                ]:
                    if token == zh_name:
                        region = region_name
                        break
                if region:
                    break
            if region:
                bias = make_position_noise_bias(region, model.latent_channels)
                latent[i] = latent[i] + bias_strength * bias

    trajectory = [latent.clone()]
    attention_trajectory = []
    timesteps = schedule.sampling_timesteps(sample_steps)
    model.eval()
    with torch.no_grad():
        for index, current in enumerate(timesteps):
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            timestep = torch.full((len(prompts),), current, dtype=torch.long)
            conditional = model(latent, timestep, ids, mask)
            unconditional = model(latent, timestep, null_ids, null_mask)
            predicted = unconditional.predicted_noise + guidance_scale * (
                conditional.predicted_noise - unconditional.predicted_noise
            )
            latent = schedule.ddim_step(latent, timestep, torch.full_like(timestep, previous), predicted)
            trajectory.append(latent.clone())
            attention_trajectory.append(conditional.cross_attention.clone())
    return {"latent": latent, "latent_trajectory": trajectory, "attention_trajectory": attention_trajectory, "timesteps": timesteps}


def main():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
    vocab = json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocab)
    pos_ids = build_position_token_ids(vocab)
    model = TinyImageDenoiser(vocab_size=len(tokenizer), position_token_ids=pos_ids, **{k: config["model"][k] for k in ("latent_channels", "base_channels", "text_dim", "time_dim", "attention_heads", "max_text_length", "dual_cross_attention")})
    load_checkpoint(root / "results/P2-W04_tiny_image_ldm/training/checkpoints/best.pt", model=model, optimizer=None, restore_rng=False)
    model.eval()
    schedule = DiffusionSchedule(timesteps=config["diffusion"]["train_timesteps"], beta_start=config["diffusion"]["beta_start"], beta_end=config["diffusion"]["beta_end"])
    vae = load_vae_decoder(root, config["source_vae_checkpoint"])
    shape_model = _load_shape_model(root, config)
    scaling = float(json.loads((root / config["output_data"] / "manifest.json").read_text(encoding="utf-8"))["latent_scaling_factor"])

    prompts_regions = [
        ("第1帧中有1个可见对象。对象1是红色的大圆形，位于左侧区域。", "center_left"),
        ("第1帧中有1个可见对象。对象1是红色的大圆形，位于右侧区域。", "center_right"),
        ("第1帧中有1个可见对象。对象1是红色的大圆形，位于上方区域。", "upper_center"),
        ("第1帧中有1个可见对象。对象1是红色的大圆形，位于下方区域。", "lower_center"),
    ]

    for strength in [0.0, 1.0, 2.0, 3.0, 5.0]:
        print(f"=== bias_strength={strength} ===")
        correct = 0
        for prompt, target_region in prompts_regions:
            sampled = sample_with_position_bias(model, schedule, tokenizer, [prompt], sample_steps=40, guidance_scale=2.5, seed=42, bias_strength=strength)
            image = vae.decode(sampled["latent"] / scaling).clamp(0, 1)
            probe = probe_image(image[0], shape_model)
            if probe["objects"]:
                obj = probe["objects"][0]
                pred_region = position_region(Position(obj["center_xy"][0], obj["center_xy"][1])).value
                match = "OK" if pred_region == target_region else "MISS"
                if pred_region == target_region:
                    correct += 1
                print(f"  [{match}] target={target_region:15s} pred={pred_region:15s} center=({obj['center_xy'][0]:.2f},{obj['center_xy'][1]:.2f}) color={obj['color']} shape={obj['shape']}")
            else:
                print(f"  [MISS] target={target_region:15s} NO OBJECT")
        print(f"  Position correct: {correct}/4\n")


if __name__ == "__main__":
    main()