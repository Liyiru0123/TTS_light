"""Quick diagnostic: test CFG values and position sensitivity without retraining."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mini_multimodal.ldm.artifacts import _load_shape_model, probe_image
from mini_multimodal.ldm.data import CharacterTokenizer
from mini_multimodal.ldm.inference import load_vae_decoder, sample_latents
from mini_multimodal.ldm.model import TinyImageDenoiser
from mini_multimodal.ldm.schedule import DiffusionSchedule
from mini_multimodal.tasks.reasoning import position_region
from mini_multimodal.training import load_checkpoint
from mini_multimodal.world.schema import Position


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
    vocab = json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocab)
    model = TinyImageDenoiser(vocab_size=len(tokenizer), **config["model"])
    load_checkpoint(root / "results/P2-W04_tiny_image_ldm/training/checkpoints/best.pt", model=model, optimizer=None, restore_rng=False)
    model.eval()
    schedule = DiffusionSchedule(
        timesteps=config["diffusion"]["train_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
    )
    vae = load_vae_decoder(root, config["source_vae_checkpoint"])
    shape_model = _load_shape_model(root, config)
    scaling = float(json.loads((root / config["output_data"] / "manifest.json").read_text(encoding="utf-8"))["latent_scaling_factor"])

    prompts = [
        ("center_left", "第1帧中有1个可见对象。对象1是红色的大圆形，位于左侧区域。"),
        ("center_right", "第1帧中有1个可见对象。对象1是红色的大圆形，位于右侧区域。"),
        ("upper_center", "第1帧中有1个可见对象。对象1是红色的大圆形，位于上方区域。"),
        ("lower_center", "第1帧中有1个可见对象。对象1是红色的大圆形，位于下方区域。"),
    ]

    for cfg in [2.5, 4.0, 6.0, 8.0]:
        print(f"=== CFG={cfg} ===")
        correct = 0
        for target_region, prompt in prompts:
            sampled = sample_latents(model, schedule, tokenizer, [prompt], sample_steps=40, guidance_scale=cfg, seed=42)
            image = vae.decode(sampled["latent"] / scaling).clamp(0, 1)
            probe = probe_image(image[0], shape_model)
            if probe["objects"]:
                obj = probe["objects"][0]
                pred_region = position_region(Position(obj["center_xy"][0], obj["center_xy"][1])).value
                match = "OK" if pred_region == target_region else "MISS"
                if pred_region == target_region:
                    correct += 1
                print(f"  [{match}] target={target_region:15s} pred={pred_region:15s} center=({obj['center_xy'][0]:.2f},{obj['center_xy'][1]:.2f})")
            else:
                print(f"  [MISS] target={target_region:15s} NO OBJECT")
        print(f"  Position correct: {correct}/4")
        print()


if __name__ == "__main__":
    main()