"""Export a self-describing TinyImageLDM inference package and checkpoint manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
    result_dir = root / config["output_results"]
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best = result_dir / "training" / "checkpoints" / "best.pt"
    last = result_dir / "training" / "checkpoints" / "last.pt"
    best_state = torch.load(best, map_location="cpu", weights_only=False)
    vocabulary = json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((root / config["output_data"] / "manifest.json").read_text(encoding="utf-8"))
    denoising = json.loads((result_dir / "denoising_metrics.json").read_text(encoding="utf-8"))
    generation = json.loads((result_dir / "generation_metrics.json").read_text(encoding="utf-8"))
    package_path = checkpoint_dir / "tiny_image_ldm_inference.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_family": "TinyImageLDM",
            "model_state": best_state["model_state"],
            "model_config": dict(config["model"]),
            "schedule_config": {
                "timesteps": config["diffusion"]["train_timesteps"],
                "beta_start": config["diffusion"]["beta_start"],
                "beta_end": config["diffusion"]["beta_end"],
            },
            "vocabulary": vocabulary,
            "latent_scaling_factor": data_manifest["latent_scaling_factor"],
            "vae_checkpoint": config["source_vae_checkpoint"],
            "vae_checkpoint_sha256": sha(root / config["source_vae_checkpoint"]),
            "default_sampling": {
                "sample_steps": config["diffusion"]["sample_steps"],
                "guidance_scale": config["diffusion"]["guidance_scale"],
            },
            "best_epoch": best_state["epoch"],
            "validation_monitor": best_state["best_metric"],
            "verified_metrics": {"denoising": denoising, "generation": generation},
        },
        package_path,
    )
    entries = []
    for role, path in (("best", best), ("last", last), ("inference", package_path)):
        state = None if role == "inference" else torch.load(path, map_location="cpu", weights_only=False)
        entries.append({
            "role": role,
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "epoch": best_state["epoch"] if role == "inference" else state["epoch"],
            "contains_optimizer": role != "inference",
            "contains_random_state": role != "inference",
        })
    manifest = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "selection_metric": "validation weighted noise_mse (minimum)",
        "external_dependency": {
            "path": config["source_vae_checkpoint"],
            "sha256": sha(root / config["source_vae_checkpoint"]),
            "role": "frozen TinyImageVAE decoder",
        },
        "entries": entries,
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"inference_package": str(package_path), "entries": len(entries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
