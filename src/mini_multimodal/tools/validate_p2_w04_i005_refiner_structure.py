from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from mini_multimodal.iteration_management import atomic_write_json, sha256_file
from mini_multimodal.ldm.data import CharacterTokenizer
from mini_multimodal.ldm.geometry_data import (
    GeometryLatentTextDataset,
    collate_geometry_text,
)
from mini_multimodal.ldm.inference import _text_batch, load_vae_decoder
from mini_multimodal.ldm.refiner_inference import sample_latents_refined
from mini_multimodal.ldm.refiner_training import (
    CleanRefinerTask,
    load_i005_config,
    refiner_model_from_config,
)
from mini_multimodal.ldm.revision_training import _state_sha, load_config


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/P2-W04_tiny_image_ldm/planning/I005/structure_validation.json"


def main() -> None:
    base = load_config(ROOT)
    i005 = load_i005_config(ROOT)
    vocabulary_path = ROOT / base["output_data"] / "vocabulary.json"
    tokenizer = CharacterTokenizer(json.loads(vocabulary_path.read_text(encoding="utf-8")))
    model = refiner_model_from_config(ROOT, base, i005, tokenizer)
    denoiser_state_before = _state_sha(model.denoiser)
    prompt = "第1帧中有1个可见对象。对象1是绿色的中等大小圆形，位于左侧区域。"
    ids, mask, _, _ = _text_batch([prompt, prompt], tokenizer, model.max_text_length)
    noisy = torch.randn(2, 4, 16, 16, generator=torch.Generator().manual_seed(29))
    timestep = torch.tensor([10, 40])
    with torch.no_grad():
        base_noise = model.denoiser(noisy, timestep, ids, mask).predicted_noise
        disabled = model(noisy, timestep, ids, mask, enable_refiner=False)
        enabled = model(noisy, timestep, ids, mask, enable_refiner=True)

    dataset = GeometryLatentTextDataset(
        ROOT,
        data_dir=base["output_data"],
        geometry_data_dir=base["geometry_auxiliary"]["data_dir"],
        split="train",
        max_text_length=int(base["model"]["max_text_length"]),
    )
    batch = collate_geometry_text([dataset[0], dataset[1]])
    settings = i005["clean_refiner"]
    weights = settings["loss_weights"]
    task = CleanRefinerTask(
        load_vae_decoder(ROOT, base["source_vae_checkpoint"]),
        latent_scaling_factor=dataset.scaling_factor,
        null_id=dataset.tokenizer.null_id,
        condition_dropout=0.0,
        max_geometry_samples=2,
        latent_weight=float(weights["latent"]),
        foreground_weight=float(weights["foreground"]),
        boundary_weight=float(weights["boundary"]),
        residual_regularization_weight=float(weights["residual_regularization"]),
        soft_chroma_threshold=float(base["geometry_auxiliary"]["soft_chroma_threshold"]),
        soft_chroma_temperature=float(base["geometry_auxiliary"]["soft_chroma_temperature"]),
    )
    model.train()
    torch.manual_seed(31)
    step = task.step(model, batch, training=True)
    step.loss.backward()
    denoiser_gradients = [
        parameter.grad for parameter in model.denoiser.parameters() if parameter.grad is not None
    ]
    refiner_gradients = [
        parameter.grad for parameter in model.refiner.parameters() if parameter.grad is not None
    ]
    signature = list(inspect.signature(sample_latents_refined).parameters)
    forbidden = {"mask", "target_mask", "box", "class_id", "target_image"}
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "planning_target": "I005",
        "status": "structure_implementation_complete_no_training",
        "sources": {
            "config": {
                "path": "configs/p2_w04_i005.json",
                "sha256": sha256_file(ROOT / "configs/p2_w04_i005.json"),
            },
            "model": {
                "path": "src/mini_multimodal/ldm/model_refiner.py",
                "sha256": sha256_file(ROOT / "src/mini_multimodal/ldm/model_refiner.py"),
            },
            "training_task": {
                "path": "src/mini_multimodal/ldm/refiner_training.py",
                "sha256": sha256_file(ROOT / "src/mini_multimodal/ldm/refiner_training.py"),
            },
            "inference": {
                "path": "src/mini_multimodal/ldm/refiner_inference.py",
                "sha256": sha256_file(ROOT / "src/mini_multimodal/ldm/refiner_inference.py"),
            },
            "warmstart": {
                "path": i005["frozen_assets"]["warmstart"],
                "sha256": sha256_file(ROOT / i005["frozen_assets"]["warmstart"]),
            },
        },
        "architecture": {
            "parameter_report": model.parameter_report(),
            "hidden_channels": int(settings["hidden_channels"]),
            "residual_blocks": int(settings["residual_blocks"]),
            "max_residual": float(settings["max_residual"]),
            "max_active_timestep": int(settings["max_active_timestep"]),
            "denoiser_frozen": all(
                not parameter.requires_grad for parameter in model.denoiser.parameters()
            ),
            "refiner_trainable": all(
                parameter.requires_grad for parameter in model.refiner.parameters()
            ),
        },
        "equivalence": {
            "disabled_max_abs_noise_difference": float(
                (disabled.predicted_noise - base_noise).abs().max()
            ),
            "inactive_high_noise_max_abs_difference": float(
                (enabled.predicted_noise[1] - base_noise[1]).abs().max()
            ),
            "active_low_noise_residual_rms": float(
                enabled.clean_residual[0].square().mean().sqrt()
            ),
        },
        "gradient_isolation": {
            "loss": float(step.loss.detach()),
            "metrics": dict(step.metrics),
            "denoiser_gradient_tensor_count": len(denoiser_gradients),
            "refiner_gradient_tensor_count": len(refiner_gradients),
            "refiner_nonzero_gradient_tensor_count": sum(
                int(torch.count_nonzero(gradient) > 0) for gradient in refiner_gradients
            ),
            "all_refiner_gradients_finite": all(
                torch.isfinite(gradient).all() for gradient in refiner_gradients
            ),
            "denoiser_state_sha256_before": denoiser_state_before,
            "denoiser_state_sha256_after": _state_sha(model.denoiser),
        },
        "inference_interface": {
            "parameters": signature,
            "forbidden_target_parameters_present": sorted(forbidden & set(signature)),
            "text_only": not bool(forbidden & set(signature)),
        },
        "governance": {
            "formal_iteration_created": (
                ROOT / "results/P2-W04_tiny_image_ldm/iterations/I005"
            ).exists(),
            "training_started": False,
            "checkpoint_created": False,
        },
    }
    limit = int(i005["resource_limits"]["max_parameters"])
    report["checks"] = {
        "parameter_limit": report["architecture"]["parameter_report"]["total"] < limit,
        "only_refiner_trainable": report["architecture"]["denoiser_frozen"]
        and report["architecture"]["refiner_trainable"],
        "disabled_exact_equivalence": report["equivalence"][
            "disabled_max_abs_noise_difference"
        ]
        == 0.0,
        "inactive_exact_equivalence": report["equivalence"][
            "inactive_high_noise_max_abs_difference"
        ]
        == 0.0,
        "active_refiner_nonzero": report["equivalence"]["active_low_noise_residual_rms"]
        > 0.0,
        "epsilon_predictor_has_no_gradient": not denoiser_gradients,
        "refiner_has_finite_nonzero_gradient": bool(refiner_gradients)
        and report["gradient_isolation"]["all_refiner_gradients_finite"]
        and report["gradient_isolation"]["refiner_nonzero_gradient_tensor_count"] > 0,
        "epsilon_predictor_state_unchanged": report["gradient_isolation"][
            "denoiser_state_sha256_before"
        ]
        == report["gradient_isolation"]["denoiser_state_sha256_after"],
        "pure_text_inference": report["inference_interface"]["text_only"],
        "no_iteration_or_training_output": not report["governance"][
            "formal_iteration_created"
        ]
        and not report["governance"]["training_started"]
        and not report["governance"]["checkpoint_created"],
    }
    report["verified"] = all(report["checks"].values())
    atomic_write_json(OUTPUT, report)
    if not report["verified"]:
        raise RuntimeError("I005 clean-refiner structure validation failed")
    print(
        json.dumps(
            {
                "output": OUTPUT.relative_to(ROOT).as_posix(),
                "parameters": report["architecture"]["parameter_report"],
                "equivalence": report["equivalence"],
                "gradient_isolation": {
                    "denoiser_gradient_tensor_count": report["gradient_isolation"][
                        "denoiser_gradient_tensor_count"
                    ],
                    "refiner_gradient_tensor_count": report["gradient_isolation"][
                        "refiner_gradient_tensor_count"
                    ],
                    "refiner_nonzero_gradient_tensor_count": report[
                        "gradient_isolation"
                    ]["refiner_nonzero_gradient_tensor_count"],
                },
                "verified": report["verified"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
