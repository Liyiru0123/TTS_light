from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

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
from mini_multimodal.ldm.revision_evaluation import sample_latents_v2
from mini_multimodal.ldm.revision_training import load_config, schedule_from_config
from mini_multimodal.ldm.schedule import DiffusionSchedule


ROOT = Path(__file__).resolve().parents[1]


def _tokenizer(base_config: dict) -> CharacterTokenizer:
    vocabulary = json.loads(
        (ROOT / base_config["output_data"] / "vocabulary.json").read_text(
            encoding="utf-8"
        )
    )
    return CharacterTokenizer(vocabulary)


def _model() -> tuple[object, dict, dict, CharacterTokenizer]:
    base = load_config(ROOT)
    i005 = load_i005_config(ROOT)
    tokenizer = _tokenizer(base)
    model = refiner_model_from_config(ROOT, base, i005, tokenizer)
    return model, base, i005, tokenizer


def test_model_ref_001_schedule_clean_noise_roundtrip() -> None:
    schedule = DiffusionSchedule(timesteps=50, beta_start=0.0001, beta_end=0.2)
    clean = torch.randn(3, 4, 16, 16)
    noise = torch.randn_like(clean)
    timestep = torch.tensor([0, 17, 49])
    noisy = schedule.q_sample(clean, timestep, noise)

    recovered_clean = schedule.predict_clean(noisy, timestep, noise)
    recovered_noise = schedule.predict_noise(noisy, timestep, clean)

    assert torch.allclose(recovered_clean, clean, atol=2e-5, rtol=2e-5)
    assert torch.allclose(recovered_noise, noise, atol=2e-5, rtol=2e-5)


def test_model_ref_002_factory_loads_i004_and_only_refiner_is_trainable() -> None:
    model, _, i005, _ = _model()
    report = model.parameter_report()

    assert report["denoiser"] == 943018
    assert report["refiner"] == report["trainable"]
    assert 0 < report["refiner"] < 100000
    assert report["total"] < i005["resource_limits"]["max_parameters"]
    assert all(not parameter.requires_grad for parameter in model.denoiser.parameters())
    assert all(parameter.requires_grad for parameter in model.refiner.parameters())
    assert model.denoiser.training is False


def test_model_ref_003_disable_and_high_noise_paths_equal_i004_exactly() -> None:
    model, _, _, tokenizer = _model()
    prompt = "第1帧中有1个可见对象。对象1是绿色的中等大小圆形，位于左侧区域。"
    ids, mask, _, _ = _text_batch([prompt, prompt], tokenizer, model.max_text_length)
    noisy = torch.randn(2, 4, 16, 16, generator=torch.Generator().manual_seed(7))
    timestep = torch.tensor([10, 40])

    base = model.denoiser(noisy, timestep, ids, mask).predicted_noise
    disabled = model(noisy, timestep, ids, mask, enable_refiner=False)
    enabled = model(noisy, timestep, ids, mask, enable_refiner=True)

    assert torch.equal(disabled.predicted_noise, base)
    assert torch.equal(disabled.clean_residual, torch.zeros_like(disabled.clean_residual))
    assert torch.count_nonzero(enabled.clean_residual[0]) > 0
    assert torch.equal(enabled.clean_residual[1], torch.zeros_like(enabled.clean_residual[1]))
    assert torch.equal(enabled.predicted_noise[1], base[1])
    assert torch.isfinite(enabled.predicted_noise).all()


def test_model_ref_004_pure_text_sampler_disabled_matches_i004_sampler() -> None:
    model, _, _, tokenizer = _model()
    prompts = [
        "第1帧中有1个可见对象。对象1是蓝色的小正方形，位于右上区域。",
        "第1帧中有1个可见对象。对象1是红色的大五边形，位于中央区域。",
    ]
    initial = torch.randn(2, 4, 16, 16, generator=torch.Generator().manual_seed(11))
    expected = sample_latents_v2(
        model.denoiser,
        model.schedule,
        tokenizer,
        prompts,
        sample_steps=3,
        guidance_scale=2.5,
        seed=11,
        initial_latent=initial,
    )
    actual = sample_latents_refined(
        model,
        tokenizer,
        prompts,
        sample_steps=3,
        guidance_scale=2.5,
        seed=11,
        enable_refiner=False,
        initial_latent=initial,
    )

    assert torch.equal(actual["latent"], expected["latent"])
    assert actual["refiner_enabled"] is False
    assert set(inspect.signature(sample_latents_refined).parameters).isdisjoint(
        {"mask", "target_mask", "box", "class_id", "target_image"}
    )


def test_model_ref_005_training_step_isolates_epsilon_predictor_gradients() -> None:
    model, base, i005, _ = _model()
    dataset = GeometryLatentTextDataset(
        ROOT,
        data_dir=base["output_data"],
        geometry_data_dir=base["geometry_auxiliary"]["data_dir"],
        split="train",
        max_text_length=base["model"]["max_text_length"],
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
        latent_weight=weights["latent"],
        foreground_weight=weights["foreground"],
        boundary_weight=weights["boundary"],
        residual_regularization_weight=weights["residual_regularization"],
        soft_chroma_threshold=base["geometry_auxiliary"]["soft_chroma_threshold"],
        soft_chroma_temperature=base["geometry_auxiliary"]["soft_chroma_temperature"],
    )
    model.train()
    torch.manual_seed(19)
    output = task.step(model, batch, training=True)
    output.loss.backward()
    denoiser_gradients = [parameter.grad for parameter in model.denoiser.parameters()]
    refiner_gradients = [
        parameter.grad for parameter in model.refiner.parameters() if parameter.grad is not None
    ]

    assert torch.isfinite(output.loss)
    assert output.loss.item() > 0
    assert all(gradient is None for gradient in denoiser_gradients)
    assert refiner_gradients
    assert all(torch.isfinite(gradient).all() for gradient in refiner_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in refiner_gradients)
    assert all(parameter.grad is None for parameter in task.vae.parameters())
    assert output.metrics["geometry_selected_fraction"] == 1.0
    assert output.metrics["foreground_loss"] > 0
    assert output.metrics["boundary_loss"] > 0


def test_model_ref_006_planning_authorization_matches_formal_iteration_state() -> None:
    iteration = ROOT / "results/P2-W04_tiny_image_ldm/iterations/I005"
    authorization = json.loads(
        (
            ROOT
            / "results/P2-W04_tiny_image_ldm/planning/I005/authorization.json"
        ).read_text(encoding="utf-8")
    )
    assert authorization["formal_iteration_created"] is iteration.exists()
    if iteration.exists():
        assert (iteration / "source_snapshot.zip").is_file()
        assert (iteration / "source_manifest.json").is_file()
        assert (iteration / "branches/continuation/branch_manifest.json").is_file()
        assert (iteration / "branches/clean_refiner/branch_manifest.json").is_file()


def test_model_ref_007_machine_structure_validation_is_complete() -> None:
    report = json.loads(
        (
            ROOT
            / "results/P2-W04_tiny_image_ldm/planning/I005/structure_validation.json"
        ).read_text(encoding="utf-8")
    )

    assert report["verified"] is True
    assert report["status"] == "structure_implementation_complete_no_training"
    assert report["architecture"]["parameter_report"] == {
        "denoiser": 943018,
        "refiner": 49860,
        "total": 992878,
        "trainable": 49860,
    }
    assert report["equivalence"]["disabled_max_abs_noise_difference"] == 0.0
    assert report["equivalence"]["inactive_high_noise_max_abs_difference"] == 0.0
    assert report["gradient_isolation"]["denoiser_gradient_tensor_count"] == 0
    assert report["gradient_isolation"]["refiner_nonzero_gradient_tensor_count"] == 33
    assert report["inference_interface"]["text_only"] is True
    assert all(report["checks"].values())
