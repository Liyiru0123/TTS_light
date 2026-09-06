from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.speech_vae import HiFiGANMelTransform, SpeechWindowDataset, TinySpeechVAE  # noqa: E402


def test_i001_default_shape_remains_supported() -> None:
    model = TinySpeechVAE()
    value = torch.zeros(2, 1, 64, 256)
    output = model(value, sample_posterior=False)
    assert output.reconstruction.shape == value.shape
    assert output.mean.shape == (2, 4, 16, 64)


def test_i002_80_mel_shape_and_gradient() -> None:
    model = TinySpeechVAE(n_mels=80, window_frames=256)
    value = torch.randn(2, 1, 80, 256)
    output = model(value)
    assert output.reconstruction.shape == value.shape
    assert output.mean.shape == (2, 4, 20, 64)
    output.reconstruction.square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_i002_feature_transform_has_80_semantic_channels() -> None:
    feature = HiFiGANMelTransform()(torch.zeros(1, 22050))
    assert feature.shape[1] == 80
    assert torch.isfinite(feature).all()


def test_i002_dataset_when_prepared() -> None:
    config_path = ROOT / "configs/p3_w02_i002_hifigan_vae.json"
    data_path = ROOT / "data/speech_vae/p3_w02_hifigan_v2/window_manifest.jsonl"
    if not data_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = SpeechWindowDataset(ROOT, data_dir=config["output_data"], splits="train", config=config)
    item = dataset[0]
    assert item["window"].shape == (1, 80, 256)
    assert item["mask"].shape == (256,)
    assert np.isfinite(item["window"].numpy()).all()


def test_i002_preflight_when_completed() -> None:
    path = ROOT / "results/P3-W02_speech_vae/preflight/PF002_hifigan_vae/preflight_report.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["all_pretraining_checks_passed"] is True
    assert all(item["passed"] for item in report["checks"])


def test_i002_formal_artifacts_when_completed() -> None:
    branch = ROOT / "results/P3-W02_speech_vae/iterations/I002/branches/hifigan_vae"
    acceptance_path = branch / "metrics/acceptance_report.json"
    if not acceptance_path.exists():
        return
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    assert acceptance["all_machine_blocking_acceptance_items_passed"] is True
    assert acceptance["teacher_perceptual_decision"] == "pending"
    for relative in (
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "checkpoints/checkpoint_manifest.json",
        "inference_package.pt",
        "metrics/evaluation_metrics.json",
        "metrics/utterance_metrics.json",
        "metrics/comparison_metrics.json",
        "predictions/test_window_predictions.jsonl",
        "examples/capability_examples.md",
        "examples/capability_examples.json",
        "evidence_index.json",
    ):
        assert (branch / relative).is_file(), relative
