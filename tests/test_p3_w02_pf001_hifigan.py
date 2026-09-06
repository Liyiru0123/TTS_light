from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.speech_vae import HiFiGANMelTransform, HiFiGANV3, load_frozen_hifigan_v3  # noqa: E402


def test_hifigan_v3_shape_and_freeze() -> None:
    model = HiFiGANV3()
    features = torch.zeros(1, 80, 5)
    with torch.inference_mode():
        waveform = model(features)
    assert waveform.shape == (1, 1, 5 * 256)
    assert torch.isfinite(waveform).all()


def test_feature_matched_mel_shape() -> None:
    transform = HiFiGANMelTransform()
    waveform = torch.zeros(1, 22050)
    features = transform(waveform)
    assert features.shape[0:2] == (1, 80)
    assert torch.isfinite(features).all()


def test_official_weight_strict_load_when_present() -> None:
    path = ROOT / "models/external/hifigan_v3_ljspeech/hifigan_vocoder_v3_ljspeech.pth"
    if not path.exists():
        return
    model = load_frozen_hifigan_v3(str(path))
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_462_273
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_pf001_artifacts_when_completed() -> None:
    report_path = ROOT / "results/P3-W02_speech_vae/preflight/PF001_hifigan/test_report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["machine_feasibility"] == "passed"
    assert report["perceptual_feasibility"] == "pending_teacher_review"
    assert all(item["status"] == "passed" for item in report["checks"] if item["status"] != "pending")


def test_pf001_capability_media_links_when_present() -> None:
    markdown_path = ROOT / "results/P3-W02_speech_vae/preflight/PF001_hifigan/examples/capability_examples.md"
    if not markdown_path.exists():
        return
    links = re.findall(r"\]\(([^)]+)\)", markdown_path.read_text(encoding="utf-8"))
    assert len(links) == 32
    assert all((markdown_path.parent / link).exists() for link in links)
