from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.speech_vae import TinySpeechVAE, load_frozen_hifigan_v3  # noqa: E402

ITERATION = ROOT / "results/P3-W02_speech_vae/iterations/I002"
BRANCH = ITERATION / "branches/hifigan_vae"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_reg_spr2_rec_001_protected_assets_unchanged() -> None:
    before = json.loads((ITERATION / "recovery/protected_assets_before.json").read_text(encoding="utf-8"))
    after = json.loads((ITERATION / "recovery/protected_assets_after.json").read_text(encoding="utf-8"))
    assert before == after
    for item in after["files"]:
        path = ITERATION / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert sha256(path) == item["sha256"]


def test_comp_spr2_rec_001_i001_common_set_is_supplemental_and_complete() -> None:
    report = json.loads((ITERATION / "comparisons/i001_common_core/metrics.json").read_text(encoding="utf-8"))
    assert report["role"] == "supplemental_common_set_evaluation_not_i001_original_acceptance"
    assert report["aggregate"]["sample_count"] == 12
    assert report["aggregate"]["all_outputs_finite"] is True
    assert len(report["record_ids"]) == len(set(report["record_ids"])) == 12
    assert all((ROOT / item["audio_path"]).is_file() for item in report["utterances"])


def test_comp_spr2_rec_002_capability_links_and_records() -> None:
    json_path = BRANCH / "examples/capability_examples.json"
    markdown_path = BRANCH / "examples/capability_examples.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["example_count"] == 8
    assert len({item["source_record_id"] for item in payload["examples"]}) == 8
    for item in payload["examples"]:
        assert len(item["media"]) == 5
        assert all((markdown_path.parent / relative).is_file() for relative in item["media"].values())
    links = re.findall(r"\]\(([^)]+)\)", markdown_path.read_text(encoding="utf-8"))
    assert len(links) == 40
    assert all((markdown_path.parent / link).is_file() for link in links)


def test_art_spr2_rec_001_inference_package_strict_load() -> None:
    package_path = BRANCH / "inference_package.pt"
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    assert sha256(BRANCH / "checkpoints/best.pt") == package["source_checkpoint_sha256"]
    model = TinySpeechVAE(**package["model_config"])
    model.load_state_dict(package["model_state"], strict=True)
    vocoder_path = ROOT / package["hifigan_weight_path"]
    assert sha256(vocoder_path) == package["hifigan_weight_sha256"]
    vocoder = load_frozen_hifigan_v3(str(vocoder_path))
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 80, 256), sample_posterior=False)
        waveform = vocoder(torch.zeros(1, 80, 4))
    assert output.mean.shape == (1, 4, 20, 64)
    assert waveform.shape == (1, 1, 1024)


def test_reg_spr2_rec_002_source_delta_manifest() -> None:
    manifest = json.loads((ITERATION / "recovery/source_delta_manifest.json").read_text(encoding="utf-8"))
    assert manifest["recovery_type"] == "approved_post_snapshot_non_model_packaging_recovery"
    assert manifest["original_source_snapshot_sha256"] == sha256(ITERATION / "source_snapshot.zip")
    assert manifest["teacher_approval"]
    for item in manifest["delta_files"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert sha256(path) == item["sha256"]
