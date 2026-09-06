from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.revision_evaluation import load_revision_checkpoint, sample_latents_v2
from mini_multimodal.ldm.revision_training import load_config


RESULT = ROOT / "results/P2-W04_tiny_image_ldm/revision_v2"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_attempt1_summary_matches_machine_metrics() -> None:
    summary = _json(RESULT / "attempt1_summary.json")
    metrics = _json(RESULT / "generation_metrics.json")
    assert summary["status"] == "failed_frozen_generation_thresholds"
    assert summary["training_holdout_object_count"] == 0
    assert summary["seen_single_verified"] is False
    for name, value in summary["seen_single"].items():
        source = metrics["metrics"]["seen_single"][name]
        source = source["value"] if isinstance(source, dict) else source
        assert value == source


def test_best_and_last_checkpoint_manifest_is_exact() -> None:
    manifest = _json(RESULT / "training_v2/checkpoints/checkpoint_manifest.json")
    for name in ("best", "last"):
        item = manifest[name]
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert _sha(path) == item["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["epoch"] == item["epoch"]
        assert payload["best_metric"] == item["best_metric"]
    assert manifest["inference_package"] is None


def test_best_checkpoint_strict_load_and_fixed_prompt_inference() -> None:
    config = load_config(ROOT)
    package = load_revision_checkpoint(ROOT, config)
    result = sample_latents_v2(
        package["model"],
        package["schedule"],
        package["tokenizer"],
        ["第1帧中有1个可见对象。对象1是绿色的中等大小圆形，位于左侧区域。"],
        sample_steps=2,
        guidance_scale=2.0,
        seed=17,
    )
    assert result["latent"].shape == (1, 4, 16, 16)
    assert torch.isfinite(result["latent"]).all()


def test_failed_overfit_attempts_are_preserved() -> None:
    names = {path.name for path in (RESULT / "failed_attempts").glob("overfit*.json")}
    assert {
        "overfit_60_updates_position_gate.json",
        "overfit_300_updates_position_gate.json",
        "overfit_attention_bias_without_high_noise_gate.json",
        "overfit_high_noise_attention_bias_position_gate.json",
    } <= names


def test_attempt1_examples_match_prediction_records() -> None:
    payload = _json(RESULT / "attempt1_capability_examples.json")
    predictions = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (RESULT / "generation_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    assert len(payload["examples"]) == 8
    assert sum(row["is_fully_correct"] for row in payload["examples"]) == 3
    for example in payload["examples"]:
        source = predictions[example["record_id"]]
        assert example["generated_image"] == source["generated_image"]
        assert example["pairs"] == source["pairs"]
        assert (ROOT / example["generated_image"]).is_file()
    markdown = (RESULT / "attempt1_capability_examples.md").read_text(encoding="utf-8")
    assert markdown.count("## ") == 8
