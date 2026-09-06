from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.grounded_evaluation import load_grounded_checkpoint
from mini_multimodal.ldm.revision_evaluation import sample_latents_v2
from mini_multimodal.ldm.revision_training import load_config


RESULT = ROOT / "results/P2-W04_tiny_image_ldm/revision_v2"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(metric: object) -> float:
    if isinstance(metric, dict) and "value" in metric:
        return float(metric["value"])
    return float(metric)


def test_attempt2_summary_matches_machine_metrics() -> None:
    summary = _json(RESULT / "attempt2_summary.json")
    generation = _json(RESULT / "grounded_generation_metrics.json")
    denoising = _json(RESULT / "grounded_denoising_metrics.json")
    assert summary["status"] == "failed_frozen_generation_thresholds"
    assert summary["training_holdout_object_count"] == 0
    assert summary["denoising_verified"] is True
    assert summary["denoising_verified"] == denoising["verified_for_generation_evaluation"]
    assert summary["seen_single_verified"] is False
    assert summary["unseen_single_verified"] is False
    for subset in ("seen_single", "unseen_single"):
        for name, value in summary[subset].items():
            assert value == _value(generation["metrics"][subset][name])


def test_attempt2_checkpoint_manifest_hashes_and_payload_are_exact() -> None:
    manifest = _json(RESULT / "training_grounded/checkpoints/checkpoint_manifest.json")
    assert manifest["status"] == "diagnostic_failed_generation_thresholds"
    assert manifest["inference_package"] is None
    for name in ("best", "last"):
        item = manifest[name]
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert _sha(path) == item["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["epoch"] == item["epoch"]
        assert payload["best_metric"] == item["best_metric"]


def test_attempt2_best_checkpoint_strict_load_and_text_only_inference() -> None:
    config = load_config(ROOT)
    package = load_grounded_checkpoint(ROOT, config)
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


def test_attempt2_examples_match_grounded_prediction_records_and_images() -> None:
    payload = _json(RESULT / "attempt2_capability_examples.json")
    predictions = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (RESULT / "grounded_generation_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    assert len(payload["examples"]) == 8
    assert sum(row["is_fully_correct"] for row in payload["examples"]) == 2
    assert {row["category"] for row in payload["examples"]} >= {
        "完整正确",
        "圆形不确定",
        "正方形错误",
        "五边形错误",
        "碎片/数量错误",
        "未见组合错误",
        "多对象边界",
    }
    for example in payload["examples"]:
        source = predictions[example["record_id"]]
        assert example["generated_image"] == source["generated_image"]
        assert example["generated_image_sha256"] == source["generated_image_sha256"]
        assert example["pairs"] == source["pairs"]
        image = ROOT / example["generated_image"]
        assert image.is_file()
        assert _sha(image) == example["generated_image_sha256"]
    markdown = (RESULT / "attempt2_capability_examples.md").read_text(encoding="utf-8")
    assert markdown.count("## ") == 8


def test_attempt2_evidence_index_resolves_exact_source_values() -> None:
    index = _json(RESULT / "attempt2_evidence_index.json")
    assert len(index["entries"]) == 20
    for entry in index["entries"]:
        source = _json(RESULT / entry["source_file"])
        value: object = source
        for part in entry["field_path"].split("."):
            value = value[part]  # type: ignore[index]
        assert entry["value"] == _value(value)


def test_frozen_evaluation_and_first_round_evidence_remain_unchanged() -> None:
    freeze = _json(RESULT / "threshold_freeze.json")
    assert freeze["thresholds"]["seen_single"]["shape_accuracy"] == 0.7764705882352941
    assert freeze["thresholds"]["seen_single"]["count_accuracy"] == 0.85
    assert freeze["thresholds"]["seen_single"]["fully_correct_rate"] == 0.45
    assert freeze["evaluation_manifest_sha256"] == "2c4f00abc1b8d0a12f2c218c47350b8275a2170f392e7533e60bc62696835543"

    attempt1 = _json(RESULT / "attempt1_summary.json")
    assert attempt1["checkpoint_sha256"] == "47a8009cb065156d1fddd3646acaf4d153dac6c99f9c9972dbda1d581da501df"
    assert (RESULT / "generation_metrics.json").is_file()
    assert (RESULT / "attempt1_capability_examples.json").is_file()

