from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mini_multimodal.training import load_checkpoint
from mini_multimodal.video_mllm import TinyVideoMLLM, VideoQADataset
from mini_multimodal.video_mllm.inference import load_tiny_video_mllm, predict_video


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/P3-W01_tiny_video_mllm"
BRANCH = RESULT / "iterations/I002/branches/temporal_core"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(config: dict, dataset: VideoQADataset, answer_count: int) -> TinyVideoMLLM:
    tokenizer = dataset.tokenizer
    return TinyVideoMLLM(
        vocabulary_size=len(tokenizer),
        answer_class_count=answer_count,
        pad_id=tokenizer.id("<pad>"),
        bos_id=tokenizer.id("<bos>"),
        eos_id=tokenizer.id("<eos>"),
        mask_id=tokenizer.id("<mask>"),
        **config["model"],
    )


def test_checkpoint_manifest_and_resume_assets_are_consistent():
    config = read_json(ROOT / "configs/p3_w01_tiny_video_mllm.json")
    dataset = VideoQADataset(ROOT, data_dir=config["data"]["output_dir"], splits="test")
    answers = read_json(ROOT / config["data"]["output_dir"] / "answer_classes.json")
    manifest = read_json(BRANCH / "checkpoints/checkpoint_manifest.json")
    by_path = {item["path"]: item for item in manifest["artifacts"]}
    assert set(by_path) == {"checkpoints/best.pt", "checkpoints/last.pt", "inference_package.pt"}
    for relative, item in by_path.items():
        path = BRANCH / relative
        assert path.stat().st_size == item["bytes"]
        assert digest(path) == item["sha256"]
    assert by_path["checkpoints/best.pt"]["sha256"] == by_path["inference_package.pt"]["source_checkpoint_sha256"]

    model = make_model(config, dataset, len(answers))
    optimizer = torch.optim.AdamW(model.parameters())
    metadata = load_checkpoint(BRANCH / "checkpoints/last.pt", model=model, optimizer=optimizer, restore_rng=False)
    assert metadata["epoch"] == 17


def test_inference_package_strictly_reproduces_saved_prediction():
    loaded = load_tiny_video_mllm(BRANCH / "inference_package.pt")
    smoke = read_json(BRANCH / "tests/inference_smoke.json")
    data_dir = ROOT / "data/video_qa/p3_w01_v1"
    rows = [json.loads(line) for line in (data_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    row = next(item for item in rows if item["record_id"] == smoke["record_id"])
    with np.load(data_dir / "videos.npz", allow_pickle=False) as cache:
        ids = [str(value) for value in cache["record_ids"].tolist()]
        frames = cache["frames"][ids.index(smoke["record_id"])]
    output = predict_video(loaded, frames, row["question_zh"])
    assert output["classification_answer"] == smoke["classification_output"]
    assert output["generated_answer"] == smoke["generated_output"]
    assert output["spatial_attention_shape"] == [1, 8, 4, 1, 64]


def test_capability_examples_are_traceable_and_include_real_errors():
    showcase = read_json(RESULT / "capability_examples.json")
    assert showcase["example_count"] == 8
    assert showcase["correct_count"] >= 4
    assert showcase["error_or_boundary_count"] >= 3
    predictions = {
        item["record_id"]: item
        for item in (
            json.loads(line)
            for line in (BRANCH / "predictions/test_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    for item in showcase["examples"]:
        source = predictions[item["record_id"]]
        assert item["classification_output"] == source["classification_answer"]
        assert item["generated_output"] == source["generated_answer"]
        assert item["is_fully_correct"] == source["is_fully_correct"]
        assert (ROOT / item["input_artifact"]).is_file()
        assert (ROOT / item["frame_montage"]).is_file()
    markdown = (RESULT / "capability_examples.md").read_text(encoding="utf-8")
    assert markdown.count("## 样例") == 8
    assert "错误或边界" in markdown


def test_acceptance_evidence_maps_to_raw_machine_results():
    acceptance = read_json(BRANCH / "metrics/acceptance_report.json")
    assert set(acceptance["failed_items"]).issubset({"REG-VIDQA-001"})
    assert acceptance["nonblocking_robustness_observations"]["secondary_overfit_seed_passed"] is False
    evidence = read_json(RESULT / "evidence_index.json")
    evaluation = read_json(BRANCH / "metrics/evaluation_metrics.json")
    entry = next(item for item in evidence["entries"] if item["claim_id"] == "test_classification_accuracy")
    assert entry["value"] == evaluation["classification_accuracy"] == 0.9
    assert entry["field_path"] == "classification_accuracy"
    canonical = next(item for item in evidence["entries"] if item["claim_id"] == "canonical_overfit_classification")
    secondary = next(item for item in evidence["entries"] if item["claim_id"] == "secondary_seed_overfit_classification")
    assert canonical["value"] == 0.96875
    assert secondary["value"] == 0.875


def test_iteration_and_preflight_numbering_are_unambiguous():
    registry = read_json(RESULT / "iteration_registry.json")
    assert [item["iteration_id"] for item in registry["iterations"]] == ["I001", "I002", "I003"]
    assert registry["current_candidate"]["iteration_id"] == "I003"
    assert registry["current_candidate"]["branch_id"] == "spacetime_queries"
    assert registry["official_release"] is None
    preflight = read_json(RESULT / "preflight/preflight_registry.json")
    assert [item["preflight_id"] for item in preflight["records"]] == [f"PF{index:03d}" for index in range(1, 10)]
    assert preflight["records"][-1]["status"] == "passed"


def test_current_report_contains_required_delivery_sections():
    report = (ROOT / "docs/P3-W01_总结报告.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. 结论",
        "## 5. 训练和核心结果",
        "## 7. 直观能力样例",
        "## 8. 检查点、推理包与复现",
        "## 9. 与当前代表性系统的差距",
        "## 10. 主要证据路径",
    ):
        assert heading in report
    assert "I003/spacetime_queries" in report
    assert "official_release" in report
