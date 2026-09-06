from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.geometry_evaluation import load_attempt3_checkpoint
from mini_multimodal.ldm.revision_evaluation import sample_latents_v2
from mini_multimodal.ldm.revision_training import load_config


RESULT = ROOT / "results/P2-W04_tiny_image_ldm/revision_v2"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(value: object) -> float:
    if isinstance(value, dict) and "value" in value:
        return float(value["value"])
    return float(value)


def test_attempt3_comparison_matches_all_source_metrics() -> None:
    comparison = _json(RESULT / "attempt3_comparison.json")
    source_names = {
        "second_round": "grounded_generation_metrics.json",
        "control": "attempt3_control_generation_metrics.json",
        "geometry": "attempt3_geometry_generation_metrics.json",
        "geometry_lite": "attempt3_geometry_lite_generation_metrics.json",
    }
    assert comparison["status"] == "failed_frozen_generation_thresholds"
    assert comparison["fixed_evaluation_sample_count"] == 420
    assert comparison["conclusions"]["resolution_is_not_hard_ceiling"] is True
    assert comparison["conclusions"]["geometry_auxiliary_uniformly_better_than_control"] is False
    for branch, filename in source_names.items():
        source = _json(RESULT / filename)
        item = comparison["branches"][branch]
        assert item["checkpoint_sha256"] == source["checkpoint_sha256"]
        for subset in ("seen_single", "unseen_single"):
            for name, value in item[subset].items():
                assert value == _value(source["metrics"][subset][name])


def test_attempt3_key_conclusions_are_exact_and_frozen_gates_still_fail() -> None:
    comparison = _json(RESULT / "attempt3_comparison.json")
    second = comparison["branches"]["second_round"]["seen_single"]
    control = comparison["branches"]["control"]
    geometry = comparison["branches"]["geometry"]
    lite = comparison["branches"]["geometry_lite"]
    assert second["shape_accuracy"] == 0.5911111111111111
    assert control["seen_single"]["shape_accuracy"] == 0.7244444444444444
    assert control["unseen_single_verified"] is True
    assert geometry["seen_single"]["fully_correct_rate"] == 0.4311111111111111
    assert geometry["seen_single"]["count_accuracy"] == 0.7644444444444445
    assert lite["seen_shape_groups"]["circle"] == 0.6666666666666666
    assert all(not comparison["branches"][name]["seen_single_verified"] for name in ("control", "geometry", "geometry_lite"))


def test_attempt3_checkpoint_manifests_are_exact_and_share_warmstart() -> None:
    warmstart = "636821b5acf0441a31520f46e4237fe77bacbcaaed6c3083d69a28386824446c"
    for branch in ("control", "geometry", "geometry_lite"):
        manifest = _json(RESULT / f"training_attempt3_{branch}/checkpoints/checkpoint_manifest.json")
        assert manifest["warmstart_sha256"] == warmstart
        assert manifest["inference_package"] is None
        for name in ("best", "last"):
            item = manifest[name]
            path = ROOT / item["path"]
            assert path.stat().st_size == item["bytes"]
            assert _sha(path) == item["sha256"]
            state = torch.load(path, map_location="cpu", weights_only=False)
            assert state["epoch"] == item["epoch"]
            assert state["best_metric"] == item["best_metric"]


def test_attempt3_checkpoints_strict_load_and_keep_text_only_inference() -> None:
    config = load_config(ROOT)
    for branch in ("control", "geometry", "geometry_lite"):
        package = load_attempt3_checkpoint(ROOT, config, branch)
        result = sample_latents_v2(
            package["model"],
            package["schedule"],
            package["tokenizer"],
            ["第1帧中有1个可见对象。对象1是绿色的中等大小正方形，位于左侧区域。"],
            sample_steps=2,
            guidance_scale=2.0,
            seed=17,
        )
        assert result["latent"].shape == (1, 4, 16, 16)
        assert torch.isfinite(result["latent"]).all()


def test_attempt3_capability_comparison_matches_all_prediction_records() -> None:
    payload = _json(RESULT / "attempt3_capability_comparison.json")
    source_names = {
        "second_round": "grounded_generation_predictions.jsonl",
        "control": "attempt3_control_predictions.jsonl",
        "geometry": "attempt3_geometry_predictions.jsonl",
        "geometry_lite": "attempt3_geometry_lite_predictions.jsonl",
    }
    predictions = {
        branch: {
            row["record_id"]: row
            for row in (
                json.loads(line)
                for line in (RESULT / filename).read_text(encoding="utf-8").splitlines()
            )
        }
        for branch, filename in source_names.items()
    }
    assert len(payload["examples"]) == 8
    assert {example["category"] for example in payload["examples"]} >= {
        "稳定正确",
        "圆形改善",
        "正方形改善",
        "五边形改善",
        "几何支联合获益",
        "控制支泛化获益",
        "持续失败",
        "多对象边界",
    }
    for example in payload["examples"]:
        for branch, compact in example["branches"].items():
            source = predictions[branch][example["record_id"]]
            assert compact["pairs"] == source["pairs"]
            assert compact["is_fully_correct"] == source["is_fully_correct"]
            image = ROOT / compact["generated_image"]
            assert image.is_file()
            assert _sha(image) == compact["generated_image_sha256"]
    markdown = (RESULT / "attempt3_capability_comparison.md").read_text(encoding="utf-8")
    lines = markdown.splitlines()
    assert sum(line.startswith("## ") for line in lines) == 8
    assert sum(line.startswith("### ") for line in lines) == 32


def test_attempt3_canonical_capability_examples_are_current_and_traceable() -> None:
    payload = _json(RESULT / "capability_examples.json")
    assert payload["formal_experiment"] == "third_round_geometry_auxiliary"
    assert payload["primary_branch"] == "geometry"
    assert payload["example_count"] == len(payload["examples"]) == 8
    assert sum(bool(item["is_fully_correct"]) for item in payload["examples"]) == 5
    assert sum(not bool(item["is_fully_correct"]) for item in payload["examples"]) == 3
    predictions = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (RESULT / "attempt3_geometry_predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        )
    }
    markdown = (RESULT / "capability_examples.md").read_text(encoding="utf-8")
    assert "P2-W04_指标与术语说明.md" in markdown
    assert "诊断检查点" in markdown
    for example in payload["examples"]:
        source = predictions[example["source_record_id"]]
        assert example["model_outputs"]["pairs"] == source["pairs"]
        assert example["model_outputs"]["predicted_object_count"] == source["predicted_object_count"]
        assert example["is_fully_correct"] == source["is_fully_correct"]
        assert example["source_record_id"] in markdown
        for field in ("input_artifacts", "display_artifacts"):
            for relative in example[field]:
                assert (ROOT / relative).is_file()
                if field == "display_artifacts":
                    assert Path(relative).name in markdown


def test_attempt3_metric_glossary_defines_semantics_and_is_linked() -> None:
    glossary_path = ROOT / "docs/P2-W04_指标与术语说明.md"
    glossary = glossary_path.read_text(encoding="utf-8")
    for required in (
        "normalized_template_iou",
        "boundary_score",
        "fully_correct_rate",
        "is_fully_correct",
        "预处理",
        "聚合",
        "局限",
        "边界分数不直接进入单张图",
    ):
        assert required in glossary
    linked_documents = (
        ROOT / "docs/P2-W04_模型改进第三轮报告.md",
        ROOT / "docs/P2-W04_测试计划.md",
        ROOT / "docs/P2-W04_TinyImageLDM模型卡.md",
        RESULT / "README.md",
        RESULT / "capability_examples.md",
    )
    for document in linked_documents:
        assert "P2-W04_指标与术语说明.md" in document.read_text(encoding="utf-8")


def test_attempt3_fixed_protocol_and_source_assets_remain_unchanged() -> None:
    freeze = _json(RESULT / "threshold_freeze.json")
    expected_eval = "2c4f00abc1b8d0a12f2c218c47350b8275a2170f392e7533e60bc62696835543"
    assert freeze["evaluation_manifest_sha256"] == expected_eval
    assert freeze["thresholds"]["seen_single"]["shape_accuracy"] == 0.7764705882352941
    for branch in ("control", "geometry", "geometry_lite"):
        report = _json(RESULT / f"attempt3_{branch}_generation_metrics.json")
        assert report["evaluation_manifest_sha256"] == expected_eval
        assert report["sample_count"] == 420
        assert report["sample_steps"] == 25
        assert report["guidance_scale"] == 2.0
        assert report["base_seed"] == 20320818
        assert report["training_holdout_object_count"] == 0
    geometry_manifest = _json(ROOT / "data/ldm/p2_w04_geometry_v3/manifest.json")
    assert geometry_manifest["source_manifest_sha256"] == _sha(ROOT / "data/ldm/p2_w04_v2/manifest.json")


def test_attempt3_evidence_index_resolves_all_exact_values() -> None:
    index = _json(RESULT / "attempt3_evidence_index.json")
    assert len(index["entries"]) == 72
    for entry in index["entries"]:
        source = _json(RESULT / entry["source_file"])
        value: object = source
        for part in entry["field_path"].split("."):
            value = value[part]  # type: ignore[index]
        assert entry["value"] == _value(value)


def test_second_round_probe_report_and_attempt3_probe_paths_are_isolated() -> None:
    grounded = _json(RESULT / "grounded_overfit_report.json")
    third = _json(RESULT / "attempt3_overfit_report.json")
    assert grounded["semantic_probes"]["shapes"]["correct_count"] == 3
    assert third["final_probes"]["shapes"]["correct_count"] == 5
    assert all(
        "grounded_overfit/spatial_shape_probe" in item["generated_image"]
        for item in grounded["semantic_probes"]["shapes"]["items"]
    )
    assert all(
        "attempt3_overfit/probes_final" in item["generated_image"]
        for item in third["final_probes"]["shapes"]["items"]
    )
