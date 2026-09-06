from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/P2-W04_tiny_image_ldm"


def test_i005_goal_evidence_remains_a_frozen_pre_iteration_record() -> None:
    evidence_path = RESULT / "planning/I005/goal_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert evidence["status"] == "goal_analysis_only_no_iteration_created"
    assert evidence["seen_single"]["sample_count"] == 225
    assert evidence["seen_single"]["count"] == {
        "additional_successes_needed": 23,
        "all_failures_are_over_or_under_segmentation": True,
        "failures": 56,
        "predicted_object_count_histogram_for_failures": {
            "0": 2,
            "2": 32,
            "3": 15,
            "4": 6,
            "5": 1,
        },
        "required_successes": 192,
        "successes": 169,
    }
    assert evidence["seen_single"]["shape"]["raw_top1_class_matches"] == 184
    assert evidence["seen_single"]["shape"]["accepted_successes"] == 163
    assert evidence["seen_single"]["fully_correct"]["successes"] == 91
    assert evidence["upper_bounds_and_data"]["training_record_count"] == 6000
    assert evidence["upper_bounds_and_data"]["training_holdout_object_count_zero"] is True


def test_i005_goal_document_keeps_model_work_blocked_pending_confirmation() -> None:
    document = (ROOT / "docs/P2-W04_I005目标论证.md").read_text(encoding="utf-8")

    assert "目标状态：教师已确认" in document
    assert "`iterations/I005/`不存在" in document
    assert "尚未实现修正器" in document
    assert "正式训练" in document
    assert "continuation" in document
    assert "clean_refiner" in document

    authorization = json.loads(
        (RESULT / "planning/I005/authorization.json").read_text(encoding="utf-8")
    )
    assert authorization["approved_branches"] == ["continuation", "clean_refiner"]
    assert authorization["formal_iteration_created"] is (RESULT / "iterations/I005").exists()


def test_i005_confirmed_design_freezes_sampler_search_and_formal_branches() -> None:
    spec = json.loads((ROOT / "configs/p2_w04_i005.json").read_text(encoding="utf-8"))
    design = (ROOT / "docs/P2-W04_I005设计.md").read_text(encoding="utf-8")
    test_plan = (ROOT / "docs/P2-W04_I005测试计划.md").read_text(encoding="utf-8")

    assert spec["parent_iteration"] == "I004"
    assert spec["warmstart_branch"] == "control"
    assert spec["formal_branches"] == ["continuation", "clean_refiner"]
    assert [(item["sample_steps"], item["guidance_scale"]) for item in spec["sampling_preflight"]["settings"]] == [
        (25, 2.0),
        (50, 2.0),
        (25, 1.5),
        (25, 2.5),
    ]
    assert "不得根据结果临时追加采样组合" in design
    assert "本文件统一使用“验收项编号”" in test_plan
    assert "关口" not in test_plan
    assert "PRE-SAMP-003" in test_plan
    assert "DEL-I005-003" in test_plan


def test_i005_sampling_preflight_selects_complete_balanced_original_batches() -> None:
    from mini_multimodal.tools.run_p2_w04_i005_sampling_preflight import _choose_batches

    rows = [
        json.loads(line)
        for line in (ROOT / "data/ldm/p2_w04_eval_v2/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    seen = _choose_batches(rows, subset="seen_single", batch_count=6, batch_size=8)
    unseen = _choose_batches(rows, subset="unseen_single", batch_count=3, batch_size=8)

    assert len(seen) == 6 and len(unseen) == 3
    assert seen == sorted(seen) and unseen == sorted(unseen)
    for subset, starts in (("seen_single", seen), ("unseen_single", unseen)):
        selected = [row for start in starts for row in rows[start : start + 8]]
        assert len(selected) == len(starts) * 8
        assert all(row["subset"] == subset for row in selected)
        assert {row["objects"][0]["shape"] for row in selected} == {
            "circle",
            "square",
            "triangle",
            "rectangle",
            "pentagon",
        }


def test_i005_sampling_preflight_reproduces_i004_and_selects_only_higher_cfg() -> None:
    output = RESULT / "planning/I005/sampling_preflight"
    report = json.loads((output / "sampling_preflight_report.json").read_text(encoding="utf-8"))

    assert report["baseline_reproduction"] == {
        "matched_records": 72,
        "total_records": 72,
        "verified": True,
    }
    assert set(report["settings"]) == {"baseline", "more_steps", "lower_cfg", "higher_cfg"}
    assert report["shortlist"]["eligible_settings"] == ["higher_cfg"]
    assert report["shortlist"]["selected_setting"] == "higher_cfg"
    assert report["settings"]["higher_cfg"]["delta_from_baseline"]["seen_single"] == {
        "accepted_shape_successes": 5,
        "count_successes": 3,
        "fully_correct_successes": 6,
        "position_successes": 0,
    }


def test_i005_sampler_full_confirmation_requires_clean_refiner() -> None:
    output = RESULT / "planning/I005/sampling_preflight/full_confirmation"
    metrics = json.loads((output / "full_candidate_metrics.json").read_text(encoding="utf-8"))
    examples = json.loads((output / "full_candidate_examples.json").read_text(encoding="utf-8"))

    assert metrics["sample_count"] == 420
    assert metrics["new_model_trained"] is False
    assert metrics["setting"] == {"id": "higher_cfg", "sample_steps": 25, "guidance_scale": 2.5}
    assert metrics["metrics"]["seen_single"]["count_accuracy"]["successes"] == 173
    assert metrics["metrics"]["seen_single"]["shape_accuracy"]["successes"] == 166
    assert metrics["metrics"]["seen_single"]["fully_correct_rate"]["successes"] == 92
    assert metrics["seen_single_verified"] is False
    assert metrics["unseen_single_verified"] is True
    assert metrics["sampler_only_passed_all_blocking_items"] is False
    assert metrics["decision"]["clean_refiner_implementation_required"] is True
    assert examples["example_count"] == 8
    assert any(item["candidate_fully_correct"] for item in examples["examples"])
    assert any(not item["candidate_fully_correct"] for item in examples["examples"])
    for item in examples["examples"]:
        assert (ROOT / item["baseline_image"]).is_file()
        assert (ROOT / item["candidate_image"]).is_file()
