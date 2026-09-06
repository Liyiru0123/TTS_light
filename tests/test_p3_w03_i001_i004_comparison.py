"""Regression checks for the frozen P3-W03 I001/I004 comparison evidence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3_w03_i001_i004_comparison.json"
RESULTS = ROOT / "results/P3-W03_tiny_tts/comparisons/I001_vs_I004"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_value(payload: dict, dotted_field: str):
    value = payload
    for part in dotted_field.split("."):
        value = value[part]
    return value


def test_formal_run_is_complete_and_strictly_paired() -> None:
    config = read_json(CONFIG)
    manifest = read_json(RESULTS / "formal/sample_manifest.json")
    rows = [
        json.loads(line)
        for line in (RESULTS / "formal/predictions/paired_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert manifest["count"] == config["formal_units"] == 120
    assert len(rows) == 120 * 2 * 2 * 3 == 1440
    expected = {
        (unit_id, model, track, factor)
        for unit_id in manifest["unit_ids"]
        for model in ("I001", "I004")
        for track in config["tracks"]
        for factor in config["factors"]
    }
    actual = {
        (row["unit_id"], row["model_id"], row["track"], row["factor"])
        for row in rows
    }
    assert actual == expected


def test_checkpoints_were_read_only_and_still_exist() -> None:
    config = read_json(CONFIG)
    audit = read_json(RESULTS / "formal/asset_audit.json")

    assert audit["unchanged"] is True
    assert audit["before"] == audit["after"]
    assert (ROOT / config["i001_checkpoint"]).is_file()
    assert (ROOT / config["i004_checkpoint"]).is_file()


def test_asr_is_diagnostic_only_after_failed_calibration() -> None:
    calibration = read_json(RESULTS / "formal/metrics/asr_calibration.json")

    assert calibration["eligible_as_version_judge"] is False
    assert calibration["decision"] == "diagnostic_only"
    thresholds = calibration["thresholds"]
    for source in calibration["sources"].values():
        assert source["n"] == 120
        assert source["character_accuracy"] < thresholds["character_accuracy"]
        assert source["keyword_recall"] < thresholds["keyword_recall"]


def test_objective_decision_matches_preregistered_gates() -> None:
    decision = read_json(RESULTS / "formal/metrics/objective_decision.json")

    assert decision["sample_count"] == 120
    assert decision["generated_condition_count"] == 1440
    assert decision["checks"] == {
        "normal_noninferiority": True,
        "scaled_average_superiority": False,
        "end_to_end_noninferiority": True,
        "duration_accuracy": True,
        "finite_and_unclipped": True,
        "asr_eligible": False,
    }
    assert decision["candidate_pointer_changed"] is False
    assert decision["paired_bootstrap"]["scaled_controlled_absolute_average"]["ci95_high"] > 0


def test_blind_listening_pack_is_complete_and_balanced() -> None:
    manifest = read_json(RESULTS / "examples/blind_manifest.json")
    answer_key = read_json(RESULTS / "examples/answer_key.json")
    cases = manifest["cases"]

    assert manifest["identity_blinded"] is True
    assert answer_key["do_not_open_before_scoring"] is True
    assert len(cases) == len(answer_key["cases"]) == 12
    assert Counter(case["rate_factor"] for case in cases) == Counter({0.8: 4, 1.0: 4, 1.2: 4})
    assert Counter(case["A"] for case in answer_key["cases"]) == Counter({"I004": 6, "I001": 6})
    assert {case["case_id"] for case in cases} == {case["case_id"] for case in answer_key["cases"]}
    for case in cases:
        assert (RESULTS / "examples" / case["A"]).is_file()
        assert (RESULTS / "examples" / case["B"]).is_file()


def test_completed_blind_review_preserves_dual_version_decision() -> None:
    review = read_json(RESULTS / "examples/blind_review_result.json")

    assert review["case_count"] == 12
    assert review["aggregate"]["completeness"] == {
        "I001_wins": 1, "I004_wins": 0, "ties": 9, "both_incomplete": 2
    }
    assert review["aggregate"]["clarity_naturalness"] == {
        "I001_wins": 0, "I004_wins": 3, "ties": 6, "both_noisy": 3
    }
    assert review["preregistered_human_checks"]["all_required_checks_passed"] is False
    assert review["final_selection"]["technical_candidate"] == {
        "iteration_id": "I001", "branch_id": "main"
    }
    assert review["final_selection"]["teaching_use_version"] == {
        "iteration_id": "I004", "branch_id": "supported_rate"
    }
    assert review["final_selection"]["candidate_pointer_changed"] is False


def test_evidence_index_values_resolve_to_sources() -> None:
    evidence = read_json(RESULTS / "evidence_index.json")

    for claim in evidence["claims"]:
        source = read_json(RESULTS / claim["source"])
        assert nested_value(source, claim["field"]) == claim["display_value"]


def test_required_human_and_machine_readable_deliverables_exist() -> None:
    required = [
        ROOT / "docs/P3-W03_I001-I004客观对比实验设计.md",
        ROOT / "docs/P3-W03_I001-I004客观对比测试计划.md",
        ROOT / "docs/P3-W03_I001-I004客观对比实验报告.md",
        RESULTS / "README.md",
        RESULTS / "examples/盲听评分表.md",
        RESULTS / "formal/metrics/objective_decision.json",
        RESULTS / "formal/metrics/aggregate_metrics.json",
        RESULTS / "formal/metrics/paired_comparison.json",
        RESULTS / "formal/metrics/token_level_comparison.json",
    ]
    assert all(path.is_file() for path in required)
