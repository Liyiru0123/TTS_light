from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/P3-W04_video_generation"


def test_current_summary_is_i008_first_and_complete() -> None:
    text = (ROOT / "docs/P3-W04_总结报告.md").read_text(encoding="utf-8")
    assert "当前候选：`I008/generated_motion_module`" in text
    assert "最终架构（I006）" not in text
    for iteration in range(1, 9):
        assert f"I{iteration:03d}" in text
    for heading in (
        "任务、数据与评价范围",
        "当前候选I008模型结构",
        "核心量化结果",
        "I001— I008完整轮次对照",
        "直观能力样例",
        "检查点、推理和复现",
        "与当前代表性系统的非规模差距",
        "验收、验证与主要证据",
    ):
        assert heading in text


def test_root_evidence_and_showcase_match_candidate() -> None:
    candidate = json.loads((RESULT / "current_candidate.json").read_text(encoding="utf-8"))["current_candidate"]
    evidence = json.loads((RESULT / "evidence_index.json").read_text(encoding="utf-8"))
    showcase = json.loads((RESULT / "examples/capability_examples.json").read_text(encoding="utf-8"))
    assert candidate["iteration_id"] == "I008"
    assert evidence["current_candidate"] == {
        "iteration_id": candidate["iteration_id"],
        "branch_id": candidate["branch_id"],
        "pointer": "results/P3-W04_video_generation/current_candidate.json",
    }
    assert showcase["current_candidate"]["iteration_id"] == "I008"
    assert showcase["example_count"] == 5
    for example in showcase["examples"]:
        assert example["display_labels"] == ["Independent LDM", "Motion OFF", "Motion ON"]
        assert (ROOT / example["asset"]).is_file()
        assert example["metrics"]["motion_on"]["direction_accuracy"] == 1.0
        assert example["metrics"]["motion_on"]["timing_accuracy"] == 1.0


def test_usage_has_executed_inference_and_no_unsafe_in_place_command() -> None:
    text = (ROOT / "docs/P3-W04_检查点与推理使用说明.md").read_text(encoding="utf-8")
    assert "scripts/p3_w04_i008_infer_example.py" in text
    assert "不得在当前工作区原地重跑" in text
    assert (ROOT / "results/P3-W04_video_generation/inference_example/comparison.gif").is_file()
    result = json.loads(
        (ROOT / "results/P3-W04_video_generation/inference_example/inference_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["iteration_id"] == "I008"
    assert result["branch_id"] == "generated_motion_module"
    assert result["metrics"]["motion_on"]["direction_accuracy"] == 1.0
    assert result["metrics"]["motion_on"]["timing_accuracy"] == 1.0


def test_readme_uses_safe_i008_entry() -> None:
    text = (RESULT / "README.md").read_text(encoding="utf-8")
    assert "p3_w04_i008_infer_example.py" in text
    assert "p3_w04_train.py" not in text
    assert "当前候选：`I008/generated_motion_module`" in text

