from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.governance import validate_summary_contract  # noqa: E402


def _fixture(tmp_path: Path) -> dict:
    (tmp_path / "docs").mkdir()
    (tmp_path / "results/pkg").mkdir(parents=True)
    (tmp_path / "evidence.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "docs/P9-W99_I001总结报告.md").write_text("历史\n", encoding="utf-8")
    (tmp_path / "docs/P9-W99_I002总结报告.md").write_text("迭代\n", encoding="utf-8")
    (tmp_path / "docs/P9-W99_总结报告.md").write_text(
        "# 总报告\n\n当前候选为`I002/control`。\n\n## 1. 直观能力样例\n\n"
        "这里解释正确样例、错误样例和边界，并引用`evidence.json`。\n",
        encoding="utf-8",
    )
    (tmp_path / "results/pkg/current_candidate.json").write_text(
        json.dumps({"current_candidate": {"iteration_id": "I002", "branch_id": "control"}}),
        encoding="utf-8",
    )
    return {
        "schema_version": 3,
        "work_package": "P9-W99",
        "result_directory": "results/pkg",
        "summary_document": "docs/P9-W99_总结报告.md",
        "iteration_summary_document": "docs/P9-W99_I002总结报告.md",
        "historical_summary_documents": ["docs/P9-W99_I001总结报告.md"],
        "summary_contract": {
            "document_role": "current_work_package_summary",
            "current_candidate": {"iteration_id": "I002", "branch_id": "control"},
            "sections": [
                {
                    "heading": "## 1. 直观能力样例",
                    "minimum_characters": 20,
                    "required_terms": ["错误样例"],
                }
            ],
        },
        "primary_evidence_paths": ["evidence.json"],
    }


def test_summary_contract_accepts_complete_current_summary(tmp_path: Path) -> None:
    assert validate_summary_contract(tmp_path, _fixture(tmp_path))["verified"]


def test_summary_contract_rejects_missing_section(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    contract["summary_contract"]["sections"][0]["heading"] = "## 2. 主要证据路径"
    result = validate_summary_contract(tmp_path, contract)
    assert not result["verified"]
    assert any("missing section" in error for error in result["errors"])


def test_summary_contract_rejects_candidate_drift(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    contract["summary_contract"]["current_candidate"]["iteration_id"] = "I001"
    result = validate_summary_contract(tmp_path, contract)
    assert not result["verified"]
    assert any("differs from current_candidate.json" in error for error in result["errors"])


def test_summary_contract_rejects_uncited_primary_evidence(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    (tmp_path / "docs/P9-W99_总结报告.md").write_text(
        "当前候选为I002/control。\n\n## 1. 直观能力样例\n\n这里解释正确样例和错误样例，但没有列出证据文件。\n",
        encoding="utf-8",
    )
    result = validate_summary_contract(tmp_path, contract)
    assert not result["verified"]
    assert any("does not cite primary evidence" in error for error in result["errors"])
