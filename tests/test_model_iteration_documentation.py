"""Governance regression tests for four-document iteration trajectories."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_model_iteration_documentation.py"
AUDIT = ROOT / "results/P0-W04_governance/model_iteration_documentation_audit.json"


def load_module():
    spec = importlib.util.spec_from_file_location("iteration_doc_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_model_work_packages_have_closed_trajectory_documentation() -> None:
    audit = load_module().build_audit()
    assert audit["all_passed"] is True
    assert audit["passed_packages"] == 9
    assert audit["scope"]["primary_documents"] == 36
    assert audit["scope"]["registered_iterations"] == 24


def test_every_multi_iteration_primary_document_covers_registry_ids() -> None:
    audit = load_module().build_audit()
    multi = [row for row in audit["packages"] if row["mode"] == "multi_iteration"]
    assert len(multi) == 6
    for package in multi:
        assert package["iteration_manifest_paths_exist"] is True
        assert package["expected_iteration_ids"]
        for document in package["documents"]:
            assert document["exists"] is True
            assert document["missing_iteration_ids"] == []
            assert document["has_role_specific_trajectory_section"] is True
            assert document["passed"] is True


def test_single_cycle_packages_explicitly_distinguish_development_attempts() -> None:
    audit = load_module().build_audit()
    single = [row for row in audit["packages"] if row["mode"] == "single_cycle"]
    assert {row["work_package"] for row in single} == {"P2-W01", "P2-W02", "P2-W03"}
    for package in single:
        assert package["expected_iteration_ids"] == []
        for document in package["documents"]:
            assert document["has_single_cycle_scope_statement"] is True
            assert document["passed"] is True


def test_persisted_audit_matches_live_audit() -> None:
    persisted = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert persisted == load_module().build_audit()


def test_long_term_governance_rule_is_recorded() -> None:
    mechanism = (ROOT / "docs/项目记忆与工作包检查机制.md").read_text(encoding="utf-8")
    spec = (ROOT / "最小系统研发规格与约束.md").read_text(encoding="utf-8")
    assert "模型迭代技术轨迹的四文档闭环" in mechanism
    assert "只更新`results/iterations/`、配置或单轮报告不算交付完成" in spec


def test_trajectory_section_is_the_final_top_level_section() -> None:
    """Prevent a trajectory insertion from separating an old heading and body."""
    module = load_module()
    audit = module.build_audit()
    for package in audit["packages"]:
        for document in package["documents"]:
            text = (ROOT / document["path"]).read_text(encoding="utf-8")
            headings = [line for line in text.splitlines() if line.startswith("## ")]
            assert headings
            assert any(marker in headings[-1] for marker in module.ROLE_MARKERS[document["role"]]), (
                document["path"],
                headings[-1],
            )
