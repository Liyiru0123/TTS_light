from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/P3-W05_model_integration"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_int_p3w05_001_manifest_has_exactly_six_distinct_tasks() -> None:
    manifest = load(RESULT / "model_manifest.json")
    assert manifest["model_count"] == 6
    assert len(manifest["models"]) == 6
    assert len({row["model_id"] for row in manifest["models"]}) == 6
    assert len({row["task"] for row in manifest["models"]}) == 6
    tts = next(row for row in manifest["models"] if row["model_id"] == "tiny_speech_flow")
    assert tts["technical_candidate"] != tts["teaching_use_version"]
    assert all(row["official_release"] is None for row in manifest["models"])


def test_int_p3w05_002_assets_exist_and_hashes_match() -> None:
    report = load(RESULT / "integration_report.json")
    for row in report["model_assets"]["items"]:
        assert row["exists"] and row["sha256_matches"]
        assert sha256(ROOT / row["path"]) == row["expected_sha256"]
    assert report["model_assets"]["sha256_match_rate"] == 1.0


def test_int_p3w05_003_all_six_real_probes_succeeded_in_project_venv() -> None:
    rows = [json.loads(line) for line in (RESULT / "runtime_checks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert {row["model_id"] for row in rows} == {
        "tiny_vlm", "tiny_image_ldm", "tiny_asr", "tiny_speech_flow", "tiny_video_mllm", "tiny_t2v"
    }
    assert all(row["success"] and row["strict_load"] for row in rows)
    assert all("最小系统研发\\.venv\\Scripts\\python.exe" in row["python_executable"] for row in rows)


def test_int_p3w05_004_dependencies_and_chain_boundaries() -> None:
    report = load(RESULT / "integration_report.json")
    assert all(row["exists"] and row["sha256_matches"] for row in report["dependencies"])
    assert report["chains"]["image"]["vlm_is_unconditional_ldm_judge"] is False
    assert report["chains"]["speech"]["absolute_short_segment_asr_is_gate"] is False
    assert report["chains"]["video"]["videoqa_is_t2v_acceptance_judge"] is False


def test_ped_p3w05_001_capability_examples_are_complete_and_media_exist() -> None:
    payload = load(RESULT / "capability_examples.json")
    assert len(payload["examples"]) == 6
    for row in payload["examples"]:
        assert row["input"] and row["actual"] and row["version"] and row["boundary"] and row["source"]
    for path in (
        RESULT / "runtime_outputs/tiny_image_ldm_anchor.png",
        RESULT / "runtime_outputs/tiny_speech_flow_i004.wav",
        RESULT / "runtime_outputs/tiny_t2v/comparison.gif",
    ):
        assert path.exists() and path.stat().st_size > 0


def test_ped_p3w05_002_gap_ledger_is_six_by_eight() -> None:
    gaps = load(RESULT / "gap_ledger.json")
    assert len(gaps["models"]) == 6
    assert len(gaps["dimensions"]) == 8
    assert all(gaps["models"][model][dimension].strip() for model in gaps["models"] for dimension in gaps["dimensions"])
    assert load(RESULT / "integration_report.json")["gap_ledger"]["dimension_coverage"] == 1.0


def test_reg_p3w05_001_delivery_contract_and_summary_status() -> None:
    required = (
        ROOT / "docs/P3-W05_六类模型集成与差距台账设计.md",
        ROOT / "docs/P3-W05_测试计划.md",
        ROOT / "docs/P3-W05_指标与术语说明.md",
        ROOT / "docs/P3-W05_六类模型差距台账.md",
        ROOT / "docs/P3-W05_总结报告.md",
        RESULT / "README.md",
        RESULT / "model_manifest.json",
        RESULT / "integration_report.json",
        RESULT / "evidence_index.json",
        RESULT / "capability_examples.md",
        RESULT / "capability_examples.json",
    )
    assert all(path.exists() and path.stat().st_size > 0 for path in required)
    summary = (ROOT / "docs/P3-W05_总结报告.md").read_text(encoding="utf-8")
    assert "已验收" in summary
    assert (ROOT / "docs/P3-W05_教学验收决定.md").is_file()
    assert "official_release" in summary and "P4-W01" in summary


def test_reg_p3w05_002_evidence_index_values_match_sources() -> None:
    evidence = load(RESULT / "evidence_index.json")
    report = load(RESULT / "integration_report.json")
    direct = {
        "runtime.runtime_coverage": report["runtime"]["runtime_coverage"],
        "model_assets.sha256_match_rate": report["model_assets"]["sha256_match_rate"],
        "runtime.total_wall_seconds": report["runtime"]["total_wall_seconds"],
        "runtime.maximum_peak_rss_bytes": report["runtime"]["maximum_peak_rss_bytes"],
        "gap_ledger.dimension_coverage": report["gap_ledger"]["dimension_coverage"],
    }
    for claim in evidence["claims"]:
        if claim.get("source") == "integration_report.json":
            assert claim["display_value"] == direct[claim["field"]]


def test_reg_p3w05_003_upstream_assets_are_immutable() -> None:
    immutability = load(RESULT / "integration_report.json")["immutability"]
    assert immutability["passed"]
    assert immutability["before"] == immutability["after"]
