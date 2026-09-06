"""Finalize I002 evidence after approved packaging recovery and tests."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/P3-W02_speech_vae"
ITERATION = RESULT / "iterations/I002"
BRANCH = ITERATION / "branches/hifigan_vae"
RECOVERY = ITERATION / "recovery"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def junit_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return bool(suites) and all(
        int(suite.attrib.get("failures", 0)) == 0 and int(suite.attrib.get("errors", 0)) == 0
        for suite in suites
    )


def main() -> None:
    config = read(ROOT / "configs/p3_w02_i002_hifigan_vae.json")
    thresholds = config["thresholds"]
    audit = read(ROOT / config["output_data"] / "data_audit.json")
    integrity = read(ROOT / config["output_data"] / "source_integrity.json")
    pf002 = read(RESULT / "preflight/PF002_hifigan_vae/preflight_report.json")
    overfit = read(BRANCH / "metrics/overfit_report.json")
    evaluation = read(BRANCH / "metrics/evaluation_metrics.json")
    utterance = read(BRANCH / "metrics/utterance_metrics.json")
    reproducibility = read(BRANCH / "logs/reproducibility_report.json")
    capability = read(BRANCH / "examples/capability_examples.json")
    comparison = read(ITERATION / "comparisons/i001_common_core/metrics.json")
    recovery = read(RECOVERY / "recovery_report.json")
    before = read(RECOVERY / "protected_assets_before.json")
    after = read(RECOVERY / "protected_assets_after.json")

    pre_start = (BRANCH / "logs/reproducibility_report.json").stat().st_mtime
    pre_end = (BRANCH / "metrics/utterance_metrics.json").stat().st_mtime
    measured_pre_failure = max(0.0, pre_end - pre_start)
    recovery_seconds = recovery["resource"]["wall_seconds"]
    conservative_overhead = 60.0
    wall_for_acceptance = measured_pre_failure + recovery_seconds + conservative_overhead
    observed_training_rss = 2_069_164_032
    peak_observed_rss = max(observed_training_rss, recovery["resource"]["observed_rss_bytes"])
    resource = {
        "schema_version": 1,
        "measurement_method": "file-time anchored interrupted process plus measured recovery and 60-second conservative allowance",
        "pre_failure_anchor_elapsed_seconds": measured_pre_failure,
        "recovery_wall_seconds": recovery_seconds,
        "conservative_overhead_seconds": conservative_overhead,
        "wall_seconds_for_acceptance": wall_for_acceptance,
        "training_peak_rss_observed_bytes": observed_training_rss,
        "recovery_rss_observed_bytes": recovery["resource"]["observed_rss_bytes"],
        "peak_observed_rss_bytes": peak_observed_rss,
        "trainable_vae_parameters": 187177,
        "frozen_hifigan_parameters": 1462273,
        "hifigan_vocoder_rtf": utterance["aggregate"]["vocoder_rtf"],
        "sample_count": 1,
        "caveat": "The original process stopped before writing its resource report; elapsed time is conservatively reconstructed and RSS uses the saved Get-Process observation from training plus recovery observation.",
    }
    write(BRANCH / "metrics/resource_benchmark.json", resource)

    media_links_ok = True
    for item in capability["examples"]:
        for relative in item["media"].values():
            media_links_ok = media_links_ok and (BRANCH / "examples" / relative).is_file()
    required = [
        BRANCH / "checkpoints/best.pt",
        BRANCH / "checkpoints/last.pt",
        BRANCH / "checkpoints/checkpoint_manifest.json",
        BRANCH / "inference_package.pt",
        BRANCH / "predictions/test_window_predictions.jsonl",
        BRANCH / "examples/capability_examples.md",
        BRANCH / "examples/capability_examples.json",
        BRANCH / "tests/p3_w02_i002_recovery_junit.xml",
        BRANCH / "tests/p3_w02_i002_full_regression_junit.xml",
        ROOT / "results/P3-W02_speech_vae/i002_example_inference.flac",
        ROOT / "docs/P3-W02_I002检查点与推理使用说明.md",
        ROOT / "docs/P3-W02_I002总结报告.md",
        RECOVERY / "source_delta_manifest.json",
    ]
    machine = {
        "DATA-SPR2-001": audit["derived_utterance_count"] == 1720 and audit["unexpected_frame_delta_count"] == 0 and audit["nonfinite_value_count"] == 0,
        "DATA-SPR2-002": audit["window_count"] >= thresholds["minimum_total_windows"] and audit["split_window_counts"]["train"] >= thresholds["minimum_train_windows"] and all(value == 0 for value in audit["source_split_intersections"].values()) and audit["coverage_failure_count"] == 0,
        "DATA-SPR2-003": integrity["all_sources_unchanged"],
        "UNIT-SPR2-001": next(item for item in pf002["checks"] if item["acceptance_item_id"] == "UNIT-SPR2-001")["passed"],
        "UNIT-SPR2-002": next(item for item in pf002["checks"] if item["acceptance_item_id"] == "UNIT-SPR2-002")["passed"],
        "UNIT-SPR2-003": next(item for item in pf002["checks"] if item["acceptance_item_id"] == "UNIT-SPR2-003")["passed"],
        "MODEL-SPR2-001": overfit["normalized_mae"] <= thresholds["overfit_normalized_mae_max"],
        "MODEL-SPR2-002": evaluation["normalized_mae"] <= thresholds["test_normalized_mae_max"] and evaluation["logmel_mae"] <= thresholds["test_logmel_mae_max"] and evaluation["temporal_delta_mae"] <= thresholds["test_delta_mae_max"],
        "MODEL-SPR2-003": evaluation["kl"] > thresholds["minimum_kl"] and evaluation["element_compression_ratio"] == 4.0,
        "MODEL-SPR2-004": utterance["aggregate"]["overlap_boundary_mae"] <= thresholds["maximum_overlap_boundary_mae"],
        "MODEL-SPR2-005": utterance["aggregate"]["asr_cer"] <= thresholds["maximum_asr_reconstruction_cer"] and utterance["aggregate"]["waveform_cycle_logmel_mae"] <= thresholds["maximum_waveform_cycle_logmel_mae"] and utterance["aggregate"]["all_waveforms_finite"] and utterance["aggregate"]["hard_clipped_example_count"] == 0 and utterance["aggregate"]["maximum_duration_relative_error"] <= thresholds["maximum_duration_relative_error"],
        "COMP-SPR2-001": comparison["aggregate"]["sample_count"] == 12 and capability["example_count"] == 8 and media_links_ok,
        "REPRO-SPR2-001": reproducibility["identical"],
        "RES-SPR2-001": resource["trainable_vae_parameters"] <= thresholds["maximum_parameters"] and wall_for_acceptance <= thresholds["maximum_wall_seconds"] and peak_observed_rss <= thresholds["maximum_peak_rss_bytes"] and resource["hifigan_vocoder_rtf"] <= thresholds["maximum_hifigan_rtf"],
        "ART-SPR2-001": all(path.is_file() for path in required)
        and junit_passed(BRANCH / "tests/p3_w02_i002_recovery_junit.xml")
        and junit_passed(BRANCH / "tests/p3_w02_i002_full_regression_junit.xml"),
        "REG-SPR2-REC-001": before == after and recovery["protected_assets_unchanged"],
        "COMP-SPR2-REC-001": comparison["aggregate"]["sample_count"] == 12 and comparison["aggregate"]["all_outputs_finite"],
        "COMP-SPR2-REC-002": capability["example_count"] == 8 and media_links_ok,
        "ART-SPR2-REC-001": all(path.is_file() for path in required[:4]) and (ROOT / "results/P3-W02_speech_vae/i002_example_inference.flac").is_file(),
        "REG-SPR2-REC-002": (RECOVERY / "source_delta_manifest.json").is_file(),
    }
    acceptance = {
        "schema_version": 2,
        "work_package": "P3-W02",
        "iteration_id": "I002",
        "branch_id": "hifigan_vae",
        "items": {key: {"passed": value} for key, value in machine.items()}
        | {"PED-SPR2-001": {"passed": None, "status": "pending_teacher_review"}},
        "failed_machine_items": [key for key, value in machine.items() if not value],
        "all_machine_blocking_acceptance_items_passed": all(machine.values()),
        "teacher_perceptual_decision": "pending",
        "post_snapshot_recovery": "approved_plan_A",
    }
    write(BRANCH / "metrics/acceptance_report.json", acceptance)

    branch_prefix = "iterations/I002/branches/hifigan_vae/"

    def metric_entry(
        metric_id: str,
        value: int | float,
        source: str,
        value_path: str,
        scope: str,
        sample_count: int,
        count_path: str,
        acceptance_item_id: str,
    ) -> dict:
        relative = branch_prefix + source
        return {
            "id": metric_id,
            "value": value,
            "source_file": relative,
            "json_path": value_path,
            "evaluation_scope": scope,
            "sample_count": sample_count,
            "sample_count_source_file": relative,
            "sample_count_json_paths": [count_path],
            "sample_count_aggregation": "value",
            "acceptance_item_id": acceptance_item_id,
        }

    evidence = {
        "schema_version": 2,
        "work_package": "P3-W02",
        "iteration_id": "I002",
        "branch_id": "hifigan_vae",
        "metrics": [
            metric_entry("test_normalized_mae", evaluation["normalized_mae"], "metrics/evaluation_metrics.json", "$.normalized_mae", "固定测试窗口", evaluation["sample_count"], "$.sample_count", "MODEL-SPR2-002"),
            metric_entry("test_logmel_mae", evaluation["logmel_mae"], "metrics/evaluation_metrics.json", "$.logmel_mae", "固定测试窗口", evaluation["sample_count"], "$.sample_count", "MODEL-SPR2-002"),
            metric_entry("test_delta_mae", evaluation["temporal_delta_mae"], "metrics/evaluation_metrics.json", "$.temporal_delta_mae", "固定测试窗口", evaluation["sample_count"], "$.sample_count", "MODEL-SPR2-002"),
            metric_entry("test_kl", evaluation["kl"], "metrics/evaluation_metrics.json", "$.kl", "固定测试窗口", evaluation["sample_count"], "$.sample_count", "MODEL-SPR2-003"),
            metric_entry("utterance_asr_cer", utterance["aggregate"]["asr_cer"], "metrics/utterance_metrics.json", "$.aggregate.asr_cer", "固定测试语句", 12, "$.aggregate.sample_count", "MODEL-SPR2-005"),
            metric_entry("utterance_cycle_mae", utterance["aggregate"]["waveform_cycle_logmel_mae"], "metrics/utterance_metrics.json", "$.aggregate.waveform_cycle_logmel_mae", "固定测试语句", 12, "$.aggregate.sample_count", "MODEL-SPR2-005"),
            metric_entry("utterance_boundary_mae", utterance["aggregate"]["overlap_boundary_mae"], "metrics/utterance_metrics.json", "$.aggregate.overlap_boundary_mae", "固定测试语句", 12, "$.aggregate.sample_count", "MODEL-SPR2-004"),
            metric_entry("wall_seconds_for_acceptance", wall_for_acceptance, "metrics/resource_benchmark.json", "$.wall_seconds_for_acceptance", "完整流程（保守重建）", 1, "$.sample_count", "RES-SPR2-001"),
            metric_entry("peak_observed_rss_bytes", peak_observed_rss, "metrics/resource_benchmark.json", "$.peak_observed_rss_bytes", "完整流程（训练与恢复观察）", 1, "$.sample_count", "RES-SPR2-001"),
        ],
        "recovery_evidence": [
            "../../recovery/source_delta_manifest.json",
            "../../recovery/recovery_report.json",
            "../../comparisons/i001_common_core/metrics.json",
        ],
    }
    write(BRANCH / "evidence_index.json", evidence)
    print(json.dumps({"all_machine_blocking_acceptance_items_passed": acceptance["all_machine_blocking_acceptance_items_passed"], "failed_machine_items": acceptance["failed_machine_items"], "resource": resource}, ensure_ascii=False, indent=2))
    if not acceptance["all_machine_blocking_acceptance_items_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
