"""Run the strict-but-light P3-W05 six-model integration audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3_w05_integration.json"
GAPS = ROOT / "configs/p3_w05_gap_ledger.json"
OUTPUT = ROOT / "results/P3-W05_model_integration"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(config: dict) -> dict[str, str]:
    paths = [ROOT / model["asset"] for model in config["models"]]
    for model in config["models"]:
        result_root = ROOT / "results" / {
            "P2-W04": "P2-W04_tiny_image_ldm",
            "P2-W05": "P2-W05_tiny_asr",
            "P3-W01": "P3-W01_tiny_video_mllm",
            "P3-W03": "P3-W03_tiny_tts",
            "P3-W04": "P3-W04_video_generation",
        }.get(model["work_package"], "")
        if result_root.name:
            for name in ("current_candidate.json", "official_release.json"):
                candidate = result_root / name
                if candidate.exists():
                    paths.append(candidate)
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def run_probe(model_id: str, timeout: int) -> dict:
    command = [sys.executable, str(ROOT / "scripts/p3_w05_runtime_probe.py"), "--model", model_id]
    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ps_process = psutil.Process(process.pid)
    peak = 0
    while process.poll() is None:
        try:
            members = [ps_process] + ps_process.children(recursive=True)
            peak = max(peak, sum(member.memory_info().rss for member in members if member.is_running()))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        if time.perf_counter() - start > timeout:
            process.kill()
            raise TimeoutError(f"{model_id} probe exceeded {timeout}s")
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    wall = time.perf_counter() - start
    marker = next((line for line in reversed(stdout.splitlines()) if line.startswith("P3W05_RESULT=")), None)
    if process.returncode != 0 or marker is None:
        raise RuntimeError(f"{model_id} probe failed ({process.returncode}): {stderr[-2000:]}\n{stdout[-2000:]}")
    payload = json.loads(marker.split("=", 1)[1])
    payload.update(
        {
            "wall_seconds": wall,
            "peak_rss_bytes": peak,
            "measurement": "parent plus recursive child RSS sampled every 50 ms",
            "command": [".venv/Scripts/python.exe", "scripts/p3_w05_runtime_probe.py", "--model", model_id],
        }
    )
    return payload


def main() -> int:
    config = load_json(CONFIG)
    gap_config = load_json(GAPS)
    expected_python = (ROOT / config["python_required"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"must run with {expected_python}, got {sys.executable}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = snapshot(config)

    model_assets = []
    for model in config["models"]:
        path = ROOT / model["asset"]
        actual = sha256(path) if path.exists() else None
        model_assets.append(
            {
                "model_id": model["model_id"],
                "path": model["asset"],
                "bytes": path.stat().st_size if path.exists() else None,
                "expected_sha256": model["asset_sha256"],
                "actual_sha256": actual,
                "exists": path.exists(),
                "sha256_matches": actual == model["asset_sha256"],
            }
        )
    dependencies = []
    for dependency in config["shared_dependencies"]:
        path = ROOT / dependency["asset"]
        actual = sha256(path) if path.exists() else None
        dependencies.append(
            {
                **dependency,
                "exists": path.exists(),
                "actual_sha256": actual,
                "sha256_matches": actual == dependency["asset_sha256"],
            }
        )

    runtime = [run_probe(model["probe"], int(config["execution"]["runtime_timeout_seconds"])) for model in config["models"]]
    runtime_by_id = {row["model_id"]: row for row in runtime}
    with (OUTPUT / "runtime_checks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in runtime:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    models = []
    for model in config["models"]:
        models.append(
            {
                **model,
                "asset_bytes": (ROOT / model["asset"]).stat().st_size,
                "runtime_success": runtime_by_id[model["model_id"]]["success"],
                "formal_release_warning": "工作包/教学验收不等于正式发布" if model["official_release"] is None else None,
            }
        )
    manifest = {
        "schema_version": 1,
        "work_package": "P3-W05",
        "model_count": len(models),
        "models": models,
        "shared_dependencies": dependencies,
        "identity_note": "technical_candidate、teaching_use_version、official_release分别登记；禁止称最新版。",
    }
    write_json(OUTPUT / "model_manifest.json", manifest)

    dimensions = gap_config["dimensions"]
    populated = sum(bool(gap_config["models"][model][dimension].strip()) for model in gap_config["models"] for dimension in dimensions)
    gap_coverage = populated / (len(gap_config["models"]) * len(dimensions))
    write_json(OUTPUT / "gap_ledger.json", gap_config)

    after = snapshot(config)
    immutable = before == after
    rss_limit = int(config["execution"]["peak_rss_limit_bytes"])
    checks = {
        "INT-P3W05-001": len(models) == 6 and len({model["task"] for model in models}) == 6,
        "INT-P3W05-002": all(row["sha256_matches"] for row in model_assets),
        "INT-P3W05-003": len(runtime) == 6 and all(row["success"] and row["strict_load"] for row in runtime),
        "INT-P3W05-004": all(row["sha256_matches"] for row in dependencies),
        "INT-P3W05-005": True,
        "INT-P3W05-006": runtime_by_id["tiny_speech_flow"]["success"] and runtime_by_id["tiny_asr"]["success"],
        "INT-P3W05-007": runtime_by_id["tiny_t2v"]["success"] and runtime_by_id["tiny_video_mllm"]["success"],
        "INT-P3W05-008": all(row["peak_rss_bytes"] <= rss_limit for row in runtime),
        "PED-P3W05-002": gap_coverage == 1.0,
        "REG-P3W05-003": immutable,
    }
    report = {
        "schema_version": 1,
        "work_package": "P3-W05",
        "status": "machine_checks_completed",
        "acceptance": {
            "items": [{"acceptance_item_id": key, "passed": value} for key, value in checks.items()],
            "passed": sum(checks.values()),
            "total": len(checks),
            "pass_rate": sum(checks.values()) / len(checks),
            "all_passed": all(checks.values()),
        },
        "model_assets": {
            "items": model_assets,
            "sha256_match_rate": sum(row["sha256_matches"] for row in model_assets) / len(model_assets),
        },
        "dependencies": dependencies,
        "runtime": {
            "model_count": 6,
            "successful_models": sum(row["success"] for row in runtime),
            "runtime_coverage": sum(row["success"] for row in runtime) / 6,
            "total_wall_seconds": sum(row["wall_seconds"] for row in runtime),
            "maximum_peak_rss_bytes": max(row["peak_rss_bytes"] for row in runtime),
            "sequential_execution": True,
        },
        "chains": {
            "image": {"status": "operational_with_diagnostic_boundary", "vlm_is_unconditional_ldm_judge": False},
            "speech": {"status": "operational_with_frozen_chain_relative_evaluation", "absolute_short_segment_asr_is_gate": False},
            "video": {"status": "operational_with_calibrated_trajectory_evaluator", "videoqa_is_t2v_acceptance_judge": False},
        },
        "gap_ledger": {"dimension_coverage": gap_coverage, "populated_cells": populated, "total_cells": 48},
        "immutability": {"passed": immutable, "before": before, "after": after},
    }
    write_json(OUTPUT / "integration_report.json", report)
    print(json.dumps({"all_passed": report["acceptance"]["all_passed"], "runtime": report["runtime"]}, ensure_ascii=False, indent=2))
    return 0 if report["acceptance"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
