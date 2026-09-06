from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.iteration_management import (  # noqa: E402
    IterationManagementError,
    atomic_write_json,
    build_source_snapshot,
    iteration_status,
    list_iterations,
    load_json,
    prepare_iteration,
    promote_release,
    select_candidate,
    sha256_file,
    show_iteration,
    update_branch_state,
    verify_all,
)
from mini_multimodal.tools.build_p2_w04_iteration_catalog import (  # noqa: E402
    _assert_migration_rebuild_is_safe,
)
from mini_multimodal.tools.refresh_p2_w04_artifact_inventory import (  # noqa: E402
    inventory_diff,
)


REAL_RESULT = ROOT / "results/P2-W04_tiny_image_ldm"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _empty_management(root: Path, result: Path, *, work_package: str = "PX-W01") -> None:
    result.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result / "iteration_registry.json", {
        "schema_version": 1,
        "work_package": work_package,
        "last_completed_iteration": None,
        "current_candidate": None,
        "official_release": None,
        "iterations": [],
    })
    atomic_write_json(result / "current_candidate.json", {
        "schema_version": 1,
        "work_package": work_package,
        "current_candidate": None,
    })
    atomic_write_json(result / "official_release.json", {
        "schema_version": 1,
        "work_package": work_package,
        "official_release": None,
    })
    atomic_write_json(result / "artifact_inventory.json", {
        "schema_version": 1,
        "work_package": work_package,
        "scan_spec": [],
        "unclassified_count": 0,
        "files": [],
    })
    atomic_write_json(result / "releases/release_registry.json", {
        "schema_version": 1,
        "work_package": work_package,
        "releases": [],
    })


def _source_tree(root: Path) -> None:
    _write(root / "src/pkg/model.py", "VALUE = 1\n")
    _write(root / "tests/test_model.py", "def test_value(): assert 1 == 1\n")
    _write(root / "docs/design.md", "# Design\n")
    _write(root / "configs/model.json", '{"width": 8}\n')
    _write(root / "examples/demo.py", "print('demo')\n")
    _write(root / "requirements-lock.txt", "pytest==9.1.1\n")
    _write(root / ".venv/ignored.txt", "ignored\n")
    _write(root / "data/ignored.txt", "ignored\n")
    _write(root / "results/ignored.txt", "ignored\n")
    _write(root / "src/pkg/__pycache__/ignored.pyc", "ignored\n")


def test_reg_snap_001_source_snapshot_is_deterministic_and_excludes_runtime_data(
    tmp_path: Path,
) -> None:
    _source_tree(tmp_path)
    first = build_source_snapshot(tmp_path, tmp_path / "snap1.zip", tmp_path / "snap1.json")
    second = build_source_snapshot(tmp_path, tmp_path / "snap2.zip", tmp_path / "snap2.json")
    assert first["snapshot_sha256"] == second["snapshot_sha256"]
    assert first["files"] == second["files"]
    paths = {item["path"] for item in first["files"]}
    assert "src/pkg/model.py" in paths
    assert not any(".venv" in path or "data/" in path or "results/" in path for path in paths)
    assert not any("__pycache__" in path for path in paths)


def test_reg_iter_005_prepare_creates_standard_snapshot_and_rejects_overwrite(
    tmp_path: Path,
) -> None:
    _source_tree(tmp_path)
    result = tmp_path / "results/work_package"
    _empty_management(tmp_path, result)
    manifest = prepare_iteration(
        tmp_path,
        result,
        iteration_id="I001",
        title="first",
        objective="verify preparation",
        parent_iteration=None,
        config_paths=["configs/model.json"],
        created_at="2026-08-09",
        authorized_by="teacher confirmation fixture",
        branch_ids=["control", "geometry_lite"],
    )
    directory = result / "iterations/I001"
    assert manifest["execution_state"] == "planned"
    assert (directory / "source_snapshot.zip").is_file()
    assert (directory / "source_manifest.json").is_file()
    assert (directory / "configs/model.json").is_file()
    for branch_id in ("control", "geometry_lite"):
        branch = directory / "branches" / branch_id
        assert (branch / "branch_manifest.json").is_file()
        assert (branch / "README.md").is_file()
        for name in ("checkpoints", "metrics", "predictions", "examples", "logs", "tests"):
            assert (branch / name).is_dir()
    assert list_iterations(result)[0]["branch_states"] == {
        "control": "planned",
        "geometry_lite": "planned",
    }
    assert list_iterations(result)[0]["iteration_id"] == "I001"
    with pytest.raises(IterationManagementError, match="already exists"):
        prepare_iteration(
            tmp_path,
            result,
            iteration_id="I001",
            title="duplicate",
            objective="must fail",
            parent_iteration=None,
            config_paths=["configs/model.json"],
            created_at="2026-08-09",
            authorized_by="fixture",
        )
    with pytest.raises(IterationManagementError, match="next iteration must be I002"):
        prepare_iteration(
            tmp_path,
            result,
            iteration_id="I003",
            title="gap",
            objective="must fail",
            parent_iteration="I001",
            config_paths=["configs/model.json"],
            created_at="2026-08-09",
            authorized_by="fixture",
        )


def test_reg_iter_007_branch_progress_is_explicit_and_terminal_states_are_locked(
    tmp_path: Path,
) -> None:
    _source_tree(tmp_path)
    result = tmp_path / "results/work_package"
    _empty_management(tmp_path, result)
    prepare_iteration(
        tmp_path,
        result,
        iteration_id="I001",
        title="progress fixture",
        objective="verify explicit branch progress",
        parent_iteration=None,
        config_paths=["configs/model.json"],
        created_at="2026-08-09T09:00:00+08:00",
        authorized_by="teacher confirmation fixture",
        branch_ids=["control", "geometry"],
    )
    first = update_branch_state(
        result,
        iteration_id="I001",
        branch_id="control",
        new_state="running",
        updated_at="2026-08-09T10:00:00+08:00",
        record="training started",
    )
    assert first["iteration_state"] == "running"
    with pytest.raises(IterationManagementError, match="lacks required artifact"):
        update_branch_state(
            result,
            iteration_id="I001",
            branch_id="control",
            new_state="completed",
            updated_at="2026-08-09T10:30:00+08:00",
            record="must fail before evidence exists",
        )
    control_dir = result / "iterations/I001/branches/control"
    for directory, name in {
        "checkpoints": "best.pt",
        "metrics": "metrics.json",
        "predictions": "predictions.jsonl",
        "examples": "capability_examples.md",
        "tests": "pytest.xml",
    }.items():
        _write(control_dir / directory / name, f"{directory} evidence\n")
    update_branch_state(
        result,
        iteration_id="I001",
        branch_id="control",
        new_state="completed",
        updated_at="2026-08-09T11:00:00+08:00",
        record="evaluation and artifacts completed",
    )
    update_branch_state(
        result,
        iteration_id="I001",
        branch_id="geometry",
        new_state="running",
        updated_at="2026-08-09T10:05:00+08:00",
        record="training started",
    )
    final = update_branch_state(
        result,
        iteration_id="I001",
        branch_id="geometry",
        new_state="aborted",
        updated_at="2026-08-09T11:10:00+08:00",
        record="predefined resource stop condition reached",
    )
    assert final["iteration_state"] == "completed"
    assert final["last_completed_iteration"] == "I001"
    dashboard = iteration_status(result)
    assert dashboard["state_counts"]["completed"] == 1
    assert dashboard["iterations"][0]["branch_progress"] == {"terminal": 2, "total": 2}
    with pytest.raises(IterationManagementError, match="invalid branch state transition"):
        update_branch_state(
            result,
            iteration_id="I001",
            branch_id="control",
            new_state="running",
            updated_at="2026-08-09T12:00:00+08:00",
            record="must not reopen terminal evidence",
        )
    assert verify_all(tmp_path, result, verify_files=True)["verified"]
    _write(control_dir / "metrics/metrics.json", "changed after completion\n")
    changed = verify_all(tmp_path, result, verify_files=True)
    assert not changed["verified"]
    assert any("branch artifact" in error for error in changed["errors"])


def test_reg_iter_008_history_catalog_cannot_erase_future_iterations(tmp_path: Path) -> None:
    result = tmp_path / "results/work_package"
    _empty_management(tmp_path, result)
    registry = load_json(result / "iteration_registry.json")
    registry["iterations"] = [{"iteration_id": "I005"}]
    atomic_write_json(result / "iteration_registry.json", registry)
    (result / "iterations/I005").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="future iteration"):
        _assert_migration_rebuild_is_safe(result)


def test_reg_iter_009_inventory_refresh_diff_is_independent_from_registry_state() -> None:
    before = {
        "files": [
            {"path": "a.txt", "bytes": 1, "sha256": "old", "owner": "shared", "scope": "docs"},
            {"path": "missing.txt", "bytes": 2, "sha256": "same", "owner": "shared", "scope": "docs"},
        ]
    }
    after = {
        "files": [
            {"path": "a.txt", "bytes": 1, "sha256": "new", "owner": "shared", "scope": "docs"},
            {"path": "added.txt", "bytes": 3, "sha256": "x", "owner": "shared", "scope": "docs"},
        ]
    }
    assert inventory_diff(before, after) == {
        "added": ["added.txt"],
        "missing": ["missing.txt"],
        "changed": ["a.txt"],
        "unchanged_count": 0,
    }


def _completed_iteration(root: Path, result: Path, *, accepted: bool, eligible: bool) -> None:
    directory = result / "iterations/I001"
    directory.mkdir(parents=True, exist_ok=True)
    roles = {
        "inference_package": "package.pt",
        "model_card": "model_card.md",
        "metrics": "metrics.json",
        "capability_examples": "capability_examples.md",
        "usage": "usage.md",
    }
    release_artifacts = []
    for role, name in roles.items():
        source = root / "artifacts" / name
        _write(source, f"{role}\n")
        release_artifacts.append({
            "branch_id": "main",
            "role": role,
            "source_path": source.relative_to(root).as_posix(),
            "destination": name,
            "sha256": sha256_file(source),
        })
    manifest = {
        "schema_version": 1,
        "work_package": "PX-W01",
        "iteration_id": "I001",
        "title": "completed",
        "objective": "release fixture",
        "parent_iteration": None,
        "execution_state": "completed",
        "selection_decision": "undecided",
        "historical_registration": False,
        "all_blocking_acceptance_items_passed": accepted,
        "branches": [{"branch_id": "main", "release_eligible": eligible}],
        "source_reproducibility": {"state": "historical_snapshot_unavailable"},
        "artifact_references": [],
        "release_artifacts": release_artifacts,
    }
    atomic_write_json(directory / "iteration_manifest.json", manifest)
    registry = load_json(result / "iteration_registry.json")
    registry["last_completed_iteration"] = "I001"
    registry["iterations"] = [{
        "iteration_id": "I001",
        "title": "completed",
        "parent_iteration": None,
        "manifest": "iterations/I001/iteration_manifest.json",
        "summary": "iterations/I001/summary.md",
        "branches": ["main"],
        "execution_state": "completed",
        "selection_decision": "undecided",
        "all_blocking_acceptance_items_passed": accepted,
    }]
    atomic_write_json(result / "iteration_registry.json", registry)


def test_reg_iter_004_candidate_is_explicit_and_not_chronological(tmp_path: Path) -> None:
    result = tmp_path / "results/work_package"
    _empty_management(tmp_path, result)
    _completed_iteration(tmp_path, result, accepted=False, eligible=False)
    pointer = select_candidate(
        result,
        iteration_id="I001",
        branch_id="main",
        decision_record="teacher selected it only as next research base",
        selected_at="2026-08-09",
    )
    assert pointer["current_candidate"]["iteration_id"] == "I001"
    assert load_json(result / "official_release.json")["official_release"] is None
    assert load_json(result / "iteration_registry.json")["official_release"] is None


def test_reg_rel_001_release_rejects_missing_acceptance_and_confirmation(tmp_path: Path) -> None:
    result = tmp_path / "results/work_package"
    _empty_management(tmp_path, result)
    _completed_iteration(tmp_path, result, accepted=False, eligible=False)
    select_candidate(
        result,
        iteration_id="I001",
        branch_id="main",
        decision_record="fixture candidate",
        selected_at="2026-08-09",
    )
    with pytest.raises(IterationManagementError, match="teacher confirmation"):
        promote_release(
            tmp_path,
            result,
            iteration_id="I001",
            branch_id="main",
            release_id="REL001",
            approval_record="",
            released_at="2026-08-09",
        )
    with pytest.raises(IterationManagementError, match="blocking acceptance"):
        promote_release(
            tmp_path,
            result,
            iteration_id="I001",
            branch_id="main",
            release_id="REL001",
            approval_record="teacher fixture",
            released_at="2026-08-09",
        )
    assert not (result / "releases/REL001").exists()
    assert load_json(result / "official_release.json")["official_release"] is None


def test_reg_rel_002_valid_release_is_materialized_once_and_verified(tmp_path: Path) -> None:
    result = tmp_path / "results/work_package"
    _empty_management(tmp_path, result)
    _completed_iteration(tmp_path, result, accepted=True, eligible=True)
    select_candidate(
        result,
        iteration_id="I001",
        branch_id="main",
        decision_record="fixture selected after all tests",
        selected_at="2026-08-09",
    )
    release = promote_release(
        tmp_path,
        result,
        iteration_id="I001",
        branch_id="main",
        release_id="REL001",
        approval_record="teacher explicit fixture confirmation",
        released_at="2026-08-09",
    )
    assert release["release_id"] == "REL001"
    assert len(release["artifacts"]) == 5
    for item in release["artifacts"]:
        path = result / "releases/REL001" / item["path"]
        assert path.is_file()
        assert sha256_file(path) == item["sha256"]
    assert load_json(result / "official_release.json")["official_release"]["release_id"] == "REL001"
    with pytest.raises(IterationManagementError, match="already exists|already registered"):
        promote_release(
            tmp_path,
            result,
            iteration_id="I001",
            branch_id="main",
            release_id="REL001",
            approval_record="teacher explicit fixture confirmation",
            released_at="2026-08-09",
        )


def test_reg_iter_001_real_registry_keeps_four_historical_iterations_and_formal_i005() -> None:
    rows = list_iterations(REAL_RESULT)
    assert [item["iteration_id"] for item in rows] == ["I001", "I002", "I003", "I004", "I005"]
    assert rows[3]["branches"] == ["control", "geometry", "geometry_lite"]
    assert rows[4]["branches"] == ["continuation", "clean_refiner"]
    assert rows[4]["execution_state"] in {"planned", "running", "completed", "aborted"}
    registry = load_json(REAL_RESULT / "iteration_registry.json")
    assert registry["last_completed_iteration"] == "I004"
    assert registry["current_candidate"]["iteration_id"] == "I004"
    assert registry["current_candidate"]["branch_id"] == "control"
    assert registry["official_release"] is None
    assert all(not item["all_blocking_acceptance_items_passed"] for item in rows)


def test_reg_iter_003_real_historical_manifests_are_traceable_without_fake_snapshots() -> None:
    for iteration_id in ("I001", "I002", "I003", "I004"):
        manifest = show_iteration(REAL_RESULT, iteration_id)
        assert manifest["historical_registration"] is True
        assert manifest["source_reproducibility"]["state"] == "historical_snapshot_unavailable"
        assert manifest["artifact_references"]
        for item in manifest["artifact_references"]:
            path = ROOT / item["path"]
            assert path.stat().st_size == item["bytes"]
            assert sha256_file(path) == item["sha256"]


def test_reg_iter_002_real_inventory_and_all_pointers_verify() -> None:
    report = verify_all(ROOT, REAL_RESULT, verify_files=True)
    assert report["verified"], report["errors"]
    assert report["inventory"]["file_count"] == report["inventory"]["current_scan_file_count"]
    assert report["inventory"]["unregistered_current_files"] == []
    assert report["inventory"]["missing_from_current_scan"] == []
    assert report["current_candidate"]["iteration_id"] == "I004"
    assert report["current_candidate"]["branch_id"] == "control"
    assert report["official_release"] is None


def test_reg_iter_006_unknown_iteration_is_rejected() -> None:
    with pytest.raises(IterationManagementError, match="unknown iteration"):
        show_iteration(REAL_RESULT, "I999")
