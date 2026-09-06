"""File-backed model iteration and release management.

The registry is deliberately independent from Git.  Completed iteration
evidence is immutable; candidate selection and official releases are stored in
separate pointer files so chronological order can never imply finality.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
import uuid
import zipfile
from typing import Any, Iterable, Mapping, Sequence


ITERATION_ID = re.compile(r"^I\d{3}$")
RELEASE_ID = re.compile(r"^REL\d{3}$")
BRANCH_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
EXECUTION_STATES = {"planned", "running", "completed", "aborted"}
BRANCH_DIRECTORY_NAMES = (
    "checkpoints",
    "metrics",
    "predictions",
    "examples",
    "logs",
    "tests",
)
REQUIRED_COMPLETION_DIRECTORIES = (
    "checkpoints",
    "metrics",
    "predictions",
    "examples",
    "tests",
)
SELECTION_DECISIONS = {
    "undecided",
    "not_selected",
    "superseded",
    "current_candidate",
    "released",
}
SNAPSHOT_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "data",
    "results",
}
DEFAULT_SNAPSHOT_PATHS = (
    "src",
    "tests",
    "docs",
    "configs",
    "examples",
    "AGENTS.md",
    "最小系统研发规格与约束.md",
    "requirements.txt",
    "requirements-torch-cpu.txt",
    "requirements-lock.txt",
)


class IterationManagementError(RuntimeError):
    """Raised when an iteration or release safety invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_json_bytes(value))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_reference(project_root: Path, path: Path, *, role: str) -> dict[str, Any]:
    resolved_root = project_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise IterationManagementError(f"artifact outside project root: {path}") from error
    if not resolved.is_file():
        raise IterationManagementError(f"artifact is not a file: {relative}")
    return {
        "path": relative,
        "role": role,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_artifact_reference(project_root: Path, item: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    relative = item.get("path")
    if not isinstance(relative, str) or not relative:
        return ["artifact reference has no path"]
    path = project_root / relative
    if not path.is_file():
        return [f"missing artifact: {relative}"]
    if int(item.get("bytes", -1)) != path.stat().st_size:
        errors.append(f"artifact size differs: {relative}")
    if item.get("sha256") != sha256_file(path):
        errors.append(f"artifact digest differs: {relative}")
    return errors


def _snapshot_files(project_root: Path, includes: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for relative in includes:
        source = project_root / relative
        if not source.exists():
            continue
        candidates = [source] if source.is_file() else source.rglob("*")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            rel = candidate.relative_to(project_root)
            if any(part in SNAPSHOT_EXCLUDED_PARTS for part in rel.parts):
                continue
            if candidate.suffix in {".pyc", ".pyo"}:
                continue
            files.add(candidate)
    return sorted(files, key=lambda item: item.relative_to(project_root).as_posix())


def build_source_snapshot(
    project_root: Path,
    zip_path: Path,
    manifest_path: Path,
    *,
    includes: Sequence[str] = DEFAULT_SNAPSHOT_PATHS,
) -> dict[str, Any]:
    """Create a deterministic ZIP and a separately verifiable file manifest."""

    project_root = project_root.resolve()
    files = _snapshot_files(project_root, includes)
    if not files:
        raise IterationManagementError("source snapshot would be empty")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_name(f".{zip_path.name}.{uuid.uuid4().hex}.tmp")
    entries: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in files:
                relative = path.relative_to(project_root).as_posix()
                data = path.read_bytes()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
                entries.append({
                    "path": relative,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                })
        os.replace(temporary, zip_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "schema_version": 1,
        "snapshot_policy": "deterministic_zip_fixed_timestamp_sorted_paths",
        "included_roots": list(includes),
        "excluded_parts": sorted(SNAPSHOT_EXCLUDED_PARTS),
        "file_count": len(entries),
        "total_source_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
        "snapshot_path": zip_path.name,
        "snapshot_bytes": zip_path.stat().st_size,
        "snapshot_sha256": sha256_file(zip_path),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_registry(result_root: Path) -> dict[str, Any]:
    path = result_root / "iteration_registry.json"
    if not path.is_file():
        raise IterationManagementError(f"missing iteration registry: {path}")
    return load_json(path)


def list_iterations(result_root: Path) -> list[dict[str, Any]]:
    registry = load_registry(result_root)
    return [
        {
            "iteration_id": item["iteration_id"],
            "title": item["title"],
            "parent_iteration": item.get("parent_iteration"),
            "branches": list(item.get("branches", [])),
            "branch_states": dict(item.get("branch_states", {})),
            "execution_state": item["execution_state"],
            "selection_decision": item["selection_decision"],
            "all_blocking_acceptance_items_passed": bool(
                item.get("all_blocking_acceptance_items_passed", False)
            ),
        }
        for item in registry["iterations"]
    ]


def show_iteration(result_root: Path, iteration_id: str) -> dict[str, Any]:
    if not ITERATION_ID.fullmatch(iteration_id):
        raise IterationManagementError(f"invalid iteration id: {iteration_id}")
    path = result_root / "iterations" / iteration_id / "iteration_manifest.json"
    if not path.is_file():
        raise IterationManagementError(f"unknown iteration: {iteration_id}")
    return load_json(path)


def iteration_status(result_root: Path, iteration_id: str | None = None) -> dict[str, Any]:
    registry = load_registry(result_root)
    rows = list_iterations(result_root)
    if iteration_id is not None:
        rows = [item for item in rows if item["iteration_id"] == iteration_id]
        if not rows:
            raise IterationManagementError(f"unknown iteration: {iteration_id}")
    state_counts = {state: 0 for state in sorted(EXECUTION_STATES)}
    detailed: list[dict[str, Any]] = []
    for row in rows:
        state_counts[row["execution_state"]] += 1
        manifest = show_iteration(result_root, row["iteration_id"])
        branch_rows = []
        for branch in manifest.get("branches", []):
            branch_state = branch.get("execution_state")
            if branch_state is None and manifest.get("historical_registration"):
                branch_state = manifest.get("execution_state")
            branch_rows.append({
                "branch_id": branch["branch_id"],
                "execution_state": branch_state,
                "release_eligible": bool(branch.get("release_eligible", False)),
            })
        terminal_count = sum(
            item["execution_state"] in {"completed", "aborted"} for item in branch_rows
        )
        detailed.append({
            **row,
            "branch_states": {
                item["branch_id"]: item["execution_state"] for item in branch_rows
            },
            "branch_progress": {
                "terminal": terminal_count,
                "total": len(branch_rows),
            },
            "branch_details": branch_rows,
        })
    return {
        "work_package": registry.get("work_package"),
        "last_completed_iteration": registry.get("last_completed_iteration"),
        "current_candidate": registry.get("current_candidate"),
        "official_release": registry.get("official_release"),
        "state_counts": state_counts,
        "iterations": detailed,
    }


def verify_iteration(project_root: Path, result_root: Path, iteration_id: str) -> dict[str, Any]:
    errors: list[str] = []
    try:
        manifest = show_iteration(result_root, iteration_id)
    except (IterationManagementError, OSError, json.JSONDecodeError) as error:
        return {"iteration_id": iteration_id, "verified": False, "errors": [str(error)]}
    if manifest.get("iteration_id") != iteration_id:
        errors.append("iteration manifest id differs from directory")
    if manifest.get("execution_state") not in EXECUTION_STATES:
        errors.append(f"invalid execution_state: {manifest.get('execution_state')}")
    if manifest.get("selection_decision") not in SELECTION_DECISIONS:
        errors.append(f"invalid selection_decision: {manifest.get('selection_decision')}")
    branches = manifest.get("branches", [])
    branch_ids = [item.get("branch_id") for item in branches]
    if not branches or len(branch_ids) != len(set(branch_ids)) or any(not value for value in branch_ids):
        errors.append("branches are empty, duplicated, or missing branch_id")
    if not manifest.get("historical_registration", False):
        iteration_dir = result_root / "iterations" / iteration_id
        for branch in branches:
            branch_id = branch.get("branch_id")
            if not isinstance(branch_id, str) or not BRANCH_ID.fullmatch(branch_id):
                errors.append(f"invalid branch_id: {branch_id}")
                continue
            if branch.get("execution_state") not in EXECUTION_STATES:
                errors.append(f"invalid branch execution_state: {branch_id}")
            branch_dir = iteration_dir / "branches" / branch_id
            for directory_name in BRANCH_DIRECTORY_NAMES:
                if not (branch_dir / directory_name).is_dir():
                    errors.append(f"missing branch directory: {branch_id}/{directory_name}")
            branch_manifest_path = branch_dir / "branch_manifest.json"
            if not branch_manifest_path.is_file():
                errors.append(f"missing branch manifest: {branch_id}")
            else:
                branch_manifest = load_json(branch_manifest_path)
                if branch_manifest.get("branch_id") != branch_id:
                    errors.append(f"branch manifest id differs: {branch_id}")
                if branch_manifest.get("execution_state") != branch.get("execution_state"):
                    errors.append(f"branch state differs from iteration manifest: {branch_id}")
                if branch.get("execution_state") == "completed":
                    artifacts = branch_manifest.get("artifact_inventory", [])
                    if not artifacts:
                        errors.append(f"completed branch has no artifact inventory: {branch_id}")
                    for artifact in artifacts:
                        relative = PurePosixPath(str(artifact.get("path", "")))
                        artifact_path = branch_dir / relative
                        if relative.is_absolute() or ".." in relative.parts or not artifact_path.is_file():
                            errors.append(f"invalid completed branch artifact: {branch_id}/{relative}")
                            continue
                        if artifact.get("bytes") != artifact_path.stat().st_size:
                            errors.append(f"branch artifact size differs: {branch_id}/{relative}")
                        if artifact.get("sha256") != sha256_file(artifact_path):
                            errors.append(f"branch artifact digest differs: {branch_id}/{relative}")
    for item in manifest.get("artifact_references", []):
        errors.extend(verify_artifact_reference(project_root, item))
    source = manifest.get("source_reproducibility", {})
    source_state = source.get("state")
    if source_state == "snapshot_available":
        for relative in (source.get("snapshot_path"), source.get("manifest_path")):
            if not relative or not (project_root / relative).is_file():
                errors.append(f"missing source snapshot artifact: {relative}")
        if not errors and source.get("snapshot_sha256") != sha256_file(
            project_root / source["snapshot_path"]
        ):
            errors.append("source snapshot digest differs")
    elif source_state != "historical_snapshot_unavailable":
        errors.append(f"invalid source reproducibility state: {source_state}")
    return {"iteration_id": iteration_id, "verified": not errors, "errors": errors}


def verify_inventory(project_root: Path, inventory_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    inventory = load_json(inventory_path)
    entries = inventory.get("files", [])
    paths = [item.get("path") for item in entries]
    if len(paths) != len(set(paths)):
        errors.append("inventory contains duplicate paths")
    if int(inventory.get("unclassified_count", -1)) != 0:
        errors.append("inventory has unclassified files")
    for item in entries:
        errors.extend(verify_artifact_reference(project_root, item))
        if not item.get("scope") or not item.get("owner"):
            errors.append(f"inventory entry lacks scope/owner: {item.get('path')}")
    current_paths = _scan_inventory_spec(project_root, inventory.get("scan_spec", []))
    registered_paths = {str(value) for value in paths}
    unregistered = sorted(current_paths - registered_paths)
    missing_from_scan = sorted(registered_paths - current_paths)
    if unregistered:
        errors.append(f"inventory has {len(unregistered)} unregistered current files: {unregistered[:5]}")
    if missing_from_scan:
        errors.append(
            f"inventory has {len(missing_from_scan)} paths outside current scan: {missing_from_scan[:5]}"
        )
    return {
        "file_count": len(entries),
        "current_scan_file_count": len(current_paths),
        "unregistered_current_files": unregistered,
        "missing_from_current_scan": missing_from_scan,
        "verified": not errors,
        "errors": errors,
    }


def _scan_inventory_spec(project_root: Path, spec: Sequence[Mapping[str, Any]]) -> set[str]:
    files: set[str] = set()
    for item in spec:
        base = project_root / str(item["base"])
        mode = item.get("mode", "recursive")
        pattern = str(item.get("pattern", "*"))
        if mode == "exact":
            candidates: Iterable[Path] = [base]
        elif mode == "glob":
            candidates = base.glob(pattern) if base.is_dir() else []
        elif mode == "recursive":
            candidates = base.rglob(pattern) if base.is_dir() else []
        else:
            raise IterationManagementError(f"unknown inventory scan mode: {mode}")
        exclude_prefixes = [PurePosixPath(value) for value in item.get("exclude_prefixes", [])]
        exclude_names = set(item.get("exclude_names", []))
        for path in candidates:
            if not path.is_file():
                continue
            try:
                relative_to_base = PurePosixPath(path.relative_to(base).as_posix())
            except ValueError:
                relative_to_base = PurePosixPath(path.name)
            if path.name in exclude_names:
                continue
            if any(
                relative_to_base == prefix or relative_to_base.is_relative_to(prefix)
                for prefix in exclude_prefixes
            ):
                continue
            if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            files.add(path.relative_to(project_root).as_posix())
    return files


def verify_all(project_root: Path, result_root: Path, *, verify_files: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    registry = load_registry(result_root)
    items = registry.get("iterations", [])
    ids = [item.get("iteration_id") for item in items]
    if len(ids) != len(set(ids)) or any(not ITERATION_ID.fullmatch(str(value)) for value in ids):
        errors.append("iteration ids are invalid or duplicated")
    if ids != sorted(ids):
        errors.append("iteration registry is not ordered")
    known = set(ids)
    for item in items:
        parent = item.get("parent_iteration")
        if parent is not None and parent not in known:
            errors.append(f"unknown parent {parent} for {item.get('iteration_id')}")
        if item.get("execution_state") not in EXECUTION_STATES:
            errors.append(f"invalid execution state for {item.get('iteration_id')}")
        if item.get("selection_decision") not in SELECTION_DECISIONS:
            errors.append(f"invalid selection decision for {item.get('iteration_id')}")
    completed = [item["iteration_id"] for item in items if item["execution_state"] == "completed"]
    expected_last = completed[-1] if completed else None
    if registry.get("last_completed_iteration") != expected_last:
        errors.append("last_completed_iteration differs from ordered completed iterations")

    iteration_results = [
        verify_iteration(project_root, result_root, iteration_id) for iteration_id in ids
    ]
    errors.extend(
        f"{item['iteration_id']}: {message}"
        for item in iteration_results
        for message in item["errors"]
    )

    candidate = load_json(result_root / "current_candidate.json")
    official = load_json(result_root / "official_release.json")
    release_registry = load_json(result_root / "releases" / "release_registry.json")
    if registry.get("current_candidate") != candidate.get("current_candidate"):
        errors.append("registry candidate differs from current_candidate.json")
    if registry.get("official_release") != official.get("official_release"):
        errors.append("registry release differs from official_release.json")
    pointer = candidate.get("current_candidate")
    selected = [item for item in items if item.get("selection_decision") == "current_candidate"]
    if pointer is None:
        if selected:
            errors.append("registry has selected candidate but pointer is null")
    else:
        if pointer.get("iteration_id") not in known:
            errors.append("candidate points to unknown iteration")
        if len(selected) != 1 or selected[0]["iteration_id"] != pointer.get("iteration_id"):
            errors.append("candidate pointer and registry selection differ")
    official_pointer = official.get("official_release")
    releases = release_registry.get("releases", [])
    if official_pointer is None and registry.get("official_release") is not None:
        errors.append("official release pointer is inconsistent")
    if official_pointer is not None:
        release_id = official_pointer.get("release_id")
        if not any(item.get("release_id") == release_id for item in releases):
            errors.append("official release is absent from release registry")
        if not (result_root / "releases" / str(release_id) / "release_manifest.json").is_file():
            errors.append("official release directory is incomplete")

    inventory_result: dict[str, Any] | None = None
    inventory_path = result_root / "artifact_inventory.json"
    if verify_files:
        inventory_result = verify_inventory(project_root, inventory_path)
        errors.extend(f"inventory: {message}" for message in inventory_result["errors"])
    return {
        "work_package": registry.get("work_package"),
        "iteration_count": len(items),
        "last_completed_iteration": registry.get("last_completed_iteration"),
        "current_candidate": candidate.get("current_candidate"),
        "official_release": official.get("official_release"),
        "iterations": iteration_results,
        "inventory": inventory_result,
        "verified": not errors,
        "errors": errors,
    }


def _copy_configs(project_root: Path, iteration_dir: Path, config_paths: Sequence[str]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    destination = iteration_dir / "configs"
    destination.mkdir(parents=True, exist_ok=True)
    for relative in config_paths:
        source = project_root / relative
        if not source.is_file():
            raise IterationManagementError(f"missing iteration config: {relative}")
        target = destination / source.name
        if target.exists():
            raise IterationManagementError(f"duplicate config basename: {source.name}")
        shutil.copy2(source, target)
        copied.append({
            "source_path": relative,
            "snapshot_path": target.name,
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    return copied


def prepare_iteration(
    project_root: Path,
    result_root: Path,
    *,
    iteration_id: str,
    title: str,
    objective: str,
    parent_iteration: str | None,
    config_paths: Sequence[str],
    created_at: str,
    authorized_by: str,
    branch_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not ITERATION_ID.fullmatch(iteration_id):
        raise IterationManagementError(f"invalid iteration id: {iteration_id}")
    if not title.strip() or not objective.strip() or not authorized_by.strip():
        raise IterationManagementError("title, objective, and authorization are required")
    branch_ids = list(branch_ids or ["main"])
    if (
        not branch_ids
        or len(branch_ids) != len(set(branch_ids))
        or any(not BRANCH_ID.fullmatch(value) for value in branch_ids)
    ):
        raise IterationManagementError(
            "branch ids must be unique lowercase identifiers matching "
            "[a-z][a-z0-9_-]{0,63}"
        )
    registry = load_registry(result_root)
    known = {item["iteration_id"] for item in registry["iterations"]}
    if iteration_id in known or (result_root / "iterations" / iteration_id).exists():
        raise IterationManagementError(f"iteration already exists: {iteration_id}")
    if parent_iteration is not None and parent_iteration not in known:
        raise IterationManagementError(f"unknown parent iteration: {parent_iteration}")
    expected = f"I{len(registry['iterations']) + 1:03d}"
    if iteration_id != expected:
        raise IterationManagementError(f"next iteration must be {expected}, not {iteration_id}")

    iterations_dir = result_root / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)
    staging = iterations_dir / f".staging-{iteration_id}-{uuid.uuid4().hex}"
    final = iterations_dir / iteration_id
    try:
        for relative in ("branches", "comparisons", "tests", "environment"):
            (staging / relative).mkdir(parents=True, exist_ok=True)
        branches: list[dict[str, Any]] = []
        for branch_id in branch_ids:
            branch_dir = staging / "branches" / branch_id
            for relative in BRANCH_DIRECTORY_NAMES:
                (branch_dir / relative).mkdir(parents=True, exist_ok=True)
            branch_manifest = {
                "schema_version": 1,
                "iteration_id": iteration_id,
                "branch_id": branch_id,
                "execution_state": "planned",
                "release_eligible": False,
                "status_history": [{
                    "state": "planned",
                    "updated_at": created_at,
                    "record": "created by prepare",
                }],
            }
            atomic_write_json(branch_dir / "branch_manifest.json", branch_manifest)
            atomic_write_text(
                branch_dir / "README.md",
                f"# {iteration_id}/{branch_id}\n\n"
                "> 状态：planned\n\n"
                "本目录只保存该迭代分支的检查点、指标、逐样本结果、"
                "能力样例、日志和测试证据。\n",
            )
            branches.append({
                "branch_id": branch_id,
                "execution_state": "planned",
                "release_eligible": False,
                "manifest": f"branches/{branch_id}/branch_manifest.json",
            })
        source_manifest = build_source_snapshot(
            project_root,
            staging / "source_snapshot.zip",
            staging / "source_manifest.json",
        )
        configs = _copy_configs(project_root, staging, config_paths)
        runtime = {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        }
        atomic_write_json(staging / "environment" / "runtime_info.json", runtime)
        manifest = {
            "schema_version": 1,
            "work_package": registry["work_package"],
            "iteration_id": iteration_id,
            "title": title,
            "objective": objective,
            "parent_iteration": parent_iteration,
            "created_at": created_at,
            "authorization_record": authorized_by,
            "execution_state": "planned",
            "selection_decision": "undecided",
            "historical_registration": False,
            "all_blocking_acceptance_items_passed": False,
            "branches": branches,
            "status_history": [{
                "state": "planned",
                "updated_at": created_at,
                "record": "iteration prepared after authorization",
            }],
            "source_reproducibility": {
                "state": "snapshot_available",
                "snapshot_path": f"{result_root.relative_to(project_root).as_posix()}/iterations/{iteration_id}/source_snapshot.zip",
                "manifest_path": f"{result_root.relative_to(project_root).as_posix()}/iterations/{iteration_id}/source_manifest.json",
                "snapshot_sha256": source_manifest["snapshot_sha256"],
            },
            "config_snapshots": configs,
            "artifact_references": [],
            "release_artifacts": [],
        }
        atomic_write_json(staging / "iteration_manifest.json", manifest)
        atomic_write_text(
            staging / "summary.md",
            f"# {iteration_id}：{title}\n\n> 状态：planned\n\n"
            f"{objective}\n\n分支：{', '.join(branch_ids)}\n",
        )
        os.replace(staging, final)
        entry = {
            "iteration_id": iteration_id,
            "title": title,
            "parent_iteration": parent_iteration,
            "manifest": f"iterations/{iteration_id}/iteration_manifest.json",
            "summary": f"iterations/{iteration_id}/summary.md",
            "branches": branch_ids,
            "branch_states": {branch_id: "planned" for branch_id in branch_ids},
            "execution_state": "planned",
            "selection_decision": "undecided",
            "all_blocking_acceptance_items_passed": False,
        }
        registry["iterations"].append(entry)
        atomic_write_json(result_root / "iteration_registry.json", registry)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if final.exists() and not any(
            item.get("iteration_id") == iteration_id for item in registry.get("iterations", [])
        ):
            shutil.rmtree(final)
        raise


def update_branch_state(
    result_root: Path,
    *,
    iteration_id: str,
    branch_id: str,
    new_state: str,
    updated_at: str,
    record: str,
) -> dict[str, Any]:
    """Advance one branch and derive the iteration's overall progress state."""

    if new_state not in EXECUTION_STATES:
        raise IterationManagementError(f"invalid execution state: {new_state}")
    if not updated_at.strip() or not record.strip():
        raise IterationManagementError("updated_at and progress record are required")
    registry = load_registry(result_root)
    entries = [item for item in registry["iterations"] if item["iteration_id"] == iteration_id]
    if not entries:
        raise IterationManagementError(f"unknown iteration: {iteration_id}")
    entry = entries[0]
    manifest_path = result_root / entry["manifest"]
    manifest = load_json(manifest_path)
    if manifest.get("historical_registration"):
        raise IterationManagementError("historical iterations are immutable")
    branches = manifest.get("branches", [])
    matches = [item for item in branches if item.get("branch_id") == branch_id]
    if not matches:
        raise IterationManagementError(f"unknown branch: {branch_id}")
    branch = matches[0]
    current = branch.get("execution_state")
    allowed = {
        "planned": {"running", "aborted"},
        "running": {"completed", "aborted"},
        "completed": set(),
        "aborted": set(),
    }
    if new_state not in allowed.get(str(current), set()):
        raise IterationManagementError(
            f"invalid branch state transition: {current} -> {new_state}"
        )

    branch_manifest_path = (
        result_root / "iterations" / iteration_id / "branches" / branch_id / "branch_manifest.json"
    )
    branch_manifest = load_json(branch_manifest_path)
    branch_dir = branch_manifest_path.parent
    if new_state == "completed":
        missing_roles = [
            name
            for name in REQUIRED_COMPLETION_DIRECTORIES
            if not any(path.is_file() for path in (branch_dir / name).rglob("*"))
        ]
        if missing_roles:
            raise IterationManagementError(
                "completed branch lacks required artifact directories: "
                + ", ".join(missing_roles)
            )
        artifact_inventory = []
        for path in sorted(branch_dir.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file() or path.name in {"branch_manifest.json", "README.md"}:
                continue
            artifact_inventory.append({
                "path": path.relative_to(branch_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        branch_manifest["artifact_inventory"] = artifact_inventory
        acceptance_path = branch_dir / "metrics" / "acceptance_report.json"
        if acceptance_path.is_file():
            acceptance = load_json(acceptance_path)
            release_eligible = bool(
                acceptance.get("all_blocking_acceptance_items_passed", False)
            )
            branch_manifest["release_eligible"] = release_eligible
            branch["release_eligible"] = release_eligible
    event = {"state": new_state, "updated_at": updated_at, "record": record}
    branch_manifest["execution_state"] = new_state
    branch_manifest.setdefault("status_history", []).append(event)
    branch["execution_state"] = new_state

    states = [item["execution_state"] for item in branches]
    if all(state == "planned" for state in states):
        iteration_state = "planned"
    elif all(state == "aborted" for state in states):
        iteration_state = "aborted"
    elif all(state in {"completed", "aborted"} for state in states):
        iteration_state = "completed"
    else:
        iteration_state = "running"
    previous_iteration_state = manifest["execution_state"]
    manifest["execution_state"] = iteration_state
    completed_branches = [
        item for item in branches if item["execution_state"] == "completed"
    ]
    manifest["all_blocking_acceptance_items_passed"] = bool(completed_branches) and any(
        bool(item.get("release_eligible", False)) for item in completed_branches
    )
    manifest.setdefault("status_history", []).append({
        "state": iteration_state,
        "updated_at": updated_at,
        "record": f"branch {branch_id}: {record}",
    })
    entry["execution_state"] = iteration_state
    entry["all_blocking_acceptance_items_passed"] = manifest[
        "all_blocking_acceptance_items_passed"
    ]
    entry["branch_states"] = {
        item["branch_id"]: item["execution_state"] for item in branches
    }
    completed = [
        item["iteration_id"]
        for item in registry["iterations"]
        if item["execution_state"] == "completed"
    ]
    registry["last_completed_iteration"] = completed[-1] if completed else None

    original_branch = branch_manifest_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    registry_path = result_root / "iteration_registry.json"
    original_registry = registry_path.read_bytes()
    branch_readme = branch_dir / "README.md"
    original_branch_readme = branch_readme.read_bytes()
    summary = result_root / entry["summary"]
    original_summary = summary.read_bytes()
    try:
        atomic_write_json(branch_manifest_path, branch_manifest)
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(registry_path, registry)
        atomic_write_text(
            branch_readme,
            f"# {iteration_id}/{branch_id}\n\n"
            f"> 状态：{new_state}\n\n"
            f"最近进度记录：{record}\n\n"
            "资产目录：`checkpoints/`、`metrics/`、`predictions/`、"
            "`examples/`、`logs/`、`tests/`。\n",
        )
        atomic_write_text(
            summary,
            f"# {iteration_id}：{manifest['title']}\n\n"
            f"> 状态：{iteration_state}\n\n"
            f"{manifest['objective']}\n\n"
            "## 分支进度\n\n"
            + "\n".join(
                f"- `{item['branch_id']}`：{item['execution_state']}"
                for item in branches
            )
            + "\n",
        )
    except Exception:
        branch_manifest_path.write_bytes(original_branch)
        manifest_path.write_bytes(original_manifest)
        registry_path.write_bytes(original_registry)
        branch_readme.write_bytes(original_branch_readme)
        summary.write_bytes(original_summary)
        raise
    return {
        "iteration_id": iteration_id,
        "branch_id": branch_id,
        "previous_branch_state": current,
        "branch_state": new_state,
        "previous_iteration_state": previous_iteration_state,
        "iteration_state": iteration_state,
        "last_completed_iteration": registry["last_completed_iteration"],
        "branch_states": entry["branch_states"],
    }


def select_candidate(
    result_root: Path,
    *,
    iteration_id: str,
    branch_id: str,
    decision_record: str,
    selected_at: str,
) -> dict[str, Any]:
    if not decision_record.strip():
        raise IterationManagementError("candidate selection requires a decision record")
    registry = load_registry(result_root)
    matches = [item for item in registry["iterations"] if item["iteration_id"] == iteration_id]
    if not matches:
        raise IterationManagementError(f"unknown iteration: {iteration_id}")
    entry = matches[0]
    if entry["execution_state"] != "completed":
        raise IterationManagementError("candidate iteration must be completed")
    manifest = show_iteration(result_root, iteration_id)
    if branch_id not in {item["branch_id"] for item in manifest["branches"]}:
        raise IterationManagementError(f"unknown branch: {branch_id}")
    for item in registry["iterations"]:
        if item["selection_decision"] == "current_candidate":
            item["selection_decision"] = "superseded"
    entry["selection_decision"] = "current_candidate"
    pointer = {
        "schema_version": 1,
        "work_package": registry["work_package"],
        "current_candidate": {
            "iteration_id": iteration_id,
            "branch_id": branch_id,
            "selected_at": selected_at,
            "decision_record": decision_record,
        },
    }
    registry["current_candidate"] = pointer["current_candidate"]
    atomic_write_json(result_root / "current_candidate.json", pointer)
    atomic_write_json(result_root / "iteration_registry.json", registry)
    return pointer


def _safe_release_destination(relative: str) -> PurePosixPath:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise IterationManagementError(f"unsafe release destination: {relative}")
    return value


def promote_release(
    project_root: Path,
    result_root: Path,
    *,
    iteration_id: str,
    branch_id: str,
    release_id: str,
    approval_record: str,
    released_at: str,
) -> dict[str, Any]:
    if not RELEASE_ID.fullmatch(release_id):
        raise IterationManagementError(f"invalid release id: {release_id}")
    if not approval_record.strip():
        raise IterationManagementError("release requires a teacher confirmation record")
    registry = load_registry(result_root)
    candidate = load_json(result_root / "current_candidate.json").get("current_candidate")
    expected_candidate = {"iteration_id": iteration_id, "branch_id": branch_id}
    if candidate is None or any(candidate.get(key) != value for key, value in expected_candidate.items()):
        raise IterationManagementError("release source is not the current candidate")
    entry = next(
        (item for item in registry["iterations"] if item["iteration_id"] == iteration_id),
        None,
    )
    if entry is None or entry["execution_state"] != "completed":
        raise IterationManagementError("release source iteration is not completed")
    manifest = show_iteration(result_root, iteration_id)
    if not manifest.get("all_blocking_acceptance_items_passed"):
        raise IterationManagementError("blocking acceptance items have not all passed")
    branch = next((item for item in manifest["branches"] if item["branch_id"] == branch_id), None)
    if branch is None or not branch.get("release_eligible"):
        raise IterationManagementError("branch is not eligible for release")
    artifacts = [
        item for item in manifest.get("release_artifacts", []) if item.get("branch_id") == branch_id
    ]
    required_roles = {"inference_package", "model_card", "metrics", "capability_examples", "usage"}
    if {item.get("role") for item in artifacts} != required_roles:
        raise IterationManagementError("release artifact roles are incomplete or duplicated")

    releases_dir = result_root / "releases"
    destination = releases_dir / release_id
    if destination.exists():
        raise IterationManagementError(f"release already exists: {release_id}")
    release_registry_path = releases_dir / "release_registry.json"
    release_registry = load_json(release_registry_path)
    if any(item.get("release_id") == release_id for item in release_registry.get("releases", [])):
        raise IterationManagementError(f"release id already registered: {release_id}")
    staging = releases_dir / f".staging-{release_id}-{uuid.uuid4().hex}"
    registry_original = json.loads(json.dumps(registry))
    release_registry_original = json.loads(json.dumps(release_registry))
    official_path = result_root / "official_release.json"
    official_original = load_json(official_path)
    copied: list[dict[str, Any]] = []
    try:
        staging.mkdir(parents=True, exist_ok=False)
        for item in artifacts:
            source = project_root / item["source_path"]
            if not source.is_file() or sha256_file(source) != item["sha256"]:
                raise IterationManagementError(f"release source missing or changed: {item['source_path']}")
            relative = _safe_release_destination(item["destination"])
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = sha256_file(target)
            if digest != item["sha256"]:
                raise IterationManagementError(f"release copy digest differs: {item['destination']}")
            copied.append({
                "role": item["role"],
                "path": relative.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": digest,
            })
        release_manifest = {
            "schema_version": 1,
            "work_package": registry["work_package"],
            "release_id": release_id,
            "source_iteration": iteration_id,
            "source_branch": branch_id,
            "released_at": released_at,
            "teacher_confirmation_record": approval_record,
            "artifacts": copied,
        }
        atomic_write_json(staging / "release_manifest.json", release_manifest)
        os.replace(staging, destination)
        release_entry = {
            "release_id": release_id,
            "source_iteration": iteration_id,
            "source_branch": branch_id,
            "released_at": released_at,
            "manifest": f"releases/{release_id}/release_manifest.json",
        }
        release_registry["releases"].append(release_entry)
        official = {
            "schema_version": 1,
            "work_package": registry["work_package"],
            "official_release": release_entry,
        }
        registry["official_release"] = release_entry
        entry["selection_decision"] = "released"
        atomic_write_json(release_registry_path, release_registry)
        atomic_write_json(official_path, official)
        atomic_write_json(result_root / "iteration_registry.json", registry)
        return release_manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if destination.exists():
            shutil.rmtree(destination)
        atomic_write_json(release_registry_path, release_registry_original)
        atomic_write_json(official_path, official_original)
        atomic_write_json(result_root / "iteration_registry.json", registry_original)
        raise
