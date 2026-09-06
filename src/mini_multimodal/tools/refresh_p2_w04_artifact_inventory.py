"""Refresh file digests without rebuilding iteration, candidate, or release state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mini_multimodal.iteration_management import atomic_write_json
from mini_multimodal.tools.build_p2_w04_iteration_catalog import RESULT, _inventory


def _by_path(inventory: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["path"]): item for item in inventory.get("files", [])}


def inventory_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    old, new = _by_path(before), _by_path(after)
    shared = sorted(set(old) & set(new))
    changed = [
        path for path in shared
        if old[path].get("bytes") != new[path].get("bytes")
        or old[path].get("sha256") != new[path].get("sha256")
        or old[path].get("owner") != new[path].get("owner")
        or old[path].get("scope") != new[path].get("scope")
    ]
    return {
        "added": sorted(set(new) - set(old)),
        "missing": sorted(set(old) - set(new)),
        "changed": changed,
        "unchanged_count": len(shared) - len(changed),
    }


def refresh_inventory(result_root: Path = RESULT) -> dict[str, Any]:
    expected = RESULT.resolve()
    if result_root.resolve() != expected:
        raise ValueError("P2-W04 inventory refresh only accepts the configured result root")
    path = result_root / "artifact_inventory.json"
    before = json.loads(path.read_text(encoding="utf-8"))
    registry_before = (result_root / "iteration_registry.json").read_bytes()
    candidate_before = (result_root / "current_candidate.json").read_bytes()
    release_before = (result_root / "official_release.json").read_bytes()
    after = _inventory()
    diff = inventory_diff(before, after)
    atomic_write_json(path, after)
    protected_unchanged = (
        registry_before == (result_root / "iteration_registry.json").read_bytes()
        and candidate_before == (result_root / "current_candidate.json").read_bytes()
        and release_before == (result_root / "official_release.json").read_bytes()
    )
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "operation": "artifact_inventory_digest_refresh_only",
        "iteration_registry_candidate_and_release_unchanged": protected_unchanged,
        "historical_migration_rebuilt": False,
        "file_count": after["file_count"],
        "total_bytes": after["total_bytes"],
        "unclassified_count": after["unclassified_count"],
        "diff": diff,
        "verified": protected_unchanged and after["unclassified_count"] == 0,
    }
    atomic_write_json(result_root / "inventory_refresh_report.json", report)
    if not report["verified"]:
        raise RuntimeError("P2-W04 inventory-only refresh failed")
    return report


def main() -> None:
    print(json.dumps(refresh_inventory(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
