from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from mini_multimodal.iteration_management import (
    artifact_reference,
    atomic_write_json,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/P2-W04_tiny_image_ldm"
CATALOG_ROOT_NAMES = {
    "artifact_inventory.json",
    "current_candidate.json",
    "development_changelog.md",
    "iteration_registry.json",
    "migration_report.json",
    "inventory_refresh_report.json",
    "official_release.json",
}
HISTORICAL_ITERATION_IDS = {"I001", "I002", "I003", "I004"}
SCAN_SPEC = [
    {
        "base": "results/P2-W04_tiny_image_ldm",
        "mode": "recursive",
        "pattern": "*",
        "exclude_prefixes": ["iterations", "releases"],
        "exclude_names": sorted(CATALOG_ROOT_NAMES),
    },
    {"base": "data/ldm", "mode": "glob", "pattern": "p2_w04*/**/*"},
    {"base": "docs", "mode": "glob", "pattern": "P2-W04*"},
    {"base": "docs/迭代与发布管理机制.md", "mode": "exact"},
    {"base": "docs/迭代与发布管理测试计划.md", "mode": "exact"},
    {"base": "configs", "mode": "glob", "pattern": "*p2_w04*"},
    {"base": "examples", "mode": "glob", "pattern": "*p2_w04*"},
    {"base": "tests", "mode": "glob", "pattern": "test_p2_w04*"},
    {"base": "tests/test_iteration_management.py", "mode": "exact"},
    {"base": "src/mini_multimodal/ldm", "mode": "recursive", "pattern": "*"},
    {"base": "src/mini_multimodal/iteration_management.py", "mode": "exact"},
    {"base": "src/mini_multimodal/tools/manage_iterations.py", "mode": "exact"},
    {"base": "src/mini_multimodal/tools/build_p2_w04_iteration_catalog.py", "mode": "exact"},
    {"base": "src/mini_multimodal/tools/run_p2_w04_i005.py", "mode": "exact"},
    {"base": "src/mini_multimodal/tools/refresh_p2_w04_artifact_inventory.py", "mode": "exact"},
]


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in item.parts):
            continue
        if item.suffix in {".pyc", ".pyo"}:
            continue
        yield item


def _scan_paths() -> list[tuple[Path, str]]:
    values: dict[Path, str] = {}
    for path in _iter_files(RESULT):
        relative = path.relative_to(RESULT)
        if relative.parts[0] in {"iterations", "releases"}:
            continue
        if len(relative.parts) == 1 and relative.name in CATALOG_ROOT_NAMES:
            continue
        values[path] = "results"
    for directory in sorted((ROOT / "data/ldm").glob("p2_w04*")):
        for path in _iter_files(directory):
            values[path] = "data"
    for path in sorted((ROOT / "docs").glob("P2-W04*")):
        if path.is_file():
            values[path] = "documentation"
    for name in ("迭代与发布管理机制.md", "迭代与发布管理测试计划.md"):
        path = ROOT / "docs" / name
        if path.is_file():
            values[path] = "governance"
    for path in sorted((ROOT / "configs").glob("*p2_w04*")):
        if path.is_file():
            values[path] = "configuration"
    for path in sorted((ROOT / "examples").glob("*p2_w04*")):
        if path.is_file():
            values[path] = "example"
    for path in sorted((ROOT / "tests").glob("test_p2_w04*")):
        if path.is_file():
            values[path] = "test"
    iteration_test = ROOT / "tests/test_iteration_management.py"
    if iteration_test.is_file():
        values[iteration_test] = "governance"
    for path in _iter_files(ROOT / "src/mini_multimodal/ldm"):
        values[path] = "source"
    for name in (
        "src/mini_multimodal/iteration_management.py",
        "src/mini_multimodal/tools/manage_iterations.py",
        "src/mini_multimodal/tools/build_p2_w04_iteration_catalog.py",
        "src/mini_multimodal/tools/run_p2_w04_i005.py",
        "src/mini_multimodal/tools/refresh_p2_w04_artifact_inventory.py",
    ):
        path = ROOT / name
        if path.is_file():
            values[path] = "governance"
    return sorted(values.items(), key=lambda item: item[0].relative_to(ROOT).as_posix())


def _result_owner(relative: Path) -> str:
    text = relative.as_posix().lower()
    if not text.startswith("revision_v2/"):
        if relative.name == "README.md":
            return "work_package_shared"
        return "I001"
    if any(token in text for token in (
        "attempt3", "training_attempt3", "generated_attempt3", "geometry_lite",
    )) or relative.name in {"capability_examples.md", "capability_examples.json", "artifact_contract.json"}:
        return "I004"
    if any(token in text for token in (
        "attempt2", "training_grounded", "generated_grounded", "grounded_",
    )):
        return "I003"
    if any(token in text for token in (
        "attempt1", "training_v2", "generated_v2", "overfit_v2",
    )) or relative.name in {
        "generation_metrics.json",
        "generation_predictions.jsonl",
        "denoising_metrics.json",
        "overfit_report.json",
    }:
        return "I002"
    return "work_package_shared"


def _owner(path: Path, scope: str) -> str:
    relative = path.relative_to(ROOT)
    if scope == "results":
        return _result_owner(path.relative_to(RESULT))
    if scope == "data":
        text = relative.as_posix()
        if "/p2_w04_v1/" in f"/{text}/":
            return "I001"
        if "/p2_w04_geometry_v3/" in f"/{text}/":
            return "I004"
        return "work_package_shared"
    if scope == "configuration":
        return "I001" if path.name == "p2_w04_tiny_image_ldm.json" else "work_package_shared"
    if scope == "documentation":
        name = path.name
        if name == "P2-W04_总结报告.md":
            return "I001"
        if "第一轮" in name:
            return "I002"
        if "第二轮" in name:
            return "I003"
        if "第三轮" in name:
            return "I004"
        return "work_package_shared"
    if scope == "test":
        name = path.name
        if "attempt1" in name:
            return "I002"
        if "attempt2" in name or "grounded" in name:
            return "I003"
        if "attempt3" in name or "geometry" in name:
            return "I004"
    return "work_package_shared"


def _inventory() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path, scope in _scan_paths():
        files.append({
            "path": path.relative_to(ROOT).as_posix(),
            "role": f"p2_w04_{scope}_inventory",
            "scope": scope,
            "owner": _owner(path, scope),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    owner_counts = Counter(item["owner"] for item in files)
    scope_counts = Counter(item["scope"] for item in files)
    return {
        "schema_version": 1,
        "work_package": "P2-W04",
        "created_at": "2026-08-09",
        "policy": "non_destructive_full_related_file_inventory",
        "scan_scope": [
            "results/P2-W04_tiny_image_ldm excluding catalog metadata",
            "data/ldm/p2_w04*",
            "docs/P2-W04* and iteration governance documents",
            "configs/examples/tests related to P2-W04",
            "src/mini_multimodal/ldm and iteration management source",
        ],
        "scan_spec": SCAN_SPEC,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "owner_counts": dict(sorted(owner_counts.items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "unclassified_count": sum(item["owner"] == "unclassified" for item in files),
        "files": files,
    }


def _ref(relative: str, role: str) -> dict[str, Any]:
    return artifact_reference(ROOT, ROOT / relative, role=role)


def _historical_manifest(
    iteration_id: str,
    title: str,
    objective: str,
    parent: str | None,
    decision: str,
    branches: list[dict[str, Any]],
    artifacts: list[tuple[str, str]],
    conclusion: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "work_package": "P2-W04",
        "iteration_id": iteration_id,
        "title": title,
        "objective": objective,
        "parent_iteration": parent,
        "created_at": "historical_before_iteration_governance_2026-08-09",
        "authorization_record": "historical conversation authorization recorded in work-package reports",
        "execution_state": "completed",
        "selection_decision": decision,
        "historical_registration": True,
        "all_blocking_acceptance_items_passed": False,
        "branches": branches,
        "source_reproducibility": {
            "state": "historical_snapshot_unavailable",
            "reason": "No deterministic source snapshot was created before this historical training run; current source must not be claimed as byte-identical historical source.",
        },
        "artifact_inventory": "results/P2-W04_tiny_image_ldm/artifact_inventory.json",
        "artifact_references": [_ref(path, role) for path, role in artifacts],
        "acceptance_conclusion": conclusion,
        "release_artifacts": [],
    }


def _manifests() -> list[dict[str, Any]]:
    return [
        _historical_manifest(
            "I001",
            "原版TinyImageLDM",
            "建立首个潜空间文本条件扩散训练、采样、推理和教学轨迹闭环。",
            None,
            "superseded",
            [{
                "branch_id": "baseline",
                "release_eligible": False,
                "checkpoint_manifest": "results/P2-W04_tiny_image_ldm/checkpoints/checkpoint_manifest.json",
                "metrics": "results/P2-W04_tiny_image_ldm/generation_metrics.json",
            }],
            [
                ("configs/p2_w04_tiny_image_ldm.json", "original_config"),
                ("data/ldm/p2_w04_v1/manifest.json", "original_data_manifest"),
                ("results/P2-W04_tiny_image_ldm/generation_metrics.json", "original_generation_metrics"),
                ("results/P2-W04_tiny_image_ldm/generation_predictions.jsonl", "original_predictions"),
                ("results/P2-W04_tiny_image_ldm/checkpoints/checkpoint_manifest.json", "original_checkpoint_manifest"),
                ("results/P2-W04_tiny_image_ldm/capability_examples.md", "original_capability_examples"),
                ("docs/P2-W04_总结报告.md", "original_summary"),
            ],
            "原版按旧评价完成，但修订评价发现位置、大小和连续轮廓定义不完整，不能作为P2-W04正式发布。",
        ),
        _historical_manifest(
            "I002",
            "修订评价后的第一轮基础模型",
            "在冻结v2数据、评价器和阈值下训练第一轮修订模型，验证位置条件并暴露形状与碎片问题。",
            "I001",
            "superseded",
            [{
                "branch_id": "baseline_v2",
                "release_eligible": False,
                "checkpoint_manifest": "results/P2-W04_tiny_image_ldm/revision_v2/training_v2/checkpoints/checkpoint_manifest.json",
                "metrics": "results/P2-W04_tiny_image_ldm/revision_v2/generation_metrics.json",
            }],
            [
                ("configs/p2_w04_tiny_image_ldm_v2.json", "v2_config"),
                ("data/ldm/p2_w04_v2/manifest.json", "v2_data_manifest"),
                ("data/ldm/p2_w04_eval_v2/manifest.json", "fixed_evaluation_manifest"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/generation_metrics.json", "attempt1_metrics"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/generation_predictions.jsonl", "attempt1_predictions"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/training_v2/checkpoints/checkpoint_manifest.json", "attempt1_checkpoint_manifest"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/attempt1_capability_examples.md", "attempt1_capability_examples"),
                ("docs/P2-W04_模型改进第一轮报告.md", "attempt1_summary"),
            ],
            "位置达到目标，但形状、数量和联合正确性未通过冻结验收，后由对象接地结构继续改进。",
        ),
        _historical_manifest(
            "I003",
            "第二轮对象接地模型",
            "增加对象语义和区域特征注入，提高位置、数量与形状条件绑定。",
            "I002",
            "not_selected",
            [{
                "branch_id": "grounded",
                "release_eligible": False,
                "checkpoint_manifest": "results/P2-W04_tiny_image_ldm/revision_v2/training_grounded/checkpoints/checkpoint_manifest.json",
                "metrics": "results/P2-W04_tiny_image_ldm/revision_v2/grounded_generation_metrics.json",
            }],
            [
                ("configs/p2_w04_tiny_image_ldm_v2.json", "shared_v2_config"),
                ("data/ldm/p2_w04_v2/manifest.json", "shared_v2_data_manifest"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/grounded_generation_metrics.json", "attempt2_metrics"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/grounded_generation_predictions.jsonl", "attempt2_predictions"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/training_grounded/checkpoints/checkpoint_manifest.json", "attempt2_checkpoint_manifest"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/attempt2_capability_examples.md", "attempt2_capability_examples"),
                ("docs/P2-W04_模型改进第二轮报告.md", "attempt2_summary"),
            ],
            "位置和数量显著改善，但已见形状、数量和联合正确性仍未全部通过；不选择为当前候选。",
        ),
        _historical_manifest(
            "I004",
            "第三轮同起点续训与几何辅助",
            "区分继续训练收益与前景/边界辅助损失收益，判断64×64是否构成硬上限。",
            "I003",
            "undecided",
            [
                {
                    "branch_id": "control",
                    "release_eligible": False,
                    "checkpoint_manifest": "results/P2-W04_tiny_image_ldm/revision_v2/training_attempt3_control/checkpoints/checkpoint_manifest.json",
                    "metrics": "results/P2-W04_tiny_image_ldm/revision_v2/attempt3_control_generation_metrics.json",
                },
                {
                    "branch_id": "geometry",
                    "release_eligible": False,
                    "checkpoint_manifest": "results/P2-W04_tiny_image_ldm/revision_v2/training_attempt3_geometry/checkpoints/checkpoint_manifest.json",
                    "metrics": "results/P2-W04_tiny_image_ldm/revision_v2/attempt3_geometry_generation_metrics.json",
                },
                {
                    "branch_id": "geometry_lite",
                    "release_eligible": False,
                    "checkpoint_manifest": "results/P2-W04_tiny_image_ldm/revision_v2/training_attempt3_geometry_lite/checkpoints/checkpoint_manifest.json",
                    "metrics": "results/P2-W04_tiny_image_ldm/revision_v2/attempt3_geometry_lite_generation_metrics.json",
                },
            ],
            [
                ("configs/p2_w04_tiny_image_ldm_v2.json", "shared_v2_config"),
                ("data/ldm/p2_w04_geometry_v3/manifest.json", "geometry_auxiliary_data_manifest"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/attempt3_comparison.json", "attempt3_comparison"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/attempt3_control_generation_metrics.json", "control_metrics"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/attempt3_geometry_generation_metrics.json", "geometry_metrics"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/attempt3_geometry_lite_generation_metrics.json", "geometry_lite_metrics"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/capability_examples.md", "attempt3_capability_examples"),
                ("results/P2-W04_tiny_image_ldm/revision_v2/attempt3_capability_comparison.md", "attempt3_branch_comparison"),
                ("docs/P2-W04_模型改进第三轮报告.md", "attempt3_summary"),
            ],
            "控制支证明64×64仍可改善且未见组合通过；强几何支联合正确率最高；没有单一分支通过全部已见冻结阈值，因此候选仍待教师决定且不得发布。",
        ),
    ]


def _summary(manifest: dict[str, Any]) -> str:
    branch_lines = "\n".join(
        f"- `{item['branch_id']}`：{item.get('metrics', '无独立指标路径')}"
        for item in manifest["branches"]
    )
    return (
        f"# {manifest['iteration_id']}：{manifest['title']}\n\n"
        f"> 执行状态：{manifest['execution_state']}  \n"
        f"> 选择决定：{manifest['selection_decision']}  \n"
        f"> 历史登记：是；没有训练前确定性源码快照  \n\n"
        f"## 目标\n\n{manifest['objective']}\n\n"
        f"## 分支\n\n{branch_lines}\n\n"
        f"## 验收结论\n\n{manifest['acceptance_conclusion']}\n\n"
        "完整路径、大小和SHA-256见同目录`iteration_manifest.json`及工作包根目录"
        "`artifact_inventory.json`。既有大型文件保持原位，没有移动、复制或重命名。\n"
    )


def _assert_migration_rebuild_is_safe(result_root: Path) -> None:
    """Prevent the one-time history cataloger from erasing future live state."""

    registry_path = result_root / "iteration_registry.json"
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        ids = {str(item.get("iteration_id")) for item in registry.get("iterations", [])}
        if not ids.issubset(HISTORICAL_ITERATION_IDS):
            raise RuntimeError("history catalog rebuild refused after a future iteration exists")
        if registry.get("current_candidate") is not None or registry.get("official_release") is not None:
            raise RuntimeError("history catalog rebuild refused after candidate or release selection")
    release_path = result_root / "releases" / "release_registry.json"
    if release_path.is_file():
        releases = json.loads(release_path.read_text(encoding="utf-8")).get("releases", [])
        if releases:
            raise RuntimeError("history catalog rebuild refused after a formal release exists")
    iteration_dir = result_root / "iterations"
    if iteration_dir.is_dir():
        unexpected = {
            path.name for path in iteration_dir.iterdir()
            if path.is_dir() and path.name not in HISTORICAL_ITERATION_IDS
        }
        if unexpected:
            raise RuntimeError(
                f"history catalog rebuild refused with future iteration directories: {sorted(unexpected)}"
            )


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    _assert_migration_rebuild_is_safe(RESULT)
    before = _inventory()
    manifests = _manifests()
    iterations_dir = RESULT / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        directory = iterations_dir / manifest["iteration_id"]
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "iteration_manifest.json", manifest)
        (directory / "summary.md").write_text(_summary(manifest), encoding="utf-8")

    registry = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "result_root": "results/P2-W04_tiny_image_ldm",
        "management_protocol": "docs/迭代与发布管理机制.md",
        "migration_policy": "non_destructive_historical_registration",
        "last_completed_iteration": "I004",
        "current_candidate": None,
        "official_release": None,
        "iterations": [
            {
                "iteration_id": item["iteration_id"],
                "title": item["title"],
                "parent_iteration": item["parent_iteration"],
                "manifest": f"iterations/{item['iteration_id']}/iteration_manifest.json",
                "summary": f"iterations/{item['iteration_id']}/summary.md",
                "branches": [branch["branch_id"] for branch in item["branches"]],
                "execution_state": item["execution_state"],
                "selection_decision": item["selection_decision"],
                "all_blocking_acceptance_items_passed": item[
                    "all_blocking_acceptance_items_passed"
                ],
            }
            for item in manifests
        ],
    }
    atomic_write_json(RESULT / "iteration_registry.json", registry)
    atomic_write_json(RESULT / "current_candidate.json", {
        "schema_version": 1,
        "work_package": "P2-W04",
        "current_candidate": None,
        "reason": "No I001-I004 branch passes every frozen blocking acceptance item; candidate selection remains a teacher decision.",
    })
    atomic_write_json(RESULT / "official_release.json", {
        "schema_version": 1,
        "work_package": "P2-W04",
        "official_release": None,
        "reason": "P2-W04 remains in progress and no branch is release eligible.",
    })
    releases = RESULT / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    atomic_write_json(releases / "release_registry.json", {
        "schema_version": 1,
        "work_package": "P2-W04",
        "releases": [],
    })
    (releases / "README.md").write_text(
        "# P2-W04正式发布\n\n当前没有正式发布。只有全部阻塞验收通过并经教师明确确认后，"
        "发布提升程序才允许创建`RELxxx/`。最近完成迭代或当前候选不会自动成为正式发布。\n",
        encoding="utf-8",
    )
    (RESULT / "development_changelog.md").write_text(
        "# P2-W04开发变更日志\n\n"
        "## 2026-08-09：迭代与发布治理补充\n\n"
        "本次只建立管理机制、登记历史文件和增加查询/验证工具，没有训练新模型，"
        "并补强未来分支脚手架、受控进度转换和完成资产核验，因此不增加模型迭代编号。"
        "I004仍是最近完成迭代；候选和正式发布均为空。\n",
        encoding="utf-8",
    )
    atomic_write_json(RESULT / "artifact_inventory.json", before)

    after = _inventory()
    before_map = {item["path"]: item["sha256"] for item in before["files"]}
    after_map = {item["path"]: item["sha256"] for item in after["files"]}
    if before_map != after_map:
        raise RuntimeError("legacy or related files changed during non-destructive catalog build")
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "date": "2026-08-09",
        "iteration_count": 4,
        "last_completed_iteration": "I004",
        "current_candidate": None,
        "official_release": None,
        "release_count": 0,
        "inventory_file_count": before["file_count"],
        "inventory_total_bytes": before["total_bytes"],
        "unclassified_count": before["unclassified_count"],
        "before_after_path_digest_maps_equal": True,
        "moved_file_count": 0,
        "renamed_file_count": 0,
        "deleted_file_count": 0,
        "copied_legacy_artifact_count": 0,
        "historical_source_snapshot_available_count": 0,
        "verified": before["unclassified_count"] == 0,
    }
    atomic_write_json(RESULT / "migration_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
