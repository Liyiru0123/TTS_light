"""Audit iteration-trajectory coverage across the four primary model documents."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/P0-W04_governance/model_iteration_documentation_audit.json"

PACKAGES = {
    "P2-W01": {
        "mode": "single_cycle",
        "documents": {
            "design": "docs/P2-W01_共享训练框架与判别基线设计.md",
            "test_plan": "docs/P2-W01_测试计划.md",
            "model_card": "docs/P2-W01_快速判别基线模型卡.md",
            "summary": "docs/P2-W01_总结报告.md",
        },
    },
    "P2-W02": {
        "mode": "single_cycle",
        "documents": {
            "design": "docs/P2-W02_图像VAE设计.md",
            "test_plan": "docs/P2-W02_测试计划.md",
            "model_card": "docs/P2-W02_图像VAE模型卡.md",
            "summary": "docs/P2-W02_总结报告.md",
        },
    },
    "P2-W03": {
        "mode": "single_cycle",
        "documents": {
            "design": "docs/P2-W03_TinyVLM设计.md",
            "test_plan": "docs/P2-W03_测试计划.md",
            "model_card": "docs/P2-W03_TinyVLM模型卡.md",
            "summary": "docs/P2-W03_总结报告.md",
        },
    },
    "P2-W04": {
        "mode": "multi_iteration",
        "registry": "results/P2-W04_tiny_image_ldm/iteration_registry.json",
        "documents": {
            "design": "docs/P2-W04_TinyImageLDM设计.md",
            "test_plan": "docs/P2-W04_测试计划.md",
            "model_card": "docs/P2-W04_TinyImageLDM模型卡.md",
            "summary": "docs/P2-W04_总结报告.md",
        },
    },
    "P2-W05": {
        "mode": "multi_iteration",
        "registry": "results/P2-W05_tiny_asr/iteration_registry.json",
        "documents": {
            "design": "docs/P2-W05_TinyASR设计.md",
            "test_plan": "docs/P2-W05_测试计划.md",
            "model_card": "docs/P2-W05_TinyASR模型卡.md",
            "summary": "docs/P2-W05_总结报告.md",
        },
    },
    "P3-W01": {
        "mode": "multi_iteration",
        "registry": "results/P3-W01_tiny_video_mllm/iteration_registry.json",
        "documents": {
            "design": "docs/P3-W01_TinyVideoMLLM设计.md",
            "test_plan": "docs/P3-W01_测试计划.md",
            "model_card": "docs/P3-W01_TinyVideoMLLM模型卡.md",
            "summary": "docs/P3-W01_总结报告.md",
        },
    },
    "P3-W02": {
        "mode": "multi_iteration",
        "registry": "results/P3-W02_speech_vae/iteration_registry.json",
        "documents": {
            "design": "docs/P3-W02_语音潜表示与固定解码链设计.md",
            "test_plan": "docs/P3-W02_测试计划.md",
            "model_card": "docs/P3-W02_语音潜表示模型卡.md",
            "summary": "docs/P3-W02_总结报告.md",
        },
    },
    "P3-W03": {
        "mode": "multi_iteration",
        "registry": "results/P3-W03_tiny_tts/iteration_registry.json",
        "documents": {
            "design": "docs/P3-W03_TinyTTS设计.md",
            "test_plan": "docs/P3-W03_测试计划.md",
            "model_card": "docs/P3-W03_TinyTTS模型卡.md",
            "summary": "docs/P3-W03_总结报告.md",
        },
    },
    "P3-W04": {
        "mode": "multi_iteration",
        "registry": "results/P3-W04_video_generation/iteration_registry.json",
        "documents": {
            "design": "docs/P3-W04_TinyT2V设计.md",
            "test_plan": "docs/P3-W04_测试计划.md",
            "model_card": "docs/P3-W04_TinyT2V模型卡.md",
            "summary": "docs/P3-W04_总结报告.md",
        },
    },
}

ROLE_MARKERS = {
    "design": ("技术设计演化", "设计决策总表", "设计收敛轨迹", "结构收敛轨迹", "开发变更说明"),
    "test_plan": ("测试协议", "评价协议", "测试变化", "测试收敛轨迹"),
    "model_card": ("模型谱系", "结构谱系", "结构来源", "结构与训练路线谱系"),
    "summary": ("技术变更轨迹", "技术轨迹", "完整技术变更轨迹"),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_audit() -> dict:
    packages = []
    all_passed = True
    for work_package, spec in PACKAGES.items():
        expected_ids: list[str] = []
        registry_path = spec.get("registry")
        if registry_path:
            registry = load_json(ROOT / registry_path)
            expected_ids = [row["iteration_id"] for row in registry["iterations"]]

        document_rows = []
        for role, relative in spec["documents"].items():
            path = ROOT / relative
            exists = path.is_file()
            text = path.read_text(encoding="utf-8") if exists else ""
            found_ids = sorted(set(re.findall(r"I\d{3}", text)))
            missing_ids = [item for item in expected_ids if item not in found_ids]
            has_role_section = any(marker in text for marker in ROLE_MARKERS[role])
            if spec["mode"] == "single_cycle":
                has_scope_statement = (
                    "没有" in text and "Ixxx" in text
                ) or "只有一个正式" in text or "一个正式模型周期" in text or "一个正式" in text
                passed = exists and has_role_section and has_scope_statement
            else:
                has_scope_statement = None
                passed = exists and has_role_section and not missing_ids
            document_rows.append(
                {
                    "role": role,
                    "path": relative,
                    "exists": exists,
                    "found_iteration_ids": found_ids,
                    "missing_iteration_ids": missing_ids,
                    "has_role_specific_trajectory_section": has_role_section,
                    "has_single_cycle_scope_statement": has_scope_statement,
                    "passed": passed,
                }
            )

        manifest_paths_exist = True
        if registry_path:
            result_root = (ROOT / registry_path).parent
            manifest_paths_exist = all(
                (result_root / row.get("manifest", f"iterations/{row['iteration_id']}/iteration_manifest.json")).is_file()
                for row in registry["iterations"]
            )
        package_passed = all(row["passed"] for row in document_rows) and manifest_paths_exist
        all_passed = all_passed and package_passed
        packages.append(
            {
                "work_package": work_package,
                "mode": spec["mode"],
                "registry": registry_path,
                "expected_iteration_ids": expected_ids,
                "iteration_manifest_paths_exist": manifest_paths_exist,
                "documents": document_rows,
                "passed": package_passed,
            }
        )

    return {
        "schema_version": 1,
        "audit": "model_iteration_trajectory_four_document_closure",
        "scope": {
            "model_work_packages": 9,
            "multi_iteration_work_packages": 6,
            "single_cycle_work_packages": 3,
            "primary_documents": 36,
            "registered_iterations": sum(len(row["expected_iteration_ids"]) for row in packages),
            "excluded_work_package": {
                "work_package": "P3-W05",
                "reason": "integration and gap-ledger work package; it creates no model checkpoint or model iteration",
            },
        },
        "packages": packages,
        "passed_packages": sum(row["passed"] for row in packages),
        "all_passed": all_passed,
    }


def main() -> int:
    audit = build_audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"all_passed": audit["all_passed"], "passed_packages": audit["passed_packages"], "scope": audit["scope"]}, ensure_ascii=False, indent=2))
    return 0 if audit["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
