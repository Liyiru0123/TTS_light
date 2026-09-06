"""Structural checks for the file-backed project memory system."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mini_multimodal.iteration_management import IterationManagementError, verify_all


LEDGER_HEADING = re.compile(r"^## (DATA-[A-Z]+-\d{3})：(.+)$", re.MULTILINE)
LEDGER_ANY_DATA_HEADING = re.compile(r"^## (DATA-[^：\r\n]+)：(.+)$", re.MULTILINE)
WORK_PACKAGE_STATUS = re.compile(
    r"^\s*(P[0-4]-W\d{2})[^\n]*\[(已验收|进行中|已完成，待确认|未启动)\]",
    re.MULTILINE,
)
CANONICAL_STATUSES = {"未启动", "进行中", "已完成，待确认", "已验收"}
REQUIRED_LEDGER_FIELDS = (
    "状态",
    "严重度",
    "影响工作包",
    "最迟处理节点",
    "当前证据",
    "处理策略",
    "关闭条件",
    "证据路径",
    "最近复核",
)


def resolve_json_path(data: Any, json_path: str) -> Any:
    if not json_path.startswith("$."):
        raise ValueError(f"unsupported JSON path: {json_path}")
    value = data
    for key in json_path[2:].split("."):
        value = value[key]
    return value


def markdown_section_body(text: str, heading: str) -> str | None:
    """Return the body below an exact level-two heading, up to the next one."""
    match = re.search(rf"^{re.escape(heading)}\s*$", text, re.MULTILINE)
    if match is None:
        return None
    next_heading = re.search(r"^##\s+", text[match.end() :], re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(text)
    return text[match.end() : end].strip()


def validate_summary_contract(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Validate the canonical work-package summary declared by schema v3 contracts."""
    if int(contract.get("schema_version", 1)) < 3:
        return {"verified": True, "errors": []}

    errors: list[str] = []
    work_package = str(contract.get("work_package", ""))
    summary_relative = contract.get("summary_document")
    iteration_summary_relative = contract.get("iteration_summary_document")
    historical_summaries = contract.get("historical_summary_documents")
    summary_contract = contract.get("summary_contract")
    primary_evidence_paths = contract.get("primary_evidence_paths")

    for field, value in (
        ("iteration_summary_document", iteration_summary_relative),
        ("historical_summary_documents", historical_summaries),
        ("summary_contract", summary_contract),
        ("primary_evidence_paths", primary_evidence_paths),
    ):
        if not value:
            errors.append(f"missing summary contract field: {field}")

    expected_name = f"{work_package}_总结报告.md"
    if not summary_relative or Path(str(summary_relative)).name != expected_name:
        errors.append(
            f"current work-package summary must be named {expected_name}: {summary_relative}"
        )

    summary_path = root / str(summary_relative or "")
    if not summary_path.is_file():
        return {"verified": False, "errors": errors + ["current summary does not exist"]}
    summary_text = summary_path.read_text(encoding="utf-8")

    if iteration_summary_relative and not (root / iteration_summary_relative).is_file():
        errors.append(f"missing iteration summary: {iteration_summary_relative}")
    if isinstance(historical_summaries, list):
        for relative in historical_summaries:
            if not (root / relative).is_file():
                errors.append(f"missing historical summary: {relative}")

    if not isinstance(summary_contract, dict):
        errors.append("summary_contract must be an object")
        summary_contract = {}
    if summary_contract.get("document_role") != "current_work_package_summary":
        errors.append("summary_contract has a non-canonical document_role")

    declared_candidate = summary_contract.get("current_candidate") or {}
    candidate_path = root / str(contract.get("result_directory", "")) / "current_candidate.json"
    if candidate_path.is_file():
        try:
            actual_candidate = json.loads(candidate_path.read_text(encoding="utf-8")).get(
                "current_candidate"
            ) or {}
            declared_pair = (
                declared_candidate.get("iteration_id"),
                declared_candidate.get("branch_id"),
            )
            actual_pair = (
                actual_candidate.get("iteration_id"),
                actual_candidate.get("branch_id"),
            )
            if declared_pair != actual_pair:
                errors.append(
                    "summary current candidate differs from current_candidate.json: "
                    f"{declared_pair} != {actual_pair}"
                )
            candidate_label = "/".join(str(item) for item in actual_pair if item)
            if candidate_label and candidate_label not in summary_text:
                errors.append(
                    f"current summary does not identify current candidate: {candidate_label}"
                )
        except (OSError, json.JSONDecodeError, TypeError) as error:
            errors.append(f"invalid current_candidate.json for summary: {error}")
    else:
        errors.append("summary validation cannot find current_candidate.json")

    sections = summary_contract.get("sections")
    if not isinstance(sections, list) or not sections:
        errors.append("summary_contract has no required sections")
    else:
        for section in sections:
            heading = section.get("heading") if isinstance(section, dict) else None
            if not heading:
                errors.append("summary section contract has no heading")
                continue
            body = markdown_section_body(summary_text, heading)
            if body is None:
                errors.append(f"current summary missing section: {heading}")
                continue
            minimum = int(section.get("minimum_characters", 1))
            if len(body) < minimum:
                errors.append(
                    f"current summary section is too short: {heading} "
                    f"({len(body)} < {minimum})"
                )
            for term in section.get("required_terms", []):
                if term not in body:
                    errors.append(f"current summary section {heading} missing term: {term}")

    if isinstance(primary_evidence_paths, list):
        for relative in primary_evidence_paths:
            if not (root / relative).exists():
                errors.append(f"summary primary evidence path does not exist: {relative}")
            if relative not in summary_text:
                errors.append(f"current summary does not cite primary evidence: {relative}")

    return {"verified": not errors, "errors": errors}


def validate_artifact_contract(root: Path, contract_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"path": str(contract_path.relative_to(root)), "verified": False, "errors": [str(error)]}

    schema_version = int(contract.get("schema_version", 1))
    required_fields = (
        "work_package",
        "status",
        "result_directory",
        "design_document",
        "test_plan",
        "summary_document",
        "result_readme",
        "capability_showcase",
        "capability_showcase_data",
        "automatic_test_reports",
        "required_artifacts",
        "capability_interfaces",
    )
    if schema_version >= 2 and contract.get("quantitative_results"):
        required_fields += ("formal_experiment", "metric_glossary")
    for field in required_fields:
        if not contract.get(field):
            errors.append(f"missing contract field: {field}")
    if contract.get("status") not in CANONICAL_STATUSES:
        errors.append(f"non-canonical status: {contract.get('status')}")

    path_fields = (
        "result_directory",
        "design_document",
        "test_plan",
        "summary_document",
        "result_readme",
        "evidence_index",
        "checkpoint_manifest",
        "usage_document",
        "capability_showcase",
        "capability_showcase_data",
        "metric_glossary",
    )
    for field in path_fields:
        relative = contract.get(field)
        if relative and not (root / relative).exists():
            errors.append(f"missing path in {field}: {relative}")
    if schema_version >= 3:
        iteration_summary = contract.get("iteration_summary_document")
        if iteration_summary and not (root / iteration_summary).is_file():
            errors.append(f"missing path in iteration_summary_document: {iteration_summary}")
        for relative in contract.get("historical_summary_documents", []):
            if not (root / relative).is_file():
                errors.append(f"missing path in historical_summary_documents: {relative}")
    for field in ("runnable_examples", "automatic_test_reports"):
        for relative in contract.get(field, []):
            if not (root / relative).is_file():
                errors.append(f"missing path in {field}: {relative}")
    for artifact in contract.get("required_artifacts", []):
        relative = artifact.get("path")
        if not relative or not (root / relative).is_file():
            errors.append(f"missing required artifact: {relative}")

    showcase_relative = contract.get("capability_showcase")
    showcase_data_relative = contract.get("capability_showcase_data")
    showcase_verified = False
    if schema_version >= 2:
        if Path(str(showcase_relative)).name != "capability_examples.md":
            errors.append("formal experiment capability showcase must be named capability_examples.md")
        if Path(str(showcase_data_relative)).name != "capability_examples.json":
            errors.append("formal experiment capability data must be named capability_examples.json")
    if showcase_relative and showcase_data_relative:
        showcase_path = root / showcase_relative
        showcase_data_path = root / showcase_data_relative
        if showcase_path.is_file() and showcase_data_path.is_file():
            try:
                showcase = json.loads(showcase_data_path.read_text(encoding="utf-8"))
                showcase_text = showcase_path.read_text(encoding="utf-8")
                examples = showcase.get("examples", [])
                if not examples:
                    errors.append("capability showcase has no examples")
                if showcase.get("example_count") != len(examples):
                    errors.append("capability showcase example_count differs from examples")
                if examples and not any(item.get("is_fully_correct") is True for item in examples):
                    errors.append("capability showcase has no fully correct example")
                if examples and not any(item.get("is_fully_correct") is False for item in examples):
                    errors.append("capability showcase has no error or boundary example")
                required_example_fields = (
                    "source_record_id",
                    "input_artifacts",
                    "display_artifacts",
                    "expected_output",
                    "model_outputs",
                    "is_fully_correct",
                    "teaching_note",
                )
                for index, example in enumerate(examples, start=1):
                    missing = [
                        field
                        for field in required_example_fields
                        if field not in example or example[field] in (None, "", [], {})
                    ]
                    if missing:
                        errors.append(
                            f"capability example {index} missing fields: {missing}"
                        )
                        continue
                    source_record_id = str(example["source_record_id"])
                    if source_record_id not in showcase_text:
                        errors.append(
                            f"capability showcase does not reference record: {source_record_id}"
                        )
                    for field in ("input_artifacts", "display_artifacts"):
                        for relative in example[field]:
                            if not (root / relative).is_file():
                                errors.append(
                                    f"capability example {index} missing {field} path: {relative}"
                                )
                            if field == "display_artifacts" and Path(relative).name not in showcase_text:
                                errors.append(
                                    f"capability showcase does not reference display artifact: {relative}"
                                )
                showcase_verified = not any(
                    message.startswith("capability") for message in errors
                )
            except (OSError, json.JSONDecodeError, TypeError) as error:
                errors.append(f"invalid capability showcase data: {error}")

    test_plan_path = root / contract.get("test_plan", "")
    test_plan_text = test_plan_path.read_text(encoding="utf-8") if test_plan_path.is_file() else ""
    if "| 验收项编号 |" not in test_plan_text:
        errors.append("test plan does not use the canonical acceptance-item header")

    evidence_relative = contract.get("evidence_index")
    if contract.get("quantitative_results") and not evidence_relative:
        errors.append("quantitative result package has no evidence_index")
    elif evidence_relative and (root / evidence_relative).is_file():
        try:
            evidence = json.loads((root / evidence_relative).read_text(encoding="utf-8"))
            result_dir = root / contract["result_directory"]
            for entry in evidence.get("metrics", []):
                metric_id = entry.get("id", "<unknown>")
                required_metric_fields = (
                    "value",
                    "source_file",
                    "json_path",
                    "evaluation_scope",
                    "sample_count",
                    "sample_count_source_file",
                    "sample_count_json_paths",
                    "sample_count_aggregation",
                    "acceptance_item_id",
                )
                missing = [field for field in required_metric_fields if field not in entry]
                if missing:
                    errors.append(f"metric {metric_id} missing fields: {missing}")
                    continue
                source_path = result_dir / entry["source_file"]
                if not source_path.is_file():
                    errors.append(f"metric {metric_id} missing source: {entry['source_file']}")
                    continue
                source = json.loads(source_path.read_text(encoding="utf-8"))
                try:
                    source_value = resolve_json_path(source, entry["json_path"])
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"metric {metric_id} invalid source path: {error}")
                    continue
                if source_value != entry["value"]:
                    errors.append(f"metric {metric_id} value differs from source")
                if entry["acceptance_item_id"] not in test_plan_text:
                    errors.append(
                        f"metric {metric_id} references undefined acceptance item: "
                        f"{entry['acceptance_item_id']}"
                    )
                if "test_gate" in entry:
                    errors.append(f"metric {metric_id} uses a non-canonical field")
                count_source_path = result_dir / entry["sample_count_source_file"]
                if not count_source_path.is_file():
                    errors.append(f"metric {metric_id} missing sample-count source")
                else:
                    count_source = json.loads(count_source_path.read_text(encoding="utf-8"))
                    try:
                        counts = [
                            int(resolve_json_path(count_source, path))
                            for path in entry["sample_count_json_paths"]
                        ]
                        if entry["sample_count_aggregation"] == "value":
                            expected_count = counts[0]
                        elif entry["sample_count_aggregation"] == "sum":
                            expected_count = sum(counts)
                        else:
                            raise ValueError("unsupported sample-count aggregation")
                        if expected_count != entry["sample_count"]:
                            errors.append(f"metric {metric_id} sample count differs from source")
                    except (KeyError, TypeError, ValueError, IndexError) as error:
                        errors.append(f"metric {metric_id} invalid sample-count source: {error}")
            for entry in evidence.get("entries", []):
                metric_id = entry.get("claim_id", "<unknown>")
                required_metric_fields = (
                    "value",
                    "source_file",
                    "field_path",
                    "scope",
                    "sample_count",
                    "acceptance_item_id",
                )
                missing = [field for field in required_metric_fields if field not in entry]
                if missing:
                    errors.append(f"metric {metric_id} missing fields: {missing}")
                    continue
                source_path = (root / contract["result_directory"]) / entry["source_file"]
                if not source_path.is_file():
                    errors.append(f"metric {metric_id} missing source: {entry['source_file']}")
                    continue
                source_value: Any = json.loads(source_path.read_text(encoding="utf-8"))
                try:
                    for key in entry["field_path"].split("."):
                        source_value = source_value[key]
                    if isinstance(source_value, dict) and "value" in source_value:
                        source_value = source_value["value"]
                except (KeyError, TypeError) as error:
                    errors.append(f"metric {metric_id} invalid source path: {error}")
                    continue
                if source_value != entry["value"]:
                    errors.append(f"metric {metric_id} value differs from source")
                if entry["acceptance_item_id"] not in test_plan_text:
                    errors.append(
                        f"metric {metric_id} references undefined acceptance item: "
                        f"{entry['acceptance_item_id']}"
                    )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid evidence index: {error}")

    glossary_relative = contract.get("metric_glossary")
    if schema_version >= 2 and contract.get("quantitative_results") and glossary_relative:
        glossary_path = root / glossary_relative
        if glossary_path.is_file():
            glossary = glossary_path.read_text(encoding="utf-8")
            for term in contract.get("glossary_required_terms", []):
                if term not in glossary:
                    errors.append(f"metric glossary missing required term: {term}")
            glossary_name = Path(glossary_relative).name
            for field in ("summary_document", "result_readme", "capability_showcase"):
                relative = contract.get(field)
                path = root / relative if relative else None
                if path and path.is_file() and glossary_name not in path.read_text(encoding="utf-8"):
                    errors.append(f"{field} does not link metric glossary: {glossary_name}")

    for capability in contract.get("capability_interfaces", []):
        for field in ("output_path", "implementation", "return_fields", "code_reference"):
            if not capability.get(field):
                errors.append(f"capability interface missing field: {field}")
        code_reference = capability.get("code_reference")
        if code_reference and not (root / code_reference).is_file():
            errors.append(f"missing capability code reference: {code_reference}")

    summary_result = validate_summary_contract(root, contract)
    errors.extend(summary_result["errors"])

    return {
        "path": str(contract_path.relative_to(root)),
        "work_package": contract.get("work_package"),
        "capability_showcase_verified": showcase_verified,
        "summary_verified": summary_result["verified"],
        "verified": not errors,
        "errors": errors,
    }


def parse_data_ledger(text: str) -> list[dict[str, Any]]:
    matches = list(LEDGER_HEADING.finditer(text))
    entries: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.end() : end]
        fields: dict[str, str] = {}
        for line in section.splitlines():
            if not line.startswith("- ") or "：" not in line:
                continue
            key, value = line[2:].split("：", 1)
            fields[key.strip()] = value.strip()
        entries.append(
            {
                "issue_id": match.group(1),
                "title": match.group(2).strip(),
                "fields": fields,
            }
        )
    return entries


def validate_project_governance(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    required = {
        "agent_entry": root / "AGENTS.md",
        "specification": root / "最小系统研发规格与约束.md",
        "roadmap": root / "docs" / "最小系统开发路线图.md",
        "memory_protocol": root / "docs" / "项目记忆与工作包检查机制.md",
        "data_ledger": root / "docs" / "数据充分性与补充台账.md",
        "environment": root / "docs" / "开发环境说明.md",
        "work_package_template": root / "docs" / "工作包文档与交付模板.md",
        "iteration_protocol": root / "docs" / "迭代与发布管理机制.md",
        "iteration_test_plan": root / "docs" / "迭代与发布管理测试计划.md",
        "teaching_risk_ledger": root / "docs" / "教学使用风险台账.md",
        "p2_w04_teaching_decision": root / "docs" / "P2-W04_教学验收决定.md",
        "p2_w04_teaching_acceptance": root
        / "results"
        / "P2-W04_tiny_image_ldm"
        / "teaching_acceptance.json",
    }
    existence = {name: path.is_file() for name, path in required.items()}
    texts = {
        name: path.read_text(encoding="utf-8")
        for name, path in required.items()
        if path.is_file()
    }
    ledger_entries = parse_data_ledger(texts.get("data_ledger", ""))
    all_data_heading_ids = [
        match.group(1)
        for match in LEDGER_ANY_DATA_HEADING.finditer(texts.get("data_ledger", ""))
    ]
    malformed_ledger_heading_ids = [
        issue_id
        for issue_id in all_data_heading_ids
        if not re.fullmatch(r"DATA-[A-Z]+-\d{3}", issue_id)
    ]
    issue_ids = [entry["issue_id"] for entry in ledger_entries]
    duplicate_ids = sorted(
        issue_id for issue_id in set(issue_ids) if issue_ids.count(issue_id) > 1
    )
    incomplete = {
        entry["issue_id"]: [
            field for field in REQUIRED_LEDGER_FIELDS if not entry["fields"].get(field)
        ]
        for entry in ledger_entries
        if any(not entry["fields"].get(field) for field in REQUIRED_LEDGER_FIELDS)
    }
    open_without_work_package = [
        entry["issue_id"]
        for entry in ledger_entries
        if entry["fields"].get("状态") in {"开放", "处理中"}
        and not re.search(r"P[0-4]-W\d{2}", entry["fields"].get("影响工作包", ""))
    ]
    open_without_deadline_node = [
        entry["issue_id"]
        for entry in ledger_entries
        if entry["fields"].get("状态") in {"开放", "处理中"}
        and not re.search(r"P[0-4]-W\d{2}", entry["fields"].get("最迟处理节点", ""))
    ]

    specification = texts.get("specification", "")
    roadmap = texts.get("roadmap", "")
    protocol = texts.get("memory_protocol", "")
    agent_entry = texts.get("agent_entry", "")
    work_package_statuses = [
        {"work_package": match.group(1), "status": match.group(2)}
        for match in WORK_PACKAGE_STATUS.finditer(roadmap)
    ]
    in_progress = [
        item["work_package"]
        for item in work_package_statuses
        if item["status"] == "进行中"
    ]
    test_plans = sorted((root / "docs").glob("P*-W*_测试计划.md"))
    noncanonical_test_plan_headers = [
        str(path.relative_to(root))
        for path in test_plans
        if "| 验收项编号 |" not in path.read_text(encoding="utf-8")
    ]
    status_documents: dict[str, str] = {}
    noncanonical_document_statuses: dict[str, str] = {}
    for path in sorted((root / "docs").glob("*.md")):
        match = re.search(r"^> 状态：(.+?)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            status = match.group(1).strip()
            status_documents[str(path.relative_to(root))] = status
            if status not in CANONICAL_STATUSES:
                noncanonical_document_statuses[str(path.relative_to(root))] = status
    markdown_paths = [root / "AGENTS.md", root / "最小系统研发规格与约束.md"]
    markdown_paths.extend(sorted((root / "docs").rglob("*.md")))
    markdown_paths.extend(sorted((root / "results").rglob("*.md")))
    ambiguous_term_locations = [
        str(path.relative_to(root))
        for path in markdown_paths
        if path.is_file()
        and any(
            term in path.read_text(encoding="utf-8")
            for term in ("关口", "门槛")
        )
    ]
    artifact_contracts = [
        validate_artifact_contract(root, path)
        for path in sorted((root / "results").rglob("artifact_contract.json"))
    ]
    p2_w04_result = root / "results/P2-W04_tiny_image_ldm"
    try:
        iteration_management = verify_all(root, p2_w04_result, verify_files=False)
    except (IterationManagementError, OSError, json.JSONDecodeError) as error:
        iteration_management = {"verified": False, "errors": [str(error)]}
    try:
        teaching_acceptance = json.loads(
            required["p2_w04_teaching_acceptance"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        teaching_acceptance = {"verified": False, "error": str(error)}
    checks = {
        "required_files_exist": all(existence.values()),
        "agent_entry_links_memory_protocol": "docs/项目记忆与工作包检查机制.md" in agent_entry,
        "agent_entry_links_data_ledger": "docs/数据充分性与补充台账.md" in agent_entry,
        "agent_entry_requires_confirmation_node": "未经教师确认，不启动下一工作包" in agent_entry,
        "specification_links_memory_protocol": "docs/项目记忆与工作包检查机制.md" in specification,
        "specification_links_data_ledger": "docs/数据充分性与补充台账.md" in specification,
        "protocol_has_terminology_standard": "## 2. 统一术语与标识" in protocol,
        "protocol_has_recovery_check": "## 4. 恢复检查" in protocol,
        "protocol_has_before_check": "## 5. 事前检查" in protocol,
        "protocol_has_during_check": "## 6. 事中检查" in protocol,
        "protocol_has_after_check": "## 7. 事后检查" in protocol,
        "protocol_defines_authority_map": "## 3. 记忆地图与权威来源" in protocol,
        "protocol_has_four_way_consistency": "## 8. 四向一致性核对" in protocol,
        "protocol_has_retrospective_rules": "## 12. 本轮问题复盘与制度化修正" in protocol,
        "protocol_requires_capability_showcase": "每个模型必须提供能力样例册及机器可读样例清单" in protocol,
        "template_requires_capability_showcase": "## 4. 能力样例册最低结构" in texts.get("work_package_template", ""),
        "protocol_requires_formal_experiment_showcase": "每次正式模型实验" in protocol
        and "capability_examples.md/.json" in protocol,
        "protocol_requires_metric_glossary": "指标与术语说明" in protocol
        and "是否进入`is_fully_correct`" in protocol,
        "template_requires_metric_glossary": "指标与术语说明最低结构" in texts.get(
            "work_package_template", ""
        ),
        "protocol_defines_canonical_summary": "当前工作包总报告" in protocol
        and "迭代总结报告" in protocol,
        "template_requires_complete_summary": all(
            heading in texts.get("work_package_template", "")
            for heading in ("直观能力样例", "与当前代表性系统的差距", "主要证据路径")
        ),
        "iteration_protocol_is_linked": "docs/迭代与发布管理机制.md" in specification
        and "docs/迭代与发布管理机制.md" in agent_entry,
        "iteration_protocol_defines_separate_pointers": all(
            term in texts.get("iteration_protocol", "")
            for term in ("最近完成迭代", "当前候选", "正式发布", "source_snapshot.zip")
        ),
        "p2_w04_iteration_registry_verified": bool(iteration_management.get("verified")),
        "p2_w04_candidate_and_release_match_decision": iteration_management.get(
            "last_completed_iteration"
        ) == "I004"
        and (iteration_management.get("current_candidate") or {}).get("iteration_id")
        == "I004"
        and (iteration_management.get("current_candidate") or {}).get("branch_id")
        == "control"
        and iteration_management.get("official_release") is None,
        "p2_w04_teaching_acceptance_is_bounded": teaching_acceptance.get(
            "work_package_status"
        ) == "已验收"
        and teaching_acceptance.get("acceptance_type")
        == "bounded_teaching_acceptance"
        and teaching_acceptance.get("strict_blocking_items_all_passed") is False
        and teaching_acceptance.get("official_release_permitted") is False
        and teaching_acceptance.get("official_release") is None,
        "ledger_has_entries": len(ledger_entries) >= 6,
        "ledger_heading_ids_are_canonical": not malformed_ledger_heading_ids
        and len(all_data_heading_ids) == len(ledger_entries),
        "ledger_ids_unique": not duplicate_ids,
        "ledger_fields_complete": not incomplete,
        "open_items_have_work_packages": not open_without_work_package,
        "open_items_have_deadline_nodes": not open_without_deadline_node,
        "roadmap_has_work_package_statuses": bool(work_package_statuses),
        "roadmap_has_at_most_one_in_progress_work_package": len(in_progress) <= 1,
        "p2_w01_contains_data_readiness_check": "P2-W01 数据就绪检查" in roadmap,
        "roadmap_has_p2_w03_accepted": bool(
            re.search(r"\| `P2-W03` \| 已验收 \|", roadmap)
        ),
        "roadmap_has_p2_w04_accepted": bool(
            re.search(r"\| `P2-W04` \| 已验收 \|", roadmap)
        ),
        "roadmap_has_p2_w05_accepted": bool(
            re.search(r"\| `P2-W05` \| 已验收 \|", roadmap)
        ),
        "test_plan_headers_are_canonical": not noncanonical_test_plan_headers,
        "document_statuses_are_canonical": not noncanonical_document_statuses,
        "ambiguous_term_absent": not ambiguous_term_locations,
        "artifact_contract_exists": bool(artifact_contracts),
        "artifact_contracts_verified": bool(artifact_contracts)
        and all(item["verified"] for item in artifact_contracts),
        "artifact_showcases_verified": bool(artifact_contracts)
        and all(item["capability_showcase_verified"] for item in artifact_contracts),
        "artifact_summaries_verified": bool(artifact_contracts)
        and all(item["summary_verified"] for item in artifact_contracts),
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "required_files": {name: str(path.relative_to(root)) for name, path in required.items()},
        "required_file_existence": existence,
        "ledger_entry_count": len(ledger_entries),
        "open_ledger_entry_count": sum(
            entry["fields"].get("状态") == "开放" for entry in ledger_entries
        ),
        "ledger_issue_ids": issue_ids,
        "all_data_heading_ids": all_data_heading_ids,
        "malformed_ledger_heading_ids": malformed_ledger_heading_ids,
        "duplicate_issue_ids": duplicate_ids,
        "incomplete_ledger_entries": incomplete,
        "open_without_work_package": open_without_work_package,
        "open_without_deadline_node": open_without_deadline_node,
        "work_package_statuses": work_package_statuses,
        "in_progress_work_packages": in_progress,
        "test_plan_count": len(test_plans),
        "noncanonical_test_plan_headers": noncanonical_test_plan_headers,
        "document_statuses": status_documents,
        "noncanonical_document_statuses": noncanonical_document_statuses,
        "ambiguous_term_locations": ambiguous_term_locations,
        "artifact_contracts": artifact_contracts,
        "iteration_management": iteration_management,
        "teaching_acceptance": teaching_acceptance,
    }
