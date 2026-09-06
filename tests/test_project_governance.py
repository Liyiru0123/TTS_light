from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.governance import (  # noqa: E402
    REQUIRED_LEDGER_FIELDS,
    parse_data_ledger,
    validate_project_governance,
)


class ProjectGovernanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validate_project_governance(ROOT)
        cls.ledger_text = (ROOT / "docs" / "数据充分性与补充台账.md").read_text(
            encoding="utf-8"
        )
        cls.entries = parse_data_ledger(cls.ledger_text)

    def test_mem_001_required_memory_files_and_authority_map_exist(self) -> None:
        self.assertTrue(self.result["checks"]["required_files_exist"])
        self.assertTrue(self.result["checks"]["protocol_defines_authority_map"])
        self.assertTrue(self.result["checks"]["agent_entry_links_memory_protocol"])
        self.assertTrue(self.result["checks"]["agent_entry_links_data_ledger"])
        self.assertTrue(self.result["checks"]["agent_entry_requires_confirmation_node"])

    def test_cycle_001_recovery_before_during_and_after_checks_exist(self) -> None:
        for key in (
            "protocol_has_recovery_check",
            "protocol_has_before_check",
            "protocol_has_during_check",
            "protocol_has_after_check",
        ):
            with self.subTest(key=key):
                self.assertTrue(self.result["checks"][key])

    def test_ledger_001_issue_ids_are_unique_and_fields_are_complete(self) -> None:
        self.assertTrue(self.result["checks"]["ledger_heading_ids_are_canonical"])
        self.assertFalse(self.result["malformed_ledger_heading_ids"])
        self.assertEqual(
            len(self.result["all_data_heading_ids"]),
            len(self.entries),
        )
        ids = [entry["issue_id"] for entry in self.entries]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 6)
        for entry in self.entries:
            with self.subTest(issue_id=entry["issue_id"]):
                self.assertTrue(
                    all(entry["fields"].get(field) for field in REQUIRED_LEDGER_FIELDS)
                )

    def test_ledger_002_open_items_have_work_packages_and_deadline_nodes(self) -> None:
        self.assertFalse(self.result["open_without_work_package"])
        self.assertFalse(self.result["open_without_deadline_node"])

    def test_route_001_work_package_state_is_unambiguous(self) -> None:
        for key in (
            "roadmap_has_work_package_statuses",
            "roadmap_has_at_most_one_in_progress_work_package",
            "p2_w01_contains_data_readiness_check",
            "roadmap_has_p2_w03_accepted",
            "roadmap_has_p2_w04_accepted",
            "roadmap_has_p2_w05_accepted",
        ):
            with self.subTest(key=key):
                self.assertTrue(self.result["checks"][key])

    def test_terms_001_terminology_headers_and_statuses_are_canonical(self) -> None:
        for key in (
            "protocol_has_terminology_standard",
            "test_plan_headers_are_canonical",
            "document_statuses_are_canonical",
            "ambiguous_term_absent",
        ):
            with self.subTest(key=key):
                self.assertTrue(self.result["checks"][key])
        self.assertFalse(self.result["noncanonical_test_plan_headers"])
        self.assertFalse(self.result["noncanonical_document_statuses"])
        self.assertFalse(self.result["ambiguous_term_locations"])

    def test_artifact_001_contract_evidence_and_interfaces_are_verified(self) -> None:
        self.assertTrue(self.result["checks"]["protocol_has_four_way_consistency"])
        self.assertTrue(self.result["checks"]["protocol_has_retrospective_rules"])
        self.assertTrue(self.result["checks"]["artifact_contract_exists"])
        self.assertTrue(self.result["checks"]["artifact_contracts_verified"])
        for contract in self.result["artifact_contracts"]:
            with self.subTest(work_package=contract["work_package"]):
                self.assertTrue(contract["verified"], contract["errors"])

    def test_showcase_001_capability_examples_are_required_and_verified(self) -> None:
        for key in (
            "protocol_requires_capability_showcase",
            "template_requires_capability_showcase",
            "artifact_showcases_verified",
        ):
            with self.subTest(key=key):
                self.assertTrue(self.result["checks"][key])
        for contract in self.result["artifact_contracts"]:
            with self.subTest(work_package=contract["work_package"]):
                self.assertTrue(contract["capability_showcase_verified"])

    def test_docs_001_formal_runs_and_metric_definitions_are_required(self) -> None:
        for key in (
            "protocol_requires_formal_experiment_showcase",
            "protocol_requires_metric_glossary",
            "template_requires_metric_glossary",
        ):
            with self.subTest(key=key):
                self.assertTrue(self.result["checks"][key])

    def test_summary_001_current_summary_is_complete_and_candidate_aligned(self) -> None:
        for key in (
            "protocol_defines_canonical_summary",
            "template_requires_complete_summary",
            "artifact_summaries_verified",
        ):
            with self.subTest(key=key):
                self.assertTrue(self.result["checks"][key])
        for contract in self.result["artifact_contracts"]:
            with self.subTest(work_package=contract["work_package"]):
                self.assertTrue(contract["summary_verified"], contract["errors"])

    def test_iteration_001_registry_candidate_and_release_are_governed(self) -> None:
        for key in (
            "iteration_protocol_is_linked",
            "iteration_protocol_defines_separate_pointers",
            "p2_w04_iteration_registry_verified",
            "p2_w04_candidate_and_release_match_decision",
            "p2_w04_teaching_acceptance_is_bounded",
        ):
            with self.subTest(key=key):
                self.assertTrue(self.result["checks"][key])
        self.assertEqual(self.result["iteration_management"]["last_completed_iteration"], "I004")
        self.assertEqual(
            self.result["iteration_management"]["current_candidate"]["iteration_id"],
            "I004",
        )
        self.assertEqual(
            self.result["iteration_management"]["current_candidate"]["branch_id"],
            "control",
        )
        self.assertIsNone(self.result["iteration_management"]["official_release"])

    def test_auto_001_full_governance_validation_passes(self) -> None:
        self.assertTrue(self.result["verified"])


if __name__ == "__main__":
    unittest.main()
