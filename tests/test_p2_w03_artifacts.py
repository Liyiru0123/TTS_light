from __future__ import annotations

import json
import hashlib
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.vlm import TinyVLM  # noqa: E402
from mini_multimodal.tools.build_p2_w03_evidence_index import (  # noqa: E402
    METRICS,
    SAMPLE_COUNT_SPECS,
    resolve_json_path,
)
RESULTS = ROOT / "results" / "P2-W03_tiny_vlm"
CONFIG = json.loads((ROOT / "configs" / "p2_w03_tiny_vlm.json").read_text(encoding="utf-8"))


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


class TinyVLMArtifactTests(unittest.TestCase):
    def test_artifact_vlm_001_all_experiment_gates_pass(self) -> None:
        summary = read_json("experiment_summary.json")
        self.assertTrue(summary["verified"])
        self.assertTrue(all(summary.values()))

    def test_artifact_vlm_002_primary_metrics_meet_frozen_thresholds(self) -> None:
        thresholds = CONFIG["thresholds"]
        classification = read_json("classification_metrics.json")
        generation = read_json("generation_metrics.json")
        self.assertGreaterEqual(
            classification["classification_accuracy"], thresholds["classification_accuracy"]
        )
        self.assertGreaterEqual(
            classification["grounding_relation_accuracy"],
            thresholds["grounding_relation_accuracy"],
        )
        self.assertGreaterEqual(generation["generation_accuracy"], thresholds["generation_accuracy"])
        self.assertGreaterEqual(
            min(
                row["generation_accuracy"]
                for row in generation["groups"]["question_kind"].values()
            ),
            thresholds["generation_kind_accuracy"],
        )

    def test_artifact_vlm_003_multimodal_and_ablation_evidence_pass(self) -> None:
        baseline = read_json("single_modality_report.json")
        ablation = read_json("ablation_report.json")
        self.assertTrue(baseline["verified"])
        self.assertTrue(ablation["verified"])
        self.assertGreaterEqual(
            baseline["advantage"], CONFIG["thresholds"]["multimodal_advantage"]
        )
        self.assertGreaterEqual(
            ablation["position_relation_drop"],
            CONFIG["thresholds"]["position_relation_drop"],
        )

    def test_artifact_vlm_004_evidence_files_are_nonempty_and_complete(self) -> None:
        expected = (
            "training_curves.png",
            "classification_confusion.png",
            "cross_attention_examples.png",
            "predictions.jsonl",
            "failure_examples.json",
            "generation_traces.json",
            "resource_benchmark.json",
            "reproducibility_report.json",
        )
        for name in expected:
            with self.subTest(name=name):
                self.assertGreater((RESULTS / name).stat().st_size, 100)
        self.assertEqual(
            sum(1 for _ in (RESULTS / "predictions.jsonl").open(encoding="utf-8")),
            1195,
        )

    def test_artifact_vlm_005_failed_attempts_remain_traceable(self) -> None:
        for index in range(1, 10):
            self.assertTrue((RESULTS / f"failed_attempt_{index:02d}.md").is_file())
        self.assertGreaterEqual(len(list((RESULTS / "failed_attempts").iterdir())), 9)

    def test_artifact_vlm_006_inference_bundle_is_self_describing_and_loadable(self) -> None:
        manifest = read_json("checkpoints/checkpoint_manifest.json")
        entry = manifest["primary_inference_bundle"]
        path = ROOT / entry["path"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        special = bundle["special_tokens"]
        model = TinyVLM(
            vocabulary_size=len(bundle["vocabulary"]["tokens"]),
            answer_class_count=len(bundle["answer_classes"]),
            pad_id=special["<pad>"],
            bos_id=special["<bos>"],
            eos_id=special["<eos>"],
            mask_id=special["<mask>"],
            target_marker_id=special["甲"],
            reference_marker_id=special["乙"],
            **bundle["model_config"],
        )
        model.load_state_dict(bundle["model_state"], strict=True)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 241381)

    def test_artifact_vlm_007_evidence_index_matches_source_results(self) -> None:
        index = read_json("evidence_index.json")
        self.assertEqual(len(index["metrics"]), len(METRICS))
        entries = {entry["id"]: entry for entry in index["metrics"]}
        self.assertEqual(len(entries), len(METRICS))
        for metric_id, _label, source_file, json_path, acceptance_item_id, evaluation_scope in METRICS:
            with self.subTest(metric_id=metric_id):
                entry = entries[metric_id]
                source = read_json(source_file)
                self.assertEqual(entry["source_file"], source_file)
                self.assertEqual(entry["json_path"], json_path)
                self.assertEqual(entry["acceptance_item_id"], acceptance_item_id)
                self.assertEqual(entry["evaluation_scope"], evaluation_scope)
                self.assertEqual(entry["value"], resolve_json_path(source, json_path))
                self.assertEqual(entry["display_4dp"], f'{entry["value"]:.4f}')
                count_file, count_paths, aggregation = SAMPLE_COUNT_SPECS[metric_id]
                count_source = read_json(count_file)
                count_values = [
                    int(resolve_json_path(count_source, path)) for path in count_paths
                ]
                expected_count = count_values[0] if aggregation == "value" else sum(count_values)
                self.assertEqual(entry["sample_count"], expected_count)
                self.assertEqual(entry["sample_count_source_file"], count_file)
                self.assertEqual(entry["sample_count_json_paths"], list(count_paths))
                self.assertEqual(entry["sample_count_aggregation"], aggregation)

        kind_accuracies = [
            row["generation_accuracy"]
            for row in read_json("generation_metrics.json")["groups"]["question_kind"].values()
        ]
        self.assertEqual(entries["generation_color"]["value"], min(kind_accuracies))


if __name__ == "__main__":
    unittest.main()
