from __future__ import annotations

import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.evaluation_data import (  # noqa: E402
    verify_fixed_evaluation_data,
)


class P2W04EvaluationV2DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eval_dir = ROOT / "data" / "ldm" / "p2_w04_eval_v2"
        cls.manifest = verify_fixed_evaluation_data(ROOT, cls.eval_dir)
        cls.rows = [
            json.loads(line)
            for line in (cls.eval_dir / "records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        cls.interventions = [
            json.loads(line)
            for line in (cls.eval_dir / "interventions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        cls.policy = json.loads(
            (ROOT / "configs" / "compositional_holdouts_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_data_ldm_004_manifest_counts_and_hashes(self) -> None:
        self.assertTrue(self.manifest["verified"])
        self.assertEqual(self.manifest["record_count"], 420)
        self.assertEqual(self.manifest["subset_counts"], {
            "seen_single": 225,
            "unseen_single": 135,
            "multi_2": 30,
            "multi_3": 30,
        })
        self.assertEqual(self.manifest["intervention_count"], 96)
        self.assertEqual(self.manifest["source_image_unique_count"], 420)

    def test_data_ldm_005_seen_single_is_25_by_9_and_size_balanced(self) -> None:
        rows = [row for row in self.rows if row["subset"] == "seen_single"]
        pairs = Counter(row["composition_pair"] for row in rows)
        regions = Counter(row["objects"][0]["position_region"] for row in rows)
        sizes = Counter(row["objects"][0]["size"] for row in rows)
        self.assertEqual(set(pairs), set(self.policy["allowed_training_pairs"]))
        self.assertTrue(all(value == 9 for value in pairs.values()))
        self.assertTrue(all(value == 25 for value in regions.values()))
        self.assertEqual(sizes, {"small": 75, "medium": 75, "large": 75})

    def test_data_ldm_006_unseen_single_is_5_by_9_by_3(self) -> None:
        rows = [row for row in self.rows if row["subset"] == "unseen_single"]
        pairs = Counter(row["composition_pair"] for row in rows)
        regions = Counter(row["objects"][0]["position_region"] for row in rows)
        sizes = Counter(row["objects"][0]["size"] for row in rows)
        self.assertEqual(set(pairs), set(self.policy["holdout_pairs"]))
        self.assertTrue(all(value == 27 for value in pairs.values()))
        self.assertTrue(all(value == 15 for value in regions.values()))
        self.assertEqual(sizes, {"small": 45, "medium": 45, "large": 45})

    def test_data_ldm_007_multi_masks_do_not_overlap(self) -> None:
        payload = np.load(self.eval_dir / "ideal_masks.npz", allow_pickle=False)
        masks = payload["amodal"]
        payload.close()
        for row in self.rows:
            if not row["subset"].startswith("multi_"):
                continue
            item = masks[row["ideal_mask_index"]]
            count = row["object_count"]
            with self.subTest(record_id=row["record_id"]):
                self.assertTrue(all(item[index].any() for index in range(count)))
                for left in range(count):
                    for right in range(left + 1, count):
                        self.assertFalse(np.logical_and(item[left], item[right]).any())

    def test_data_ldm_008_interventions_change_only_target_attribute(self) -> None:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in self.interventions:
            groups[row["group_id"]].append(row)
        self.assertEqual(len(groups), 23)
        for group_id, rows in groups.items():
            attribute = rows[0]["changed_attribute"]
            expected_count = {"position": 3, "shape": 5, "color": 6}[attribute]
            self.assertEqual(len(rows), expected_count)
            objects = [row["objects"][0] for row in rows]
            changed_values = {
                "position": [tuple(obj["position"]) for obj in objects],
                "shape": [obj["shape"] for obj in objects],
                "color": [obj["color"] for obj in objects],
            }[attribute]
            with self.subTest(group_id=group_id):
                self.assertEqual(len(set(changed_values)), expected_count)
                for other in {"color", "shape", "size"} - {attribute}:
                    self.assertEqual(len({obj[other] for obj in objects}), 1)
                if attribute != "position":
                    self.assertEqual(len({tuple(obj["position"]) for obj in objects}), 1)
                self.assertEqual(len({row["shared_sample_seed"] for row in rows}), 1)

    def test_data_ldm_009_evaluation_assets_do_not_enter_training_view(self) -> None:
        train_rows = [
            json.loads(line)
            for line in (ROOT / "data" / "ldm" / "p2_w04_v2" / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if json.loads(line)["split"] == "train"
        ]
        train_ids = {row["scene_id"] for row in train_rows}
        train_hashes = {row["image_sha256"] for row in train_rows}
        self.assertFalse(train_ids & {row["scene_id"] for row in self.rows})
        self.assertFalse(train_hashes & {row["image_sha256"] for row in self.rows})


if __name__ == "__main__":
    unittest.main()

