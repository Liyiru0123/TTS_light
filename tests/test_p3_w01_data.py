from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.video_mllm import VideoQADataset, verify_temporal_challenge  # noqa: E402


DATA = ROOT / "data/video_qa/p3_w01_v1"


def test_data_vidqa_001_balanced_fixed_counts() -> None:
    audit = json.loads((DATA / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["record_count"] == 1740
    assert audit["split_counts"] == {"train": 1200, "validation": 240, "test": 300}
    assert audit["question_kind_counts_by_split"]["train"] == {
        "appear_order": 300,
        "disappear_order": 300,
        "motion_direction": 300,
        "move_start_order": 300,
    }
    for split in ("train", "validation", "test"):
        answers = audit["answer_counts_by_split"][split]
        assert answers["前者更早"] == answers["后者更早"] == answers["同时发生"]


def test_data_vidqa_002_no_split_or_p1_source_leakage() -> None:
    result = verify_temporal_challenge(ROOT, DATA)
    assert result["verified"]
    audit = json.loads((DATA / "data_audit.json").read_text(encoding="utf-8"))
    assert set(audit["record_id_overlap_counts"].values()) == {0}
    assert set(audit["video_sha256_overlap_counts"].values()) == {0}
    assert audit["source_p1_modified"] is False
    assert audit["p1_color_shape_compositional_claim"] is False


def test_data_vidqa_003_text_is_decidable_without_answer_leak() -> None:
    audit = json.loads((DATA / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["answer_string_leak_count"] == 0
    assert audit["unknown_text_count_by_split"] == {}
    rows = [json.loads(line) for line in (DATA / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        descriptors = [item["descriptor_zh"] for item in row["objects"]]
        assert len(descriptors) == len(set(descriptors))
        assert row["answer_zh"] not in row["question_zh"]


def test_data_vidqa_004_reversal_pairs_are_exact_and_opposite() -> None:
    rows = [json.loads(line) for line in (DATA / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    with np.load(DATA / "videos.npz", allow_pickle=False) as cache:
        ids = [str(value) for value in cache["record_ids"].tolist()]
        frames = cache["frames"]
        index = {record_id: position for position, record_id in enumerate(ids)}
        checked = set()
        opposites = {
            "向左移动": "向右移动",
            "向右移动": "向左移动",
            "向上移动": "向下移动",
            "向下移动": "向上移动",
        }
        for row in rows:
            if not row["pair_id"] or row["pair_id"] in checked:
                continue
            checked.add(row["pair_id"])
            partner = next(item for item in rows if item["record_id"] == row["partner_record_id"])
            assert opposites[row["answer_zh"]] == partner["answer_zh"]
            assert np.array_equal(frames[index[row["record_id"]]][::-1], frames[index[partner["record_id"]]])
    assert len(checked) == 174


def test_data_vidqa_005_dataset_returns_valid_tensors() -> None:
    dataset = VideoQADataset(ROOT, data_dir=DATA, splits="test")
    sample = dataset[0]
    assert tuple(sample["video"].shape) == (8, 3, 64, 64)
    assert sample["video"].dtype.is_floating_point
    assert 0.0 <= float(sample["video"].min()) <= float(sample["video"].max()) <= 1.0
    assert sample["question_ids"].ndim == sample["answer_input_ids"].ndim == 1


def test_data_vidqa_006_overfit_subset_covers_all_question_kinds() -> None:
    record_ids = json.loads((DATA / "overfit_ids_balanced.json").read_text(encoding="utf-8"))
    rows = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (DATA / "records.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    assert len(record_ids) == len(set(record_ids)) == 32
    selected = [rows[record_id] for record_id in record_ids]
    assert all(row["split"] == "train" for row in selected)
    assert Counter(row["question_kind"] for row in selected) == {
        "motion_direction": 8,
        "appear_order": 8,
        "disappear_order": 8,
        "move_start_order": 8,
    }
