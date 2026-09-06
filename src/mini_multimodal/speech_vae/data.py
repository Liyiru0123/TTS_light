"""Leakage-safe fixed Log-Mel windows inherited from source utterance splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def export_window_manifest(
    root: str | Path, *, source_dir: str, output_dir: str,
    window_frames: int, hop_frames: int, minimum_valid_frames: int,
) -> dict:
    root = Path(root)
    source = root / source_dir
    rows = [json.loads(line) for line in (source / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    windows, split_sources = [], {name: set() for name in ("train", "validation", "test")}
    coverage_failures = []
    for row in rows:
        utterance_id = row["annotation"]["utterance_id"]
        frames = int(row["annotation"]["logmel_shape"][1])
        split = row["split"]
        split_sources[split].add(utterance_id)
        starts = list(range(0, max(1, frames - minimum_valid_frames + 1), hop_frames))
        if not starts or starts[-1] + window_frames < frames:
            starts.append(max(0, frames - window_frames))
        starts = sorted(set(starts))
        coverage = np.zeros(frames, dtype=np.int16)
        for index, start in enumerate(starts):
            valid = min(window_frames, frames - start)
            if valid < minimum_valid_frames:
                continue
            coverage[start : start + valid] += 1
            windows.append({
                "window_id": f"{hashlib.sha256(utterance_id.encode()).hexdigest()[:16]}_w{index:02d}",
                "source_utterance_id": utterance_id, "scene_id": row["annotation"]["scene_id"],
                "split": split, "pool": row["pool"], "variant": row["variant"],
                "source_logmel_path": f"{source_dir}/{row['logmel_path']}",
                "source_audio_path": f"{source_dir}/{row['audio_path']}",
                "text": row["annotation"]["text"], "start_frame": start,
                "valid_frames": valid, "source_frames": frames,
            })
        if not np.all(coverage > 0):
            coverage_failures.append(utterance_id)
    output = root / output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "window_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in windows),
        encoding="utf-8", newline="\n",
    )
    split_counts = Counter(row["split"] for row in windows)
    intersections = {
        "train_validation": len(split_sources["train"] & split_sources["validation"]),
        "train_test": len(split_sources["train"] & split_sources["test"]),
        "validation_test": len(split_sources["validation"] & split_sources["test"]),
    }
    audit = {
        "schema_version": 1, "work_package": "P3-W02", "source_utterance_count": len(rows),
        "window_count": len(windows), "split_window_counts": dict(split_counts),
        "split_source_counts": {key: len(value) for key, value in split_sources.items()},
        "source_split_intersections": intersections, "coverage_failure_count": len(coverage_failures),
        "minimum_valid_frames": min(row["valid_frames"] for row in windows),
        "maximum_valid_frames": max(row["valid_frames"] for row in windows),
        "window_frames": window_frames, "hop_frames": hop_frames,
    }
    _write_json(output / "data_audit.json", audit)
    _write_json(output / "manifest.json", {
        "schema_version": 1, "dataset_id": "p3_w02_v1", "source_dir": source_dir,
        "window_manifest": "window_manifest.jsonl", "statistics": audit,
    })
    return audit


class SpeechWindowDataset(Dataset):
    def __init__(self, root: str | Path, *, data_dir: str, splits: str | Sequence[str], config: dict):
        self.root = Path(root); self.data_dir = self.root / data_dir; self.config = config
        wanted = {splits} if isinstance(splits, str) else set(splits)
        all_rows = [json.loads(line) for line in (self.data_dir / "window_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        self.rows = [row for row in all_rows if row["split"] in wanted]
        if not self.rows: raise ValueError("speech window selection is empty")

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]; feature = np.load(self.root / row["source_logmel_path"], allow_pickle=False).astype(np.float32)
        frames = int(self.config["window"]["frames"]); valid = int(row["valid_frames"]); start = int(row["start_frame"])
        n_mels = int(self.config["feature"]["n_mels"])
        if feature.ndim != 2 or feature.shape[0] != n_mels:
            raise ValueError(f"expected {n_mels} Mel channels, got {feature.shape}")
        window = np.full((n_mels, frames), float(self.config["window"]["padding_value"]), dtype=np.float32)
        window[:, :valid] = feature[:, start : start + valid]
        low, high = self.config["feature"]["minimum_logmel"], self.config["feature"]["maximum_logmel"]
        normalized = np.clip((window - low) / (high - low) * 2.0 - 1.0, -1.0, 1.0)
        mask = np.zeros(frames, dtype=np.bool_); mask[:valid] = True
        return {"window": torch.from_numpy(normalized)[None], "mask": torch.from_numpy(mask), "metadata": row}


def collate_speech_windows(items: Sequence[dict]) -> dict:
    return {"window": torch.stack([item["window"] for item in items]), "mask": torch.stack([item["mask"] for item in items]), "metadata": [item["metadata"] for item in items]}


def stitch_windows(values: Sequence[np.ndarray], starts: Sequence[int], valid_lengths: Sequence[int], total_frames: int) -> tuple[np.ndarray, np.ndarray]:
    if not values or not (len(values) == len(starts) == len(valid_lengths)): raise ValueError("window inputs must be non-empty and aligned")
    channels = values[0].shape[0]; total = np.zeros((channels, total_frames), np.float64); weight = np.zeros(total_frames, np.float64)
    for value, start, valid in zip(values, starts, valid_lengths):
        if value.ndim != 2 or value.shape[0] != channels or valid <= 0 or start < 0 or start + valid > total_frames: raise ValueError("invalid window for stitching")
        total[:, start:start+valid] += value[:, :valid]; weight[start:start+valid] += 1.0
    if np.any(weight == 0): raise ValueError("stitching left uncovered frames")
    return (total / weight[None]).astype(np.float32), weight.astype(np.float32)
