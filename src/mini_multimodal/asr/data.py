"""Data views, character tokens, padding and masks for TinyASR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import spec_augment


_TRANSLATION = str.maketrans({"1": "一", "2": "二", "3": "三"})
_REMOVED = frozenset("。，；、！？,.!?;:： \t\r\n")


def normalize_transcript(text: str) -> str:
    """Normalize digits and remove punctuation not pronounced by the corpus TTS."""
    return "".join(char for char in text.translate(_TRANSLATION) if char not in _REMOVED)


class CharacterCTCTokenizer:
    """Closed character vocabulary with CTC blank fixed at index zero."""

    def __init__(self, vocabulary: dict | Sequence[str]):
        tokens = vocabulary["tokens"] if isinstance(vocabulary, dict) else list(vocabulary)
        if not tokens or tokens[0] != "<blank>" or len(tokens) != len(set(tokens)):
            raise ValueError("CTC vocabulary must be unique and begin with <blank>")
        self.tokens = tuple(tokens)
        self._ids = {token: index for index, token in enumerate(self.tokens)}

    @property
    def blank_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        normalized = normalize_transcript(text)
        missing = sorted(set(normalized) - self._ids.keys())
        if missing:
            raise ValueError(f"Transcript contains out-of-vocabulary characters: {missing}")
        return [self._ids[char] for char in normalized]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.tokens[index] for index in ids if index != self.blank_id)


class ASRDataset(Dataset):
    def __init__(self, root: str | Path, *, data_dir: str | Path, record_ids: Sequence[str], augmentation: dict | None = None, augmentation_seed: int = 0):
        self.root = Path(root)
        self.data_dir = self.root / data_dir
        wanted = set(record_ids)
        rows = [json.loads(line) for line in (self.data_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        by_id = {row["record_id"]: row for row in rows}
        missing = wanted - by_id.keys()
        if missing:
            raise ValueError(f"Unknown ASR record ids: {sorted(missing)[:3]}")
        self.rows = [by_id[record_id] for record_id in record_ids]
        self.tokenizer = CharacterCTCTokenizer(json.loads((self.data_dir / "vocabulary.json").read_text(encoding="utf-8")))
        self.augmentation = dict(augmentation) if augmentation else None
        self.augmentation_seed = int(augmentation_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        feature = np.load(self.root / row["logmel_path"], allow_pickle=False).astype(np.float32).T
        feature = (feature - feature.mean(axis=0, keepdims=True)) / np.maximum(feature.std(axis=0, keepdims=True), 1e-5)
        if self.augmentation:
            feature = spec_augment(feature, record_id=row["record_id"], epoch=self.epoch, seed=self.augmentation_seed, **{key: self.augmentation[key] for key in ("probability", "frequency_mask_count", "frequency_mask_max_width", "time_mask_count", "time_mask_max_width")})
        target = self.tokenizer.encode(row["normalized_transcript"])
        return {
            "record_id": row["record_id"], "logmel": torch.from_numpy(feature),
            "target": torch.tensor(target, dtype=torch.long), "transcript": row["normalized_transcript"],
            "pool": row["pool"], "variant": row["variant"], "duration_seconds": row["duration_seconds"],
        }


def collate_asr(items: Sequence[dict]) -> dict:
    if not items:
        raise ValueError("Cannot collate an empty ASR batch")
    lengths = torch.tensor([item["logmel"].shape[0] for item in items], dtype=torch.long)
    target_lengths = torch.tensor([item["target"].numel() for item in items], dtype=torch.long)
    padded = torch.zeros(len(items), int(lengths.max()), items[0]["logmel"].shape[1], dtype=torch.float32)
    padding_mask = torch.ones(len(items), int(lengths.max()), dtype=torch.bool)
    for index, item in enumerate(items):
        length = item["logmel"].shape[0]
        padded[index, :length] = item["logmel"]
        padding_mask[index, :length] = False
    return {
        "record_ids": [item["record_id"] for item in items], "logmels": padded,
        "input_lengths": lengths, "input_padding_mask": padding_mask,
        "targets": torch.cat([item["target"] for item in items]), "target_lengths": target_lengths,
        "transcripts": [item["transcript"] for item in items],
        "pools": [item["pool"] for item in items], "variants": [item["variant"] for item in items],
    }
