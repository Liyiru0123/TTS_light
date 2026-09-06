"""TinyT2V video-latent dataset (P3-W04 data/video/p3_w04_v1)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mini_multimodal.ldm.data import CharacterTokenizer


class VideoLatentDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: Path, data_dir: str | Path, split: str | None = None,
                 max_text_length: int = 128) -> None:
        self.root = Path(root).resolve()
        directory = Path(data_dir)
        self.data_dir = directory if directory.is_absolute() else self.root / directory
        self.rows = [json.loads(line)
                     for line in io.open(self.data_dir / "records.jsonl", encoding="utf-8")]
        if split is not None:
            self.rows = [row for row in self.rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no records for split {split!r}")
        with io.open(self.data_dir / "vocabulary.json", encoding="utf-8") as f:
            self.tokenizer = CharacterTokenizer(json.load(f))
        self.max_text_length = int(max_text_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        latent = np.load(self.data_dir / row["latent_path"], allow_pickle=False)["latent"]
        ids = self.tokenizer.encode(row["prompt"], self.max_text_length)
        # GT trajectory of the first object (for auxiliary position supervision)
        trajectory = row["objects"][0].get("trajectory", [])
        position_gt = torch.tensor(trajectory[:8], dtype=torch.float32) if trajectory else torch.zeros(8, 2)
        if position_gt.shape != (8, 2):
            position_gt = torch.zeros(8, 2)
        return {
            "latent": torch.from_numpy(latent.astype(np.float32)),  # [F, 4, 16, 16]
            "text_ids": torch.tensor(ids, dtype=torch.long),
            "position_gt": position_gt,
            "record_id": row["record_id"],
        }


def collate_video(samples: list[dict]) -> dict:
    width = max(int(sample["text_ids"].numel()) for sample in samples)
    batch = len(samples)
    text_ids = torch.zeros((batch, width), dtype=torch.long)
    text_mask = torch.zeros((batch, width), dtype=torch.bool)
    for index, sample in enumerate(samples):
        length = int(sample["text_ids"].numel())
        text_ids[index, :length] = sample["text_ids"]
        text_mask[index, :length] = True
    latent = torch.stack([sample["latent"] for sample in samples])  # [B, F, 4, 16, 16]
    # model expects [B, C, F, H, W]
    latent = latent.permute(0, 2, 1, 3, 4)
    return {
        "latent": latent,
        "text_ids": text_ids,
        "text_mask": text_mask,
        "position_gt": torch.stack([sample["position_gt"] for sample in samples]),
        "record_ids": [sample["record_id"] for sample in samples],
    }
