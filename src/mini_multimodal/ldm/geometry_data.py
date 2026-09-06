"""Training-only foreground masks and calibration for P2-W04 attempt 3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .data import LatentTextDataset, collate_latent_text
from .inference import load_vae_decoder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hard_palette_foreground(image: np.ndarray, *, chroma_threshold: float = 0.1) -> np.ndarray:
    """Return colored object pixels while ignoring every grayscale cue background."""
    value = image.astype(np.float32)
    if value.max() > 1.0:
        value = value / 255.0
    chroma = value.max(axis=-1) - value.min(axis=-1)
    return chroma > float(chroma_threshold)


def soft_foreground_probability(image: torch.Tensor, *, threshold: float, temperature: float) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B,3,H,W]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    chroma = image.amax(dim=1, keepdim=True) - image.amin(dim=1, keepdim=True)
    return torch.sigmoid((chroma - float(threshold)) / float(temperature))


def build_geometry_masks(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source_dir = root / config["output_data"]
    output_dir = root / config["geometry_auxiliary"]["data_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = source_dir / "records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    threshold = float(config["geometry_auxiliary"]["source_chroma_threshold"])
    masks: list[np.ndarray] = []
    record_ids: list[str] = []
    areas: list[int] = []
    for row in rows:
        image = np.asarray(Image.open(root / row["image_path"]).convert("RGB"), dtype=np.uint8)
        if image.shape != (64, 64, 3):
            raise ValueError(f"unexpected source image shape: {row['record_id']}")
        mask = hard_palette_foreground(image, chroma_threshold=threshold)
        if not mask.any():
            raise ValueError(f"source image has no colored foreground: {row['record_id']}")
        masks.append(mask)
        record_ids.append(row["record_id"])
        areas.append(int(mask.sum()))
    asset = output_dir / "foreground_masks.npz"
    np.savez_compressed(asset, masks=np.stack(masks), record_ids=np.asarray(record_ids))
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "acceptance_item_id": "DATA-LDG-001",
        "record_count": len(record_ids),
        "unique_record_count": len(set(record_ids)),
        "mask_shape": [len(record_ids), 64, 64],
        "area_pixels": {
            "min": min(areas),
            "median": float(np.median(np.asarray(areas))),
            "max": max(areas),
        },
        "source_records": records_path.relative_to(root).as_posix(),
        "source_records_sha256": _sha(records_path),
        "source_manifest_sha256": _sha(source_dir / "manifest.json"),
        "source_chroma_threshold": threshold,
        "asset": asset.relative_to(root).as_posix(),
        "asset_sha256": _sha(asset),
        "checks": {
            "all_records_present": len(record_ids) == len(rows),
            "record_ids_unique": len(set(record_ids)) == len(rows),
            "all_masks_nonempty": min(areas) > 0,
        },
    }
    report["verified"] = all(report["checks"].values())
    _write_json(output_dir / "manifest.json", report)
    return report


class GeometryLatentTextDataset(LatentTextDataset):
    def __init__(self, project_root: str | Path, *, geometry_data_dir: str | Path, **kwargs: Any) -> None:
        super().__init__(project_root, **kwargs)
        directory = Path(geometry_data_dir)
        directory = directory if directory.is_absolute() else self.root / directory
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        asset = self.root / manifest["asset"]
        if _sha(asset) != manifest["asset_sha256"]:
            raise ValueError("geometry mask asset differs from manifest")
        payload = np.load(asset, allow_pickle=False)
        masks = payload["masks"].astype(np.bool_)
        ids = [str(value) for value in payload["record_ids"].tolist()]
        payload.close()
        lookup = {record_id: index for index, record_id in enumerate(ids)}
        missing = [row["record_id"] for row in self.rows if row["record_id"] not in lookup]
        if missing:
            raise ValueError(f"geometry mask asset misses records: {missing[:3]}")
        self.geometry_masks = masks
        self.geometry_index = lookup

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        record_id = self.rows[index]["record_id"]
        item["geometry_mask"] = torch.from_numpy(self.geometry_masks[self.geometry_index[record_id]].copy())[None].float()
        return item


def collate_geometry_text(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch = collate_latent_text(samples)
    batch["geometry_mask"] = torch.stack([sample["geometry_mask"] for sample in samples])
    return batch


@torch.no_grad()
def calibrate_soft_foreground(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    eval_dir = root / config["output_eval_data"]
    result_dir = root / config["output_results"]
    rows = [json.loads(line) for line in (eval_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    payload = np.load(eval_dir / "ideal_masks.npz", allow_pickle=False)
    target = torch.from_numpy(payload["visible"].any(axis=1).copy()).unsqueeze(1).float()
    payload.close()
    vae = load_vae_decoder(root, config["source_vae_checkpoint"])
    threshold = float(config["geometry_auxiliary"]["soft_chroma_threshold"])
    temperature = float(config["geometry_auxiliary"]["soft_chroma_temperature"])
    intersection = union = false_positive = background = foreground_hit = foreground = 0.0
    soft_dice_sum = 0.0
    for start in range(0, len(rows), 32):
        images = torch.stack([
            torch.from_numpy(np.asarray(Image.open(root / row["image_path"]).convert("RGB"), dtype=np.uint8).copy())
            .permute(2, 0, 1).float().div(255.0)
            for row in rows[start : start + 32]
        ])
        mean, _ = vae.encode(images)
        recon = vae.decode(mean).clamp(0, 1)
        probability = soft_foreground_probability(recon, threshold=threshold, temperature=temperature)
        truth = target[start : start + len(images)]
        predicted = probability >= 0.5
        truth_bool = truth.bool()
        intersection += float((predicted & truth_bool).sum())
        union += float((predicted | truth_bool).sum())
        false_positive += float((predicted & ~truth_bool).sum())
        background += float((~truth_bool).sum())
        foreground_hit += float((predicted & truth_bool).sum())
        foreground += float(truth_bool.sum())
        dice = (2 * (probability * truth).flatten(1).sum(1) + 1.0) / (
            probability.flatten(1).sum(1) + truth.flatten(1).sum(1) + 1.0
        )
        soft_dice_sum += float(dice.sum())
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "acceptance_item_id": "COMP-LDG-001",
        "sample_count": len(rows),
        "soft_chroma_threshold": threshold,
        "soft_chroma_temperature": temperature,
        "vae_mask_iou": intersection / union,
        "vae_soft_dice": soft_dice_sum / len(rows),
        "background_false_activation_rate": false_positive / background,
        "foreground_recall": foreground_hit / foreground,
        "vae_checkpoint": config["source_vae_checkpoint"],
        "vae_checkpoint_sha256": _sha(root / config["source_vae_checkpoint"]),
    }
    report["checks"] = {
        "mask_iou": report["vae_mask_iou"] >= 0.90,
        "background_false_activation": report["background_false_activation_rate"] <= 0.01,
        "foreground_recall": report["foreground_recall"] >= 0.90,
    }
    report["verified_for_training"] = all(report["checks"].values())
    _write_json(result_dir / "attempt3_geometry_calibration.json", report)
    return report

