"""Scene-level and QA-level views over the same four-modality assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import soundfile as sf
from PIL import Image

from mini_multimodal.data.artifacts import load_dataset_bundle
from mini_multimodal.data.schema import DatasetRecord
from mini_multimodal.video.artifacts import load_frames

from .cache import ByteLRUCache, CacheInfo
from .routing import route_question_kind


VALID_MODALITIES = frozenset({"text", "image", "speech", "video"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _as_values(values: str | Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


class UnifiedMultimodalDataset:
    """Load selected modalities without duplicating scene assets across QA rows."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        view: Literal["scene", "qa"] = "scene",
        pools: str | Iterable[str] | None = None,
        splits: str | Iterable[str] | None = None,
        variants: str | Iterable[str] | None = None,
        question_kinds: str | Iterable[str] | None = None,
        required_modality: str | None = None,
        modalities: Iterable[str] = ("text", "image", "speech", "video"),
        include_annotations: bool = False,
        cache: ByteLRUCache | None = None,
        cache_max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.index_dir = Path(index_dir).resolve()
        self.project_root = self.index_dir.parents[2]
        self.manifest = json.loads(
            (self.index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if view not in {"scene", "qa"}:
            raise ValueError("view must be 'scene' or 'qa'")
        self.view = view
        self.modalities = frozenset(str(value) for value in modalities)
        unknown = self.modalities - VALID_MODALITIES
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
        self.include_annotations = bool(include_annotations)
        self.cache = cache if cache is not None else ByteLRUCache(cache_max_bytes)
        selected_question_kinds = _as_values(question_kinds)
        if (selected_question_kinds is not None or required_modality is not None) and view != "qa":
            raise ValueError("question filters require view='qa'")
        if required_modality is not None and required_modality not in VALID_MODALITIES:
            raise ValueError(f"unknown required modality: {required_modality}")

        selected_pools = _as_values(pools)
        selected_splits = _as_values(splits)
        selected_variants = _as_values(variants)
        all_rows = _read_jsonl(self.index_dir / "scenes.jsonl")
        self.rows = [
            row
            for row in all_rows
            if (selected_pools is None or row["pool"] in selected_pools)
            and (selected_splits is None or row["split"] in selected_splits)
            and (selected_variants is None or row["variant"] in selected_variants)
        ]
        if not self.rows:
            raise ValueError("dataset filters selected no scenes")
        self._row_by_scene = {row["scene_id"]: row for row in self.rows}

        dataset_path = self.project_root / self.manifest["sources"]["dataset"]["path"]
        self.bundle = load_dataset_bundle(dataset_path)
        self._scene_by_id = {scene.scene_id: scene for scene in self.bundle.all_scenes}
        if set(self._row_by_scene) - set(self._scene_by_id):
            raise ValueError("index references scenes absent from the source dataset")
        selected_scene_ids = set(self._row_by_scene)
        self.records: list[DatasetRecord] = [
            record
            for record in self.bundle.all_records
            if record.scene_id in selected_scene_ids
            and (
                selected_question_kinds is None
                or record.task.question.kind.value in selected_question_kinds
            )
            and (
                required_modality is None
                or route_question_kind(record.task.question.kind).supports(required_modality)
            )
        ]
        if view == "qa" and not self.records:
            raise ValueError("question filters selected no records")
        self._annotation_rows: dict[str, list[dict[str, Any]]] | None = None

    def __len__(self) -> int:
        return len(self.rows) if self.view == "scene" else len(self.records)

    def _resolve(self, relative_path: str) -> Path:
        path = (self.project_root / relative_path).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as error:
            raise ValueError("asset path escapes the project root") from error
        return path

    def _array(self, key: str, loader: Any) -> np.ndarray:
        return self.cache.get_or_load(key, loader)

    def _load_image(self, path: Path) -> np.ndarray:
        def loader() -> np.ndarray:
            with Image.open(path) as source:
                return np.array(source.convert("RGB"), dtype=np.uint8, copy=True)

        return self._array(f"image:{path}", loader)

    def _load_waveform(self, path: Path, expected_rate: int) -> np.ndarray:
        def loader() -> np.ndarray:
            waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
            if sample_rate != expected_rate or waveform.ndim != 1:
                raise ValueError(f"invalid speech format: {path}")
            return np.asarray(waveform, dtype=np.float32)

        return self._array(f"waveform:{path}", loader)

    def _load_logmel(self, path: Path) -> np.ndarray:
        def loader() -> np.ndarray:
            value = np.load(path, allow_pickle=False)
            if value.ndim != 2:
                raise ValueError(f"invalid log-mel shape: {path}")
            return value.astype(np.float32, copy=False)

        return self._array(f"logmel:{path}", loader)

    def _load_video(self, path: Path) -> np.ndarray:
        return self._array(f"video:{path}", lambda: load_frames(path))

    def _ensure_annotations(self) -> dict[str, list[dict[str, Any]]]:
        if self._annotation_rows is None:
            sources = self.manifest["sources"]
            self._annotation_rows = {
                name: _read_jsonl(
                    self.project_root / sources[name]["path"] / "annotations.jsonl"
                )
                for name in ("image", "speech", "video")
            }
        return self._annotation_rows

    def _sample(self, row: dict[str, Any], record: DatasetRecord | None) -> dict[str, Any]:
        scene_id = row["scene_id"]
        sample: dict[str, Any] = {
            "metadata": {
                key: row[key]
                for key in (
                    "scene_id",
                    "pool",
                    "split",
                    "variant",
                    "cue_token",
                    "cue_aligned",
                    "composition_key",
                    "qa_record_ids",
                    "assets",
                )
            },
            "scene": self._scene_by_id[scene_id],
        }
        if "text" in self.modalities:
            conditions = row["generation_conditions"]
            sample["text"] = {
                "scene_description": conditions["image"]["text_zh"],
                "video_description": conditions["video"]["text_zh"],
                "generation_conditions": conditions,
            }
        if record is not None:
            sample["task"] = record.task
            sample["metadata"]["task_route"] = route_question_kind(
                record.task.question.kind
            ).to_dict()

        assets = row["assets"]
        if "image" in self.modalities:
            sample["image"] = self._load_image(self._resolve(assets["image"]["path"]))
        if "speech" in self.modalities:
            sample["waveform"] = self._load_waveform(
                self._resolve(assets["audio"]["path"]),
                int(assets["audio"]["sample_rate"]),
            )
            sample["logmel"] = self._load_logmel(
                self._resolve(assets["logmel"]["path"])
            )
        if "video" in self.modalities:
            sample["video"] = self._load_video(self._resolve(assets["video"]["path"]))

        if self.include_annotations:
            annotations = self._ensure_annotations()
            source_row = int(row["source_row"])
            sample["annotations"] = {
                name: annotations[name][source_row]["annotation"]
                for name in ("image", "speech", "video")
            }
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.view == "scene":
            row = self.rows[index]
            return self._sample(row, None)
        record = self.records[index]
        return self._sample(self._row_by_scene[record.scene_id], record)

    def cache_info(self) -> CacheInfo:
        return self.cache.info()
