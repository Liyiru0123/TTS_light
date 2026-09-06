"""Persistent train-only expansion and latent/text datasets for P2-W04."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mini_multimodal.data.artifacts import load_dataset_bundle
from mini_multimodal.tasks.reasoning import position_region
from mini_multimodal.tasks.text_zh import ChineseTextAdapter
from mini_multimodal.training import load_checkpoint
from mini_multimodal.vae.model import TinyImageVAE
from mini_multimodal.vision.renderer import render_scene
from mini_multimodal.world.sampling import SamplerConfig, SceneSampler
from mini_multimodal.world.serialization import scene_to_dict


SPECIAL_TOKENS = ("<pad>", "<null>", "<unk>")
SEMANTIC_TOKENS = (
    "红色", "蓝色", "绿色", "黄色", "紫色", "橙色",
    "圆形", "正方形", "三角形", "矩形", "五边形",
    "中等大小", "左上区域", "上方区域", "右上区域", "左侧区域",
    "中央区域", "右侧区域", "左下区域", "下方区域", "右下区域",
    "1个", "2个", "3个",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _object_records(scene: Any) -> list[dict[str, Any]]:
    return [
        {
            "object_id": obj.object_id,
            "color": obj.color.value,
            "shape": obj.shape.value,
            "size": obj.size.value,
            "position_region": position_region(obj.position_at(0)).value,
            "position": [obj.position_at(0).x, obj.position_at(0).y],
        }
        for obj in scene.objects
    ]


def _color_shape_pair(obj: Mapping[str, Any]) -> str:
    return f"{obj['color']}|{obj['shape']}"


def _load_holdout_policy(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, Path | None]:
    relative = config.get("data", {}).get("holdout_policy")
    if not relative:
        return None, None
    path = root / str(relative)
    policy = json.loads(path.read_text(encoding="utf-8"))
    authority = root / policy["authority"]["path"]
    authority_manifest = json.loads(authority.read_text(encoding="utf-8"))
    authoritative_pairs = authority_manifest["compositional_report"]["holdout_pairs"]
    if sorted(policy["holdout_pairs"]) != sorted(authoritative_pairs):
        raise ValueError("holdout policy differs from the P1 authority manifest")
    all_pairs = {
        f"{color}|{shape}"
        for color in policy["colors"]
        for shape in policy["shapes"]
    }
    if set(policy["allowed_training_pairs"]) | set(policy["holdout_pairs"]) != all_pairs:
        raise ValueError("holdout policy does not partition the full color-shape space")
    if set(policy["allowed_training_pairs"]) & set(policy["holdout_pairs"]):
        raise ValueError("holdout and allowed color-shape pairs overlap")
    return policy, path


def _load_vae(root: Path, checkpoint: str) -> TinyImageVAE:
    config = json.loads((root / "configs" / "p2_w02_image_vae.json").read_text(encoding="utf-8"))
    model = TinyImageVAE(**{
        key: config["model"][key]
        for key in ("base_channels", "latent_channels", "logvar_min", "logvar_max")
    })
    load_checkpoint(root / checkpoint, model=model, optimizer=None, restore_rng=False)
    model.eval()
    return model


def prepare_p2_w04_data(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = root / config["output_data"]
    output.mkdir(parents=True, exist_ok=True)
    extension_dir = output / "extension_images"
    extension_dir.mkdir(parents=True, exist_ok=True)

    index_dir = root / config["source_index"]
    index_manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    holdout_policy, holdout_policy_path = _load_holdout_policy(root, config)
    holdout_pairs = set() if holdout_policy is None else set(holdout_policy["holdout_pairs"])
    allowed_pairs = None if holdout_policy is None else set(holdout_policy["allowed_training_pairs"])
    filter_p1_train = bool(config["data"].get("filter_p1_train_records_with_holdouts", False))
    bundle = load_dataset_bundle(root / index_manifest["sources"]["dataset"]["path"])
    scenes = {scene.scene_id: scene for scene in bundle.all_scenes}
    index_rows = [
        json.loads(line)
        for line in (index_dir / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    latent_path = root / config["source_latents"]
    source_cache = np.load(latent_path, allow_pickle=False)
    source_latents = source_cache["posterior_mean"].astype(np.float32)
    latent_by_scene = {
        str(scene_id): source_latents[index]
        for index, scene_id in enumerate(source_cache["scene_ids"].tolist())
    }
    source_cache.close()
    latent_manifest = json.loads((latent_path.parent / "manifest.json").read_text(encoding="utf-8"))
    scaling_factor = float(latent_manifest["scaling_factor"])

    records: list[dict[str, Any]] = []
    latent_values: list[np.ndarray] = []
    source_image_hashes: dict[str, set[str]] = {name: set() for name in ("train", "validation", "test")}
    source_original_counts: Counter[str] = Counter()
    excluded_original_rows: list[dict[str, Any]] = []
    for row in index_rows:
        scene = scenes[row["scene_id"]]
        prompt = row["generation_conditions"]["image"]["text_zh"]
        objects = _object_records(scene)
        source_original_counts[row["split"]] += 1
        source_image_hashes[row["split"]].add(row["assets"]["image"]["sha256"])
        holdout_hits = [_color_shape_pair(obj) for obj in objects if _color_shape_pair(obj) in holdout_pairs]
        if row["split"] == "train" and filter_p1_train and holdout_hits:
            excluded_original_rows.append({
                "record_id": f"original:{row['scene_id']}",
                "scene_id": row["scene_id"],
                "variant": row["variant"],
                "holdout_pairs": holdout_hits,
                "reason": "contains_holdout_color_shape_pair",
            })
            continue
        records.append({
            "record_id": f"original:{row['scene_id']}",
            "scene_id": row["scene_id"],
            "split": row["split"],
            "variant": row["variant"],
            "source": "p1_frozen",
            "prompt": prompt,
            "image_path": row["assets"]["image"]["path"],
            "image_sha256": row["assets"]["image"]["sha256"],
            "objects": objects,
            "composition_key": row.get("composition_key"),
        })
        latent_values.append(latent_by_scene[row["scene_id"]])

    adapter = ChineseTextAdapter()
    sampler = SceneSampler(
        int(config["data"]["extension_seed"]),
        SamplerConfig(min_objects=1, max_objects=3),
    )
    extension_count = int(config["data"]["extension_count"])
    extension_rows: list[dict[str, Any]] = []
    image_tensors: list[torch.Tensor] = []
    image_paths: list[Path] = []
    blocked_hashes = source_image_hashes["validation"] | source_image_hashes["test"]
    accepted_hashes: set[str] = set()
    candidate_index = 0
    rejected_duplicate_count = 0
    rejected_holdout_count = 0
    extension_id_prefix = str(config["data"].get("extension_id_prefix", "ldmext_scene_"))
    while len(extension_rows) < extension_count:
        accepted_index = len(extension_rows)
        scene = replace(
            sampler.sample(candidate_index),
            scene_id=f"{extension_id_prefix}{accepted_index:06d}",
        )
        candidate_index += 1
        scene_objects = _object_records(scene)
        if any(_color_shape_pair(obj) in holdout_pairs for obj in scene_objects):
            rejected_holdout_count += 1
            continue
        rendered = render_scene(scene, frame_index=0)
        buffer = io.BytesIO()
        rendered.image.save(buffer, format="PNG", optimize=False, compress_level=9)
        image_bytes = buffer.getvalue()
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest in blocked_hashes or digest in accepted_hashes:
            rejected_duplicate_count += 1
            continue
        accepted_hashes.add(digest)
        path = extension_dir / f"{scene.scene_id}.png"
        path.write_bytes(image_bytes)
        relative = path.relative_to(root).as_posix()
        extension_rows.append({
            "record_id": f"extension:{scene.scene_id}",
            "scene_id": scene.scene_id,
            "split": "train",
            "variant": "extension_train_only",
            "source": "p2_w04_extension",
            "prompt": adapter.describe_scene(scene, 0),
            "image_path": relative,
            "image_sha256": digest,
            "objects": scene_objects,
            "composition_key": None,
            "scene": scene_to_dict(scene),
        })
        image_tensors.append(
            torch.from_numpy(np.array(rendered.image, dtype=np.uint8, copy=True))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )
        image_paths.append(path)

    vae = _load_vae(root, str(config["source_vae_checkpoint"]))
    with torch.no_grad():
        for start in range(0, len(image_tensors), 64):
            batch = torch.stack(image_tensors[start : start + 64])
            mean, _ = vae.encode(batch)
            latent_values.extend(mean.cpu().numpy())
    records.extend(extension_rows)

    latents = np.stack(latent_values).astype(np.float16)
    scene_ids = np.asarray([row["scene_id"] for row in records])
    splits = np.asarray([row["split"] for row in records])
    np.savez_compressed(
        output / "latents.npz",
        posterior_mean=latents,
        scene_ids=scene_ids,
        splits=splits,
    )
    _write_jsonl(output / "records.jsonl", records)
    characters = sorted({char for row in records if row["split"] == "train" for char in row["prompt"]})
    vocabulary = {
        "tokens": list(SPECIAL_TOKENS) + list(SEMANTIC_TOKENS)
        + [char for char in characters if char not in SEMANTIC_TOKENS]
    }
    vocabulary["token_to_id"] = {token: index for index, token in enumerate(vocabulary["tokens"])}
    _write_json(output / "vocabulary.json", vocabulary)

    train_rows = [row for row in records if row["split"] == "train"]
    color_shape = Counter(
        f"{obj['color']}|{obj['shape']}" for row in extension_rows for obj in row["objects"]
    )
    object_counts = Counter(len(row["objects"]) for row in extension_rows)
    regions = Counter(obj["position_region"] for row in extension_rows for obj in row["objects"])
    extension_hashes = {row["image_sha256"] for row in extension_rows}
    cross_split_hash_overlap = {
        split: len(extension_hashes & source_image_hashes[split])
        for split in ("validation", "test")
    }
    original_counts = Counter(row["split"] for row in records if row["source"] == "p1_frozen")
    all_counts = Counter(row["split"] for row in records)
    final_train_holdout = Counter(
        _color_shape_pair(obj)
        for row in train_rows
        for obj in row["objects"]
        if _color_shape_pair(obj) in holdout_pairs
    )
    expected_total_train = int(
        config["data"].get("expected_total_train_count", len(train_rows))
    )
    expected_retained_p1 = int(
        config["data"].get("expected_retained_p1_train_count", original_counts.get("train", 0))
    )
    expected_excluded_p1 = int(
        config["data"].get("expected_excluded_p1_train_count", len(excluded_original_rows))
    )
    original_manifest_digest = _sha(index_dir / "manifest.json")
    report = {
        "extension_count": len(extension_rows),
        "original_split_counts": dict(original_counts),
        "source_original_split_counts": dict(source_original_counts),
        "excluded_original_train_count": len(excluded_original_rows),
        "combined_split_counts": dict(all_counts),
        "unique_record_ids": len({row["record_id"] for row in records}),
        "unique_scene_ids": len({row["scene_id"] for row in records}),
        "color_shape_coverage_count": len(color_shape),
        "color_shape_counts": dict(sorted(color_shape.items())),
        "object_count_coverage": {str(key): value for key, value in sorted(object_counts.items())},
        "position_region_coverage_count": len(regions),
        "position_region_counts": dict(sorted(regions.items())),
        "extension_unique_image_hash_count": len(extension_hashes),
        "rejected_duplicate_candidate_count": rejected_duplicate_count,
        "rejected_holdout_candidate_count": rejected_holdout_count,
        "scanned_candidate_count": candidate_index,
        "extension_validation_test_hash_overlap": cross_split_hash_overlap,
        "source_index_manifest_sha256_before": original_manifest_digest,
        "source_index_manifest_sha256_after": _sha(index_dir / "manifest.json"),
        "latent_shape": list(latents.shape),
        "latent_scaling_factor": scaling_factor,
        "checks": {
            "extension_count": len(extension_rows) == extension_count,
            "combined_counts": dict(all_counts) == {
                "train": expected_total_train,
                "validation": int(source_original_counts["validation"]),
                "test": int(source_original_counts["test"]),
            },
            "retained_p1_train_count": int(original_counts.get("train", 0)) == expected_retained_p1,
            "excluded_p1_train_count": len(excluded_original_rows) == expected_excluded_p1,
            "identity_unique": len(records) == len({row["record_id"] for row in records}) == len({row["scene_id"] for row in records}),
            "all_color_shape_pairs": (
                len(color_shape) == 30
                if allowed_pairs is None
                else set(color_shape) == allowed_pairs
            ),
            "training_holdout_object_count_zero": sum(final_train_holdout.values()) == 0,
            "all_object_counts": set(object_counts) == {1, 2, 3},
            "all_position_regions": len(regions) == 9,
            "no_extension_eval_hash_overlap": all(value == 0 for value in cross_split_hash_overlap.values()),
            "source_manifest_unchanged": original_manifest_digest == _sha(index_dir / "manifest.json"),
        },
    }
    report["verified"] = all(report["checks"].values())
    holdout_audit = None
    if holdout_policy is not None:
        excluded_pair_counts = Counter(
            pair for row in excluded_original_rows for pair in row["holdout_pairs"]
        )
        holdout_audit = {
            "policy_id": holdout_policy["policy_id"],
            "policy_path": str(holdout_policy_path.relative_to(root).as_posix()),
            "policy_sha256": _sha(holdout_policy_path),
            "holdout_pairs": sorted(holdout_pairs),
            "allowed_training_pairs": sorted(allowed_pairs or set()),
            "source_p1_train_count": int(source_original_counts["train"]),
            "retained_p1_train_count": int(original_counts.get("train", 0)),
            "excluded_p1_train_count": len(excluded_original_rows),
            "excluded_p1_holdout_object_count": sum(excluded_pair_counts.values()),
            "excluded_p1_holdout_pair_counts": dict(sorted(excluded_pair_counts.items())),
            "extension_count": len(extension_rows),
            "extension_holdout_object_count": 0,
            "final_training_record_count": len(train_rows),
            "final_training_holdout_object_count": sum(final_train_holdout.values()),
            "final_training_pair_counts": dict(sorted(Counter(
                _color_shape_pair(obj) for row in train_rows for obj in row["objects"]
            ).items())),
            "excluded_records": excluded_original_rows,
            "checks": {
                "authority_matches": True,
                "source_p1_immutable": original_manifest_digest == _sha(index_dir / "manifest.json"),
                "expected_excluded_count": len(excluded_original_rows) == expected_excluded_p1,
                "all_holdouts_absent_from_training": sum(final_train_holdout.values()) == 0,
                "all_allowed_pairs_present": set(color_shape) == (allowed_pairs or set()),
            },
        }
        holdout_audit["verified"] = all(holdout_audit["checks"].values())
        _write_json(output / "holdout_audit.json", holdout_audit)
    _write_json(output / "data_readiness_report.json", report)

    managed = ["records.jsonl", "vocabulary.json", "latents.npz", "data_readiness_report.json"]
    if holdout_audit is not None:
        managed.append("holdout_audit.json")
    manifest = {
        "artifact_version": 1,
        "source_index": config["source_index"],
        "source_index_manifest_sha256": original_manifest_digest,
        "source_latents": config["source_latents"],
        "source_latents_sha256": _sha(latent_path),
        "source_vae_checkpoint": config["source_vae_checkpoint"],
        "source_vae_checkpoint_sha256": _sha(root / config["source_vae_checkpoint"]),
        "holdout_policy": None if holdout_policy_path is None else holdout_policy_path.relative_to(root).as_posix(),
        "holdout_policy_sha256": None if holdout_policy_path is None else _sha(holdout_policy_path),
        "latent_scaling_factor": scaling_factor,
        "files": {
            name: {"bytes": (output / name).stat().st_size, "sha256": _sha(output / name)}
            for name in managed
        },
        "extension_image_count": extension_count,
        "extension_image_tree_sha256": hashlib.sha256(
            "".join(row["image_sha256"] for row in extension_rows).encode("ascii")
        ).hexdigest(),
        "statistics": report,
    }
    _write_json(output / "manifest.json", manifest)
    return report


def verify_p2_w04_data(project_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    directory = Path(data_dir)
    if not directory.is_absolute():
        directory = root / directory
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or _sha(path) != expected["sha256"]:
            raise ValueError(f"P2-W04 data file differs from manifest: {name}")
    if _sha(root / manifest["source_index"] / "manifest.json") != manifest["source_index_manifest_sha256"]:
        raise ValueError("P2-W04 source index manifest changed")
    report = json.loads((directory / "data_readiness_report.json").read_text(encoding="utf-8"))
    if not report.get("verified"):
        raise ValueError("P2-W04 data readiness report is not verified")
    return report


class CharacterTokenizer:
    def __init__(self, vocabulary: Mapping[str, Any]) -> None:
        self.tokens = tuple(str(token) for token in vocabulary["tokens"])
        self.token_to_id = {str(key): int(value) for key, value in vocabulary["token_to_id"].items()}
        if self.tokens[:3] != SPECIAL_TOKENS:
            raise ValueError("unexpected special-token order")
        self.semantic_tokens = tuple(
            sorted(
                (token for token in self.tokens if len(token) > 1 and not token.startswith("<")),
                key=len,
                reverse=True,
            )
        )

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def null_id(self) -> int:
        return self.token_to_id["<null>"]

    def encode(self, text: str, max_length: int) -> list[int]:
        values: list[int] = []
        cursor = 0
        while cursor < len(text):
            token = next(
                (candidate for candidate in self.semantic_tokens if text.startswith(candidate, cursor)),
                text[cursor],
            )
            values.append(self.token_to_id.get(token, self.token_to_id["<unk>"]))
            cursor += len(token)
        return values[:max_length]


class LatentTextDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        project_root: str | Path,
        *,
        data_dir: str | Path,
        split: str | None = None,
        record_ids: Sequence[str] | None = None,
        max_text_length: int = 128,
    ) -> None:
        self.root = Path(project_root).resolve()
        directory = Path(data_dir)
        self.data_dir = directory if directory.is_absolute() else self.root / directory
        self.rows = [
            json.loads(line)
            for line in (self.data_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        selected = None if record_ids is None else set(record_ids)
        self.rows = [
            row for row in self.rows
            if (split is None or row["split"] == split)
            and (selected is None or row["record_id"] in selected)
        ]
        if not self.rows:
            raise ValueError("latent-text dataset selected no records")
        payload = np.load(self.data_dir / "latents.npz", allow_pickle=False)
        self.latents = payload["posterior_mean"].astype(np.float32)
        scene_ids = [str(value) for value in payload["scene_ids"].tolist()]
        payload.close()
        self.latent_index = {scene_id: index for index, scene_id in enumerate(scene_ids)}
        vocabulary = json.loads((self.data_dir / "vocabulary.json").read_text(encoding="utf-8"))
        self.tokenizer = CharacterTokenizer(vocabulary)
        self.max_text_length = int(max_text_length)
        self.scaling_factor = float(
            json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8"))["latent_scaling_factor"]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        ids = self.tokenizer.encode(row["prompt"], self.max_text_length)
        foreground = torch.zeros((1, 16, 16), dtype=torch.float32)
        size_radius = {"small": 1, "medium": 2, "large": 3}
        for obj in row["objects"]:
            x = round(2.75 + float(obj["position"][0]) * 10.5)
            y = round(2.75 + float(obj["position"][1]) * 10.5)
            radius = size_radius[obj["size"]]
            foreground[:, max(0, y - radius) : min(16, y + radius + 1), max(0, x - radius) : min(16, x + radius + 1)] = 1.0
        return {
            "latent": torch.from_numpy(self.latents[self.latent_index[row["scene_id"]]]).mul(self.scaling_factor),
            "text_ids": torch.tensor(ids, dtype=torch.long),
            "foreground_latent_mask": foreground,
            "metadata": row,
        }


def collate_latent_text(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty latent-text batch")
    width = max(int(sample["text_ids"].numel()) for sample in samples)
    text_ids = torch.zeros((len(samples), width), dtype=torch.long)
    text_mask = torch.zeros((len(samples), width), dtype=torch.bool)
    for index, sample in enumerate(samples):
        length = int(sample["text_ids"].numel())
        text_ids[index, :length] = sample["text_ids"]
        text_mask[index, :length] = True
    return {
        "latent": torch.stack([sample["latent"] for sample in samples]),
        "text_ids": text_ids,
        "text_mask": text_mask,
        "foreground_latent_mask": torch.stack([sample["foreground_latent_mask"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }
