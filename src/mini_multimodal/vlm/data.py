"""Grounded static-VQA view and frozen visual tokens for P2-W03."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from mini_multimodal.data.artifacts import load_dataset_bundle
from mini_multimodal.multimodal.routing import route_question_kind
from mini_multimodal.tasks.reasoning import object_by_id, position_region, visible_objects
from mini_multimodal.tasks.schema import QATask, QuestionKind
from mini_multimodal.tasks.text_zh import COLOR_ZH, REGION_ZH, SHAPE_ZH, SIZE_ZH
from mini_multimodal.world.schema import Scene, SceneObject


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<mask>")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _stable_key(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _attribute_descriptor(obj: SceneObject, kind: QuestionKind, region: str) -> str:
    if kind is QuestionKind.OBJECT_COLOR:
        return f"{SIZE_ZH[obj.size.value]}{SHAPE_ZH[obj.shape.value]}（位于{region}）"
    if kind is QuestionKind.OBJECT_SHAPE:
        return f"{COLOR_ZH[obj.color.value]}的{SIZE_ZH[obj.size.value]}对象（位于{region}）"
    if kind is QuestionKind.OBJECT_SIZE:
        return f"{COLOR_ZH[obj.color.value]}的{SHAPE_ZH[obj.shape.value]}（位于{region}）"
    raise ValueError(f"not an attribute question: {kind.value}")


def _relation_descriptor(obj: SceneObject) -> str:
    return f"{COLOR_ZH[obj.color.value]}的{SIZE_ZH[obj.size.value]}{SHAPE_ZH[obj.shape.value]}"


def _unique_descriptor(
    target: SceneObject,
    candidates: Sequence[SceneObject],
    describe: Any,
) -> str | None:
    target_text = str(describe(target))
    return target_text if sum(str(describe(obj)) == target_text for obj in candidates) == 1 else None


def render_grounded_question(scene: Scene, task: QATask) -> tuple[str | None, str | None]:
    """Render an answer-free question whose object references are visually decidable."""

    question = task.question
    if question.kind is QuestionKind.OBJECT_COUNT:
        return "图中有几个可见对象？", None
    if question.frame is None:
        return None, "static_question_without_frame"
    candidates = list(visible_objects(scene, question.frame))
    if not candidates:
        return None, "no_visible_objects"
    if question.target_object_id is None:
        return None, "missing_target_object"
    target = object_by_id(scene, question.target_object_id)

    if question.kind in {
        QuestionKind.OBJECT_COLOR,
        QuestionKind.OBJECT_SHAPE,
        QuestionKind.OBJECT_SIZE,
    }:
        if len(candidates) == 1:
            descriptor = "唯一的对象"
        else:
            describe = lambda obj: _attribute_descriptor(  # noqa: E731
                obj,
                question.kind,
                REGION_ZH[position_region(obj.position_at(question.frame)).value],
            )
            descriptor = _unique_descriptor(target, candidates, describe)
            if descriptor is None:
                return None, "ambiguous_attribute_reference"
        endings = {
            QuestionKind.OBJECT_COLOR: "是什么颜色？",
            QuestionKind.OBJECT_SHAPE: "是什么形状？",
            QuestionKind.OBJECT_SIZE: "的尺寸属于哪一类？",
        }
        return f"图中，目标甲是{descriptor}。甲{endings[question.kind]}", None

    if question.kind in {
        QuestionKind.HORIZONTAL_RELATION,
        QuestionKind.VERTICAL_RELATION,
    }:
        if question.reference_object_id is None:
            return None, "missing_reference_object"
        reference = object_by_id(scene, question.reference_object_id)
        target_text = _unique_descriptor(target, candidates, _relation_descriptor)
        reference_text = _unique_descriptor(reference, candidates, _relation_descriptor)
        if target_text is None or reference_text is None:
            return None, "ambiguous_relation_reference"
        axis = "水平方向" if question.kind is QuestionKind.HORIZONTAL_RELATION else "垂直方向"
        return (
            f"图中，目标甲是{target_text}，参照乙是{reference_text}。"
            f"甲相对乙在{axis}是什么关系？",
            None,
        )

    return None, "non_static_question"


def _source_rows(root: Path, source_index: str) -> tuple[dict[str, dict[str, Any]], Any]:
    index_dir = root / source_index
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = {
        row["scene_id"]: row
        for row in (
            json.loads(line)
            for line in (index_dir / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    bundle = load_dataset_bundle(root / manifest["sources"]["dataset"]["path"])
    return rows, bundle


def build_p2_w03_data_spec(
    project_root: str | Path,
    *,
    source_index: str,
    source_latents: str,
    overfit_count: int,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    rows, bundle = _source_rows(root, source_index)
    scene_by_id = {scene.scene_id: scene for scene in bundle.all_scenes}
    exclusions: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for source in bundle.all_records:
        route = route_question_kind(source.task.question.kind)
        if not route.supports("image"):
            continue
        scene = scene_by_id[source.scene_id]
        grounded, reason = render_grounded_question(scene, source.task)
        if grounded is None:
            exclusions[str(reason)] += 1
            continue
        answer = source.task.answer_zh
        if answer in grounded:
            exclusions["answer_string_leak"] += 1
            continue
        row = rows[source.scene_id]
        target = source.task.question.target_object_id
        target_position = None
        if target is not None and source.task.question.frame is not None:
            position = object_by_id(scene, target).position_at(source.task.question.frame)
            target_position = [position.x, position.y]
        reference = source.task.question.reference_object_id
        reference_position = None
        if reference is not None and source.task.question.frame is not None:
            position = object_by_id(scene, reference).position_at(source.task.question.frame)
            reference_position = [position.x, position.y]
        records.append({
            "record_id": source.record_id,
            "scene_id": source.scene_id,
            "split": source.split.value,
            "variant": source.variant.value,
            "pool": row["pool"],
            "question_kind": source.task.question.kind.value,
            "source_question_zh": source.task.question_zh,
            "question_zh": grounded,
            "answer_zh": answer,
            "target_object_id": target,
            "reference_object_id": source.task.question.reference_object_id,
            "target_position": target_position,
            "reference_position": reference_position,
            "object_count": len(scene.objects),
            "cue_token": source.cue_token,
            "cue_aligned": source.cue_aligned,
            "composition_key": source.composition_key,
        })

    train_records = [row for row in records if row["split"] == "train"]
    answer_classes = sorted({row["answer_zh"] for row in train_records})
    answer_to_class = {answer: index for index, answer in enumerate(answer_classes)}
    for row in records:
        if row["answer_zh"] not in answer_to_class:
            raise ValueError(f"unseen answer in {row['split']}: {row['answer_zh']}")
        row["answer_class"] = answer_to_class[row["answer_zh"]]

    characters = sorted({char for row in train_records for text in (row["question_zh"], row["answer_zh"]) for char in text})
    tokens = list(SPECIAL_TOKENS) + characters
    token_to_id = {token: index for index, token in enumerate(tokens)}
    unknown_by_split = Counter()
    for row in records:
        for text in (row["question_zh"], row["answer_zh"]):
            if any(char not in token_to_id for char in text):
                unknown_by_split[row["split"]] += 1

    kind_counts = Counter(row["question_kind"] for row in train_records)
    for row in records:
        row["sample_weight"] = len(train_records) / (len(kind_counts) * kind_counts[row["question_kind"]]) if row["split"] == "train" else 1.0

    chosen: list[str] = []
    by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_records:
        by_answer[row["answer_zh"]].append(row)
    for answer in sorted(by_answer):
        candidate = min(by_answer[answer], key=lambda row: _stable_key("overfit", row["record_id"]))
        chosen.append(candidate["record_id"])
    remaining = sorted(
        (row for row in train_records if row["record_id"] not in set(chosen)),
        key=lambda row: _stable_key("overfit-fill", row["record_id"]),
    )
    chosen.extend(row["record_id"] for row in remaining[: max(0, overfit_count - len(chosen))])
    overfit_ids = chosen[:overfit_count]

    statistics = {
        "source_static_qa_count": sum(
            route_question_kind(record.task.question.kind).supports("image")
            for record in bundle.all_records
        ),
        "retained_count": len(records),
        "excluded_count": sum(exclusions.values()),
        "exclusion_reason_counts": dict(sorted(exclusions.items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in records).items())),
        "variant_counts": dict(sorted(Counter(row["variant"] for row in records).items())),
        "question_kind_counts": dict(sorted(Counter(row["question_kind"] for row in records).items())),
        "multi_object_count": sum(row["object_count"] > 1 for row in records),
        "answer_class_count": len(answer_classes),
        "vocabulary_size": len(tokens),
        "max_question_tokens": max(len(row["question_zh"]) + 1 for row in records),
        "max_answer_tokens": max(len(row["answer_zh"]) + 1 for row in records),
        "unknown_text_count_by_split": dict(sorted(unknown_by_split.items())),
        "overfit_count": len(overfit_ids),
    }
    latent_manifest = json.loads((root / source_latents / "manifest.json").read_text(encoding="utf-8"))
    return {
        "records": records,
        "vocabulary": {"tokens": tokens, "token_to_id": token_to_id},
        "answer_classes": answer_classes,
        "overfit_ids": overfit_ids,
        "statistics": statistics,
        "source_index_manifest_sha256": _sha(root / source_index / "manifest.json"),
        "source_latent_manifest_sha256": _sha(root / source_latents / "manifest.json"),
        "source_latent_file_sha256": latent_manifest["files"]["posterior_means.npz"]["sha256"],
    }


def export_p2_w03_data(
    project_root: str | Path,
    output_dir: str | Path,
    *,
    source_index: str,
    source_latents: str,
    overfit_count: int,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)
    spec = build_p2_w03_data_spec(
        root,
        source_index=source_index,
        source_latents=source_latents,
        overfit_count=overfit_count,
    )
    records_path = destination / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in spec["records"]),
        encoding="utf-8",
        newline="\n",
    )
    _write_json(destination / "vocabulary.json", spec["vocabulary"])
    _write_json(destination / "answer_classes.json", spec["answer_classes"])
    _write_json(destination / "overfit_ids.json", spec["overfit_ids"])
    _write_json(destination / "data_statistics.json", spec["statistics"])
    files = {}
    for name in ("records.jsonl", "vocabulary.json", "answer_classes.json", "overfit_ids.json", "data_statistics.json"):
        path = destination / name
        files[name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    manifest = {
        "artifact_version": 1,
        "source_index": source_index,
        "source_latents": source_latents,
        "source_index_manifest_sha256": spec["source_index_manifest_sha256"],
        "source_latent_manifest_sha256": spec["source_latent_manifest_sha256"],
        "source_latent_file_sha256": spec["source_latent_file_sha256"],
        "files": files,
        "statistics": spec["statistics"],
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def verify_p2_w03_data(project_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    directory = Path(data_dir)
    if not directory.is_absolute():
        directory = root / directory
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or _sha(path) != expected["sha256"]:
            raise ValueError(f"P2-W03 data file differs from manifest: {name}")
    if _sha(root / manifest["source_index"] / "manifest.json") != manifest["source_index_manifest_sha256"]:
        raise ValueError("P2-W03 source index manifest differs")
    if _sha(root / manifest["source_latents"] / "manifest.json") != manifest["source_latent_manifest_sha256"]:
        raise ValueError("P2-W03 source latent manifest differs")
    latent_manifest = json.loads((root / manifest["source_latents"] / "manifest.json").read_text(encoding="utf-8"))
    if latent_manifest["files"]["posterior_means.npz"]["sha256"] != manifest["source_latent_file_sha256"]:
        raise ValueError("P2-W03 latent payload digest differs")
    return {"verified": True, "statistics": manifest["statistics"]}


class CharacterTokenizer:
    def __init__(self, vocabulary: Mapping[str, Any]) -> None:
        self.tokens = tuple(str(token) for token in vocabulary["tokens"])
        self.token_to_id = {str(token): int(index) for token, index in vocabulary["token_to_id"].items()}
        if tuple(self.tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError("special token order differs")

    def __len__(self) -> int:
        return len(self.tokens)

    def id(self, token: str) -> int:
        return self.token_to_id[token]

    def encode_question(self, text: str) -> list[int]:
        return [self.id("<cls>")] + [self.token_to_id.get(char, self.id("<unk>")) for char in text]

    def encode_answer(self, text: str) -> tuple[list[int], list[int]]:
        chars = [self.token_to_id.get(char, self.id("<unk>")) for char in text]
        return [self.id("<bos>")] + chars, chars + [self.id("<eos>")]

    def decode_answer(self, ids: Iterable[int]) -> str:
        result: list[str] = []
        for index in ids:
            token = self.tokens[int(index)]
            if token == "<eos>":
                break
            if token not in SPECIAL_TOKENS:
                result.append(token)
        return "".join(result)


class VLMQADataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        project_root: str | Path,
        *,
        data_dir: str | Path,
        source_latents: str | Path,
        splits: str | Iterable[str] | None = None,
        record_ids: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        directory = Path(data_dir)
        self.data_dir = directory if directory.is_absolute() else self.root / directory
        vocabulary = json.loads((self.data_dir / "vocabulary.json").read_text(encoding="utf-8"))
        self.tokenizer = CharacterTokenizer(vocabulary)
        selected_splits = None if splits is None else ({splits} if isinstance(splits, str) else set(splits))
        selected_ids = None if record_ids is None else set(record_ids)
        all_rows = [json.loads(line) for line in (self.data_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.rows = [
            row for row in all_rows
            if (selected_splits is None or row["split"] in selected_splits)
            and (selected_ids is None or row["record_id"] in selected_ids)
        ]
        if not self.rows:
            raise ValueError("VLM dataset filters selected no records")
        latent_dir = Path(source_latents)
        latent_path = (latent_dir if latent_dir.is_absolute() else self.root / latent_dir) / "posterior_means.npz"
        cache = np.load(latent_path, allow_pickle=False)
        scene_ids = [str(value) for value in cache["scene_ids"].tolist()]
        self.latents = cache["posterior_mean"].astype(np.float32)
        self.latent_index = {scene_id: index for index, scene_id in enumerate(scene_ids)}
        cache.close()
        missing = {row["scene_id"] for row in self.rows} - set(self.latent_index)
        if missing:
            raise ValueError(f"VLM records lack latent scenes: {len(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        question_ids = self.tokenizer.encode_question(row["question_zh"])
        answer_input, answer_target = self.tokenizer.encode_answer(row["answer_zh"])
        return {
            "visual": torch.from_numpy(self.latents[self.latent_index[row["scene_id"]]]),
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "answer_input_ids": torch.tensor(answer_input, dtype=torch.long),
            "answer_target_ids": torch.tensor(answer_target, dtype=torch.long),
            "answer_class": torch.tensor(row["answer_class"], dtype=torch.long),
            "sample_weight": torch.tensor(row["sample_weight"], dtype=torch.float32),
            "target_position": torch.tensor(
                row["target_position"] if row["target_position"] is not None else [float("nan"), float("nan")],
                dtype=torch.float32,
            ).mul(2.0).sub(1.0),
            "reference_position": torch.tensor(
                row["reference_position"] if row["reference_position"] is not None else [float("nan"), float("nan")],
                dtype=torch.float32,
            ).mul(2.0).sub(1.0),
            "metadata": row,
        }


def _pad(sequences: Sequence[torch.Tensor], pad_value: int) -> torch.Tensor:
    width = max(int(value.numel()) for value in sequences)
    result = torch.full((len(sequences), width), pad_value, dtype=sequences[0].dtype)
    for index, value in enumerate(sequences):
        result[index, : value.numel()] = value
    return result


def collate_vlm_qa(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate empty VLM batch")
    pad_id = 0
    return {
        "visual": torch.stack([sample["visual"] for sample in samples]),
        "question_ids": _pad([sample["question_ids"] for sample in samples], pad_id),
        "answer_input_ids": _pad([sample["answer_input_ids"] for sample in samples], pad_id),
        "answer_target_ids": _pad([sample["answer_target_ids"] for sample in samples], pad_id),
        "answer_class": torch.stack([sample["answer_class"] for sample in samples]),
        "sample_weight": torch.stack([sample["sample_weight"] for sample in samples]),
        "target_position": torch.stack([sample["target_position"] for sample in samples]),
        "reference_position": torch.stack([sample["reference_position"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }
