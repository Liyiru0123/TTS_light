"""Deterministic temporal-conflict data for P3-W01 video question answering."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from mini_multimodal.video import render_video
from mini_multimodal.world import (
    Color,
    MotionDirection,
    MotionSpec,
    Position,
    Scene,
    SceneObject,
    Shape,
    Size,
)


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<mask>")
COLOR_ZH = {
    Color.RED: "红色",
    Color.BLUE: "蓝色",
    Color.GREEN: "绿色",
    Color.YELLOW: "黄色",
    Color.PURPLE: "紫色",
    Color.ORANGE: "橙色",
}
SHAPE_ZH = {
    Shape.CIRCLE: "圆形",
    Shape.SQUARE: "正方形",
    Shape.TRIANGLE: "三角形",
    Shape.RECTANGLE: "长方形",
    Shape.PENTAGON: "五边形",
}
SIZE_ZH = {Size.MEDIUM: "中等大小", Size.LARGE: "大"}
DIRECTION_ZH = {
    MotionDirection.LEFT: "向左移动",
    MotionDirection.RIGHT: "向右移动",
    MotionDirection.UP: "向上移动",
    MotionDirection.DOWN: "向下移动",
    MotionDirection.STATIC: "保持静止",
}
OPPOSITE_DIRECTION = {
    MotionDirection.LEFT: MotionDirection.RIGHT,
    MotionDirection.RIGHT: MotionDirection.LEFT,
    MotionDirection.UP: MotionDirection.DOWN,
    MotionDirection.DOWN: MotionDirection.UP,
}
ORDER_ANSWERS = ("前者更早", "后者更早", "同时发生")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _descriptor(obj: SceneObject) -> str:
    return f"{COLOR_ZH[obj.color]}的{SIZE_ZH[obj.size]}{SHAPE_ZH[obj.shape]}"


def _attribute_pair(rng: np.random.Generator) -> tuple[tuple[Color, Shape, Size], tuple[Color, Shape, Size]]:
    colors = list(Color)
    shapes = list(Shape)
    first = (colors[int(rng.integers(len(colors)))], shapes[int(rng.integers(len(shapes)))], Size.LARGE)
    while True:
        second = (colors[int(rng.integers(len(colors)))], shapes[int(rng.integers(len(shapes)))], Size.MEDIUM)
        if second[:2] != first[:2]:
            return first, second


def _object(
    object_id: str,
    attributes: tuple[Color, Shape, Size],
    *,
    position: tuple[float, float],
    direction: MotionDirection = MotionDirection.STATIC,
    speed: float = 0.0,
    move_start: int = 0,
    move_end: int = 7,
    visible_start: int = 0,
    visible_end: int = 7,
) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        color=attributes[0],
        shape=attributes[1],
        size=attributes[2],
        initial_position=Position(*position),
        motion=MotionSpec(direction, speed, move_start, move_end),
        visible_start=visible_start,
        visible_end=visible_end,
    )


def _scene(scene_id: str, seed: int, objects: Sequence[SceneObject]) -> Scene:
    return Scene(scene_id, seed, 32, 32, 8, 4, tuple(objects))


def _record(
    *,
    scene: Scene,
    split: str,
    kind: str,
    question: str,
    answer: str,
    group_id: str,
    target: str | None,
    reference: str | None,
    events: Mapping[str, int],
    pair_id: str | None = None,
    partner_record_id: str | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    rendered = render_video(scene)
    row = {
        "record_id": scene.scene_id,
        "scene_id": scene.scene_id,
        "split": split,
        "question_kind": kind,
        "question_zh": question,
        "answer_zh": answer,
        "group_id": group_id,
        "pair_id": pair_id,
        "partner_record_id": partner_record_id,
        "target_object_id": target,
        "reference_object_id": reference,
        "events": dict(events),
        "objects": [
            {
                "object_id": obj.object_id,
                "descriptor_zh": _descriptor(obj),
                "color": obj.color.value,
                "shape": obj.shape.value,
                "size": obj.size.value,
                "direction": obj.motion.direction.value,
                "speed": obj.motion.speed,
                "move_start": obj.motion.start_frame,
                "move_end": obj.motion.end_frame,
                "visible_start": obj.visible_start,
                "visible_end": obj.visible_end,
            }
            for obj in scene.objects
        ],
    }
    return row, rendered.frames


def _direction_group(
    split: str, group_index: int, rng: np.random.Generator, seed: int
) -> list[tuple[dict[str, Any], np.ndarray]]:
    attributes = _attribute_pair(rng)
    target_attributes, distractor_attributes = attributes
    horizontal_y = float(rng.uniform(0.32, 0.68))
    vertical_x = float(rng.uniform(0.32, 0.68))
    static_position = (float(rng.uniform(0.35, 0.65)), float(rng.uniform(0.35, 0.65)))
    distractor_position = (
        float(rng.uniform(0.15, 0.85)),
        0.18 if group_index % 2 == 0 else 0.82,
    )
    specs = (
        ("left", MotionDirection.LEFT, (0.72, horizontal_y), 0.04, "horizontal"),
        ("right", MotionDirection.RIGHT, (0.44, horizontal_y), 0.04, "horizontal"),
        ("up", MotionDirection.UP, (vertical_x, 0.72), 0.04, "vertical"),
        ("down", MotionDirection.DOWN, (vertical_x, 0.44), 0.04, "vertical"),
        ("static", MotionDirection.STATIC, static_position, 0.0, "static"),
    )
    ids = {
        label: f"p3w01_{split}_motion_{group_index:04d}_{label}" for label, *_ in specs
    }
    result = []
    for offset, (label, direction, position, speed, axis) in enumerate(specs):
        pair_id = None
        partner = None
        if direction in OPPOSITE_DIRECTION:
            opposite = OPPOSITE_DIRECTION[direction].value
            pair_id = f"p3w01_{split}_motion_{group_index:04d}_{axis}_pair"
            partner = ids[opposite]
        target = _object(
            "target",
            target_attributes,
            position=position,
            direction=direction,
            speed=speed,
        )
        distractor = _object(
            "distractor",
            distractor_attributes,
            position=distractor_position,
        )
        scene = _scene(ids[label], seed + group_index * 10 + offset, (target, distractor))
        question = f"视频中，{_descriptor(target)}向哪个方向运动？"
        result.append(
            _record(
                scene=scene,
                split=split,
                kind="motion_direction",
                question=question,
                answer=DIRECTION_ZH[direction],
                group_id=f"p3w01_{split}_motion_{group_index:04d}",
                target="target",
                reference=None,
                events={"move_start": target.motion.start_frame, "move_end": target.motion.end_frame},
                pair_id=pair_id,
                partner_record_id=partner,
            )
        )
    return result


def _order_group(
    split: str,
    kind: str,
    group_index: int,
    rng: np.random.Generator,
    seed: int,
) -> list[tuple[dict[str, Any], np.ndarray]]:
    first_attributes, second_attributes = _attribute_pair(rng)
    first_position = (float(rng.uniform(0.18, 0.34)), float(rng.uniform(0.28, 0.72)))
    second_position = (float(rng.uniform(0.66, 0.82)), float(rng.uniform(0.28, 0.72)))
    labels = ("first", "second", "same")
    answers = dict(zip(labels, ORDER_ANSWERS, strict=True))
    result = []
    for offset, label in enumerate(labels):
        if kind == "appear_order":
            starts = {"first": (0, 3), "second": (3, 0), "same": (2, 2)}[label]
            first = _object("first", first_attributes, position=first_position, visible_start=starts[0])
            second = _object("second", second_attributes, position=second_position, visible_start=starts[1])
            events = {"first_appear": starts[0], "second_appear": starts[1]}
            verb = "先出现"
        elif kind == "disappear_order":
            ends = {"first": (4, 7), "second": (7, 4), "same": (6, 6)}[label]
            first = _object("first", first_attributes, position=first_position, visible_end=ends[0])
            second = _object("second", second_attributes, position=second_position, visible_end=ends[1])
            events = {"first_disappear": ends[0] + 1, "second_disappear": ends[1] + 1}
            verb = "先消失"
        elif kind == "move_start_order":
            starts = {"first": (0, 3), "second": (3, 0), "same": (2, 2)}[label]
            first_speed = 0.18 / (7 - starts[0])
            second_speed = 0.18 / (7 - starts[1])
            first = _object(
                "first",
                first_attributes,
                position=(first_position[0], min(first_position[1], 0.48)),
                direction=MotionDirection.RIGHT,
                speed=first_speed,
                move_start=starts[0],
            )
            second = _object(
                "second",
                second_attributes,
                position=(second_position[0], max(second_position[1], 0.52)),
                direction=MotionDirection.LEFT,
                speed=second_speed,
                move_start=starts[1],
            )
            events = {"first_move_start": starts[0], "second_move_start": starts[1]}
            verb = "先开始移动"
        else:
            raise ValueError(f"unsupported temporal order kind: {kind}")
        scene_id = f"p3w01_{split}_{kind}_{group_index:04d}_{label}"
        scene = _scene(scene_id, seed + group_index * 10 + offset, (first, second))
        question = f"视频中，{_descriptor(first)}和{_descriptor(second)}，哪一个{verb}？"
        result.append(
            _record(
                scene=scene,
                split=split,
                kind=kind,
                question=question,
                answer=answers[label],
                group_id=f"p3w01_{split}_{kind}_{group_index:04d}",
                target="first",
                reference="second",
                events=events,
            )
        )
    return result


def build_temporal_challenge(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    seed = int(config["seed"])
    split_counts = config["data"]["split_counts"]
    records: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    for split_offset, (split, total) in enumerate(split_counts.items()):
        if int(total) % 20:
            raise ValueError("each split count must be divisible by 20")
        per_kind = int(total) // 4
        if per_kind % 5 or per_kind % 3:
            raise ValueError("per-kind count must be divisible by both 5 and 3")
        rng = np.random.default_rng(seed + split_offset * 100000)
        groups: list[tuple[dict[str, Any], np.ndarray]] = []
        for group_index in range(per_kind // 5):
            groups.extend(_direction_group(split, group_index, rng, seed + split_offset * 100000))
        for kind_offset, kind in enumerate(("appear_order", "disappear_order", "move_start_order")):
            for group_index in range(per_kind // 3):
                groups.extend(
                    _order_group(
                        split,
                        kind,
                        group_index,
                        rng,
                        seed + split_offset * 100000 + (kind_offset + 1) * 10000,
                    )
                )
        if len(groups) != int(total):
            raise AssertionError("split generation count differs")
        for row, video in groups:
            records.append(row)
            frames.append(video)
    answer_classes = list(config["data"]["direction_answers"]) + list(config["data"]["order_answers"])
    answer_to_class = {answer: index for index, answer in enumerate(answer_classes)}
    for row in records:
        row["answer_class"] = answer_to_class[row["answer_zh"]]
    return records, np.stack(frames).astype(np.uint8)


def export_temporal_challenge(project_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = root / config["data"]["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    records, videos = build_temporal_challenge(config)
    answer_classes = list(config["data"]["direction_answers"]) + list(config["data"]["order_answers"])
    train = [row for row in records if row["split"] == "train"]
    characters = sorted(
        {char for row in train for text in (row["question_zh"], row["answer_zh"]) for char in text}
    )
    tokens = list(SPECIAL_TOKENS) + characters
    vocabulary = {"tokens": tokens, "token_to_id": {token: index for index, token in enumerate(tokens)}}
    unknown = Counter()
    for row in records:
        for text in (row["question_zh"], row["answer_zh"]):
            if any(char not in vocabulary["token_to_id"] for char in text):
                unknown[row["split"]] += 1
    records_path = output / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
        newline="\n",
    )
    np.savez_compressed(
        output / "videos.npz",
        record_ids=np.asarray([row["record_id"] for row in records]),
        frames=videos,
    )
    _write_json(output / "vocabulary.json", vocabulary)
    _write_json(output / "answer_classes.json", answer_classes)
    overfit_ids = [row["record_id"] for row in train[: int(config["data"]["overfit_count"])]]
    _write_json(output / "overfit_ids.json", overfit_ids)
    split_counts = Counter(row["split"] for row in records)
    kind_counts = {
        split: dict(sorted(Counter(row["question_kind"] for row in records if row["split"] == split).items()))
        for split in split_counts
    }
    answer_counts = {
        split: dict(sorted(Counter(row["answer_zh"] for row in records if row["split"] == split).items()))
        for split in split_counts
    }
    ids_by_split = {
        split: {row["record_id"] for row in records if row["split"] == split} for split in split_counts
    }
    frame_hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in videos]
    hashes_by_split = {
        split: {frame_hashes[index] for index, row in enumerate(records) if row["split"] == split}
        for split in split_counts
    }
    pair_rows = [row for row in records if row["pair_id"]]
    index = {row["record_id"]: position for position, row in enumerate(records)}
    pair_errors = []
    seen_pairs = set()
    for row in pair_rows:
        if row["pair_id"] in seen_pairs:
            continue
        seen_pairs.add(row["pair_id"])
        partner = row["partner_record_id"]
        if partner not in index or not np.array_equal(videos[index[row["record_id"]]][::-1], videos[index[partner]]):
            pair_errors.append(row["pair_id"])
    p1_manifest = root / config["data"]["source_p1_video_manifest"]
    audit = {
        "record_count": len(records),
        "video_shape": list(videos.shape[1:]),
        "split_counts": dict(sorted(split_counts.items())),
        "question_kind_counts_by_split": kind_counts,
        "answer_counts_by_split": answer_counts,
        "record_id_overlap_counts": {
            "train_validation": len(ids_by_split["train"] & ids_by_split["validation"]),
            "train_test": len(ids_by_split["train"] & ids_by_split["test"]),
            "validation_test": len(ids_by_split["validation"] & ids_by_split["test"]),
        },
        "video_sha256_overlap_counts": {
            "train_validation": len(hashes_by_split["train"] & hashes_by_split["validation"]),
            "train_test": len(hashes_by_split["train"] & hashes_by_split["test"]),
            "validation_test": len(hashes_by_split["validation"] & hashes_by_split["test"]),
        },
        "answer_string_leak_count": sum(row["answer_zh"] in row["question_zh"] for row in records),
        "unknown_text_count_by_split": dict(sorted(unknown.items())),
        "motion_reversal_pair_count_by_split": dict(
            sorted(Counter(row["split"] for row in pair_rows).items())
        ),
        "unique_motion_reversal_pair_count_by_split": dict(
            sorted(Counter(row["split"] for row in pair_rows[::2]).items())
        ),
        "motion_reversal_pixel_mismatch_pair_ids": pair_errors,
        "source_p1_manifest": config["data"]["source_p1_video_manifest"],
        "source_p1_manifest_sha256": _sha256(p1_manifest),
        "source_p1_modified": False,
        "p1_color_shape_compositional_claim": False,
    }
    _write_json(output / "data_audit.json", audit)
    _write_json(
        output / "temporal_pair_audit.json",
        {
            "unique_pair_count": len(seen_pairs),
            "pair_counts_by_split": dict(
                sorted(Counter(row["split"] for row in pair_rows[::2]).items())
            ),
            "pixel_mismatch_pair_ids": pair_errors,
            "event_value_counts": dict(
                sorted(Counter(value for row in records for value in row["events"].values()).items())
            ),
        },
    )
    files = {}
    for name in (
        "records.jsonl",
        "videos.npz",
        "vocabulary.json",
        "answer_classes.json",
        "overfit_ids.json",
        "data_audit.json",
        "temporal_pair_audit.json",
    ):
        path = output / name
        files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema_version": 1,
        "dataset_id": "p3_w01_v1",
        "work_package": "P3-W01",
        "source_p1_video_manifest": config["data"]["source_p1_video_manifest"],
        "source_p1_manifest_sha256": audit["source_p1_manifest_sha256"],
        "files": files,
        "statistics": {
            "record_count": len(records),
            "split_counts": audit["split_counts"],
            "question_kind_counts_by_split": kind_counts,
            "answer_counts_by_split": answer_counts,
            "vocabulary_size": len(tokens),
            "answer_class_count": len(answer_classes),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def verify_temporal_challenge(project_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    directory = Path(data_dir)
    if not directory.is_absolute():
        directory = root / directory
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != metadata["bytes"] or _sha256(path) != metadata["sha256"]:
            raise ValueError(f"P3-W01 data artifact differs: {name}")
    source = root / manifest["source_p1_video_manifest"]
    if _sha256(source) != manifest["source_p1_manifest_sha256"]:
        raise ValueError("P1-W06 source manifest changed")
    audit = json.loads((directory / "data_audit.json").read_text(encoding="utf-8"))
    if any(audit["record_id_overlap_counts"].values()) or any(audit["video_sha256_overlap_counts"].values()):
        raise ValueError("P3-W01 split leakage detected")
    if audit["motion_reversal_pixel_mismatch_pair_ids"]:
        raise ValueError("P3-W01 reversal pairs differ at pixel level")
    return {"verified": True, "statistics": manifest["statistics"]}


class CharacterTokenizer:
    def __init__(self, vocabulary: Mapping[str, Any]) -> None:
        self.tokens = tuple(str(token) for token in vocabulary["tokens"])
        self.token_to_id = {str(key): int(value) for key, value in vocabulary["token_to_id"].items()}
        if self.tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError("P3-W01 special token order differs")

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
        result = []
        for index in ids:
            token = self.tokens[int(index)]
            if token == "<eos>":
                break
            if token not in SPECIAL_TOKENS:
                result.append(token)
        return "".join(result)


class VideoQADataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        project_root: str | Path,
        *,
        data_dir: str | Path,
        splits: str | Sequence[str] | None = None,
        record_ids: Sequence[str] | None = None,
    ) -> None:
        root = Path(project_root).resolve()
        directory = Path(data_dir)
        self.directory = directory if directory.is_absolute() else root / directory
        self.tokenizer = CharacterTokenizer(
            json.loads((self.directory / "vocabulary.json").read_text(encoding="utf-8"))
        )
        selected_splits = None if splits is None else ({splits} if isinstance(splits, str) else set(splits))
        selected_ids = None if record_ids is None else set(record_ids)
        rows = [json.loads(line) for line in (self.directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        with np.load(self.directory / "videos.npz", allow_pickle=False) as cache:
            ids = [str(value) for value in cache["record_ids"].tolist()]
            all_video = cache["frames"]
            source_video_index = {record_id: index for index, record_id in enumerate(ids)}
            rows = [
                row
                for row in rows
                if (selected_splits is None or row["split"] in selected_splits)
                and (selected_ids is None or row["record_id"] in selected_ids)
            ]
            video = np.stack([all_video[source_video_index[row["record_id"]]] for row in rows]).copy()
        self.rows = rows
        if not self.rows:
            raise ValueError("P3-W01 dataset filters selected no records")
        self.videos = video
        self.video_index = {row["record_id"]: index for index, row in enumerate(self.rows)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        frames = self.videos[self.video_index[row["record_id"]]].astype(np.float32) / 255.0
        question_tokens = self.tokenizer.encode_question(row["question_zh"])
        object_query_masks = torch.zeros(2, len(question_tokens), dtype=torch.bool)
        objects_by_id = {item["object_id"]: item for item in row["objects"]}
        query_object_ids = (row["target_object_id"], row["reference_object_id"])
        for slot, object_id in enumerate(query_object_ids):
            if object_id is None:
                object_query_masks[slot, 0] = True
                continue
            descriptor = objects_by_id[object_id]["descriptor_zh"]
            start = row["question_zh"].find(descriptor)
            if start < 0:
                raise ValueError(f"object descriptor is absent from question: {row['record_id']}")
            object_query_masks[slot, 1 + start : 1 + start + len(descriptor)] = True
        move_start_targets = torch.full((2,), -100, dtype=torch.long)
        if row["question_kind"] == "move_start_order":
            move_start_targets[0] = int(objects_by_id[row["target_object_id"]]["move_start"])
            move_start_targets[1] = int(objects_by_id[row["reference_object_id"]]["move_start"])
        answer_input, answer_target = self.tokenizer.encode_answer(row["answer_zh"])
        return {
            "video": torch.from_numpy(frames).permute(0, 3, 1, 2),
            "question_ids": torch.tensor(question_tokens, dtype=torch.long),
            "object_query_masks": object_query_masks,
            "move_start_targets": move_start_targets,
            "answer_input_ids": torch.tensor(answer_input, dtype=torch.long),
            "answer_target_ids": torch.tensor(answer_target, dtype=torch.long),
            "answer_class": torch.tensor(row["answer_class"], dtype=torch.long),
            "metadata": row,
        }


def _pad(sequences: Sequence[torch.Tensor], value: int) -> torch.Tensor:
    width = max(int(item.numel()) for item in sequences)
    result = torch.full((len(sequences), width), value, dtype=sequences[0].dtype)
    for index, item in enumerate(sequences):
        result[index, : item.numel()] = item
    return result


def collate_video_qa(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate empty P3-W01 batch")
    question_ids = _pad([sample["question_ids"] for sample in samples], 0)
    object_query_masks = torch.zeros(
        len(samples), 2, question_ids.shape[1], dtype=torch.bool
    )
    for index, sample in enumerate(samples):
        width = sample["object_query_masks"].shape[1]
        object_query_masks[index, :, :width] = sample["object_query_masks"]
    return {
        "video": torch.stack([sample["video"] for sample in samples]),
        "question_ids": question_ids,
        "object_query_masks": object_query_masks,
        "move_start_targets": torch.stack([sample["move_start_targets"] for sample in samples]),
        "answer_input_ids": _pad([sample["answer_input_ids"] for sample in samples], 0),
        "answer_target_ids": _pad([sample["answer_target_ids"] for sample in samples], 0),
        "answer_class": torch.stack([sample["answer_class"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }
