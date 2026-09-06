"""Fixed task-routing and diagnostic-baseline data for P2-W01."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from mini_multimodal.multimodal.dataset import UnifiedMultimodalDataset
from mini_multimodal.multimodal.routing import route_question_kind, route_table
from mini_multimodal.world.schema import Shape


SHAPE_CLASSES = tuple(shape.value for shape in Shape)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _baseline_entry(
    dataset: UnifiedMultimodalDataset,
    index: int,
    image_annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    record = dataset.records[index]
    scene = dataset._scene_by_id[record.scene_id]
    if len(scene.objects) != 1:
        raise ValueError(f"baseline scene must contain one object: {record.scene_id}")
    answer = str(record.task.answer.value)
    if answer not in SHAPE_CLASSES:
        raise ValueError(f"unexpected shape label: {answer}")
    row = dataset._row_by_scene[record.scene_id]
    annotation = image_annotations[int(row["source_row"])]["annotation"]
    if len(annotation["objects"]) != 1:
        raise ValueError(f"baseline annotation must contain one object: {record.scene_id}")
    return {
        "record_id": record.record_id,
        "scene_id": record.scene_id,
        "label": answer,
        "label_index": SHAPE_CLASSES.index(answer),
        "split": record.split.value,
        "variant": record.variant.value,
        "cue_token": record.cue_token,
        "cue_aligned": record.cue_aligned,
        "image_path": row["assets"]["image"]["path"],
        "image_sha256": row["assets"]["image"]["sha256"],
        "object_bbox_xywh": annotation["objects"][0]["visible_bbox_xywh"],
    }


def _balanced_overfit(entries: list[dict[str, Any]], per_class: int) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in sorted(entries, key=lambda item: item["record_id"]):
        by_class[entry["label"]].append(entry)
    if any(len(by_class[label]) < per_class for label in SHAPE_CLASSES):
        raise ValueError("insufficient examples for balanced overfit fixture")
    return [entry for label in SHAPE_CLASSES for entry in by_class[label][:per_class]]


def _group_report(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(entries)
    return {
        "count": len(values),
        "label_counts": dict(sorted(Counter(item["label"] for item in values).items())),
        "variant_counts": dict(
            sorted(Counter(item["variant"] for item in values).items())
        ),
    }


def build_p2_w01_data_spec(
    index_dir: str | Path, *, overfit_examples_per_class: int = 4
) -> tuple[dict[str, Any], dict[str, Any]]:
    index = Path(index_dir).resolve()
    all_qa = UnifiedMultimodalDataset(index, view="qa", modalities=())
    image_qa = UnifiedMultimodalDataset(
        index, view="qa", required_modality="image", modalities=()
    )
    video_qa = UnifiedMultimodalDataset(
        index, view="qa", required_modality="video", modalities=()
    )
    image_by_split = Counter(record.split.value for record in image_qa.records)
    video_by_split = Counter(record.split.value for record in video_qa.records)
    image_by_kind = Counter(record.task.question.kind.value for record in image_qa.records)
    video_by_kind = Counter(record.task.question.kind.value for record in video_qa.records)
    routing_report = {
        "route_table": route_table(),
        "all_qa_count": len(all_qa),
        "image_qa": {
            "count": len(image_qa),
            "split_counts": dict(sorted(image_by_split.items())),
            "question_kind_counts": dict(sorted(image_by_kind.items())),
        },
        "video_qa": {
            "count": len(video_qa),
            "split_counts": dict(sorted(video_by_split.items())),
            "question_kind_counts": dict(sorted(video_by_kind.items())),
        },
        "unrouted_count": sum(
            1
            for record in all_qa.records
            if not route_question_kind(record.task.question.kind).supported_modalities
        ),
    }

    shape_qa = UnifiedMultimodalDataset(
        index,
        view="qa",
        pools=("shortcut", "compositional"),
        question_kinds="object_shape",
        modalities=(),
    )
    image_source = (
        shape_qa.project_root
        / shape_qa.manifest["sources"]["image"]["path"]
        / "annotations.jsonl"
    )
    image_annotations = [
        json.loads(line) for line in image_source.read_text(encoding="utf-8").splitlines()
    ]
    entries = [
        _baseline_entry(shape_qa, item_index, image_annotations)
        for item_index in range(len(shape_qa))
    ]
    groups = {
        "train": [
            item
            for item in entries
            if item["split"] == "train"
            and item["variant"] in {"shortcut_iid", "compositional"}
        ],
        "validation": [
            item
            for item in entries
            if item["split"] == "validation"
            and item["variant"] in {"shortcut_iid", "compositional"}
        ],
        "test": [
            item
            for item in entries
            if item["split"] == "test"
            and item["variant"] in {"counterfactual", "compositional"}
        ],
    }
    for values in groups.values():
        values.sort(key=lambda item: item["record_id"])
    overfit = _balanced_overfit(groups["train"], overfit_examples_per_class)
    ids = {name: {item["record_id"] for item in values} for name, values in groups.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise ValueError("baseline splits overlap")
    baseline = {
        "classes": list(SHAPE_CLASSES),
        "groups": groups,
        "overfit": overfit,
        "statistics": {name: _group_report(values) for name, values in groups.items()},
        "overfit_statistics": _group_report(overfit),
    }
    return routing_report, baseline


def export_p2_w01_data(
    index_dir: str | Path,
    output_dir: str | Path,
    *,
    overfit_examples_per_class: int = 4,
) -> dict[str, Any]:
    index = Path(index_dir).resolve()
    output = Path(output_dir)
    routing, baseline = build_p2_w01_data_spec(
        index, overfit_examples_per_class=overfit_examples_per_class
    )
    _write_json(output / "task_modality_routes.json", routing)
    _write_json(output / "baseline_splits.json", baseline)
    source_manifest = index / "manifest.json"
    manifest = {
        "artifact_version": 1,
        "source_index": index.relative_to(index.parents[2]).as_posix(),
        "source_manifest_sha256": _sha(source_manifest),
        "files": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": _sha(output / name),
            }
            for name in ("task_modality_routes.json", "baseline_splits.json")
        },
        "routing_summary": {
            "all_qa_count": routing["all_qa_count"],
            "image_qa_count": routing["image_qa"]["count"],
            "video_qa_count": routing["video_qa"]["count"],
        },
        "baseline_statistics": baseline["statistics"],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def verify_p2_w01_data(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    project_root = output.parents[2]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = output / name
        if not path.is_file() or _sha(path) != expected["sha256"]:
            raise ValueError(f"P2-W01 data file differs from manifest: {name}")
    source_manifest = project_root / manifest["source_index"] / "manifest.json"
    if _sha(source_manifest) != manifest["source_manifest_sha256"]:
        raise ValueError("P2-W01 source index differs from manifest")
    routing = json.loads((output / "task_modality_routes.json").read_text(encoding="utf-8"))
    baseline = json.loads((output / "baseline_splits.json").read_text(encoding="utf-8"))
    expected_summary = {
        "all_qa_count": routing["all_qa_count"],
        "image_qa_count": routing["image_qa"]["count"],
        "video_qa_count": routing["video_qa"]["count"],
    }
    if expected_summary != manifest["routing_summary"]:
        raise ValueError("P2-W01 routing summary differs from manifest")
    if baseline["statistics"] != manifest["baseline_statistics"]:
        raise ValueError("P2-W01 baseline statistics differ from manifest")
    return {"verified": True, **expected_summary, "files": manifest["files"]}
