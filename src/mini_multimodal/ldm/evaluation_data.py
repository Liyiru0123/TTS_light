"""Fixed, balanced P2-W04 v2 evaluation and intervention fixtures."""

from __future__ import annotations

import hashlib
import io
import json
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mini_multimodal.tasks.reasoning import position_region
from mini_multimodal.tasks.text_zh import ChineseTextAdapter
from mini_multimodal.vision.renderer import render_scene
from mini_multimodal.world.schema import Color, Position, Scene, SceneObject, Shape, Size
from mini_multimodal.world.serialization import scene_to_dict


REGION_POSITIONS: dict[str, tuple[float, float]] = {
    "upper_left": (0.18, 0.18),
    "upper_center": (0.50, 0.18),
    "upper_right": (0.82, 0.18),
    "center_left": (0.18, 0.50),
    "center": (0.50, 0.50),
    "center_right": (0.82, 0.50),
    "lower_left": (0.18, 0.82),
    "lower_center": (0.50, 0.82),
    "lower_right": (0.82, 0.82),
}
REGION_ORDER = tuple(REGION_POSITIONS)
SIZE_ORDER = (Size.SMALL, Size.MEDIUM, Size.LARGE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_digest(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _jittered_position(region: str, key: int, attempt: int) -> tuple[float, float]:
    base_x, base_y = REGION_POSITIONS[region]
    offsets = (-0.04, -0.02, 0.0, 0.02, 0.04)
    variant = key + attempt
    return (
        base_x + offsets[variant % len(offsets)],
        base_y + offsets[(variant // len(offsets)) % len(offsets)],
    )


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


def _object_record(obj: SceneObject) -> dict[str, Any]:
    position = obj.position_at(0)
    return {
        "object_id": obj.object_id,
        "color": obj.color.value,
        "shape": obj.shape.value,
        "size": obj.size.value,
        "position": [position.x, position.y],
        "position_region": position_region(position).value,
    }


def _scene(scene_id: str, seed: int, specs: Sequence[Mapping[str, Any]]) -> Scene:
    objects = []
    for index, spec in enumerate(specs):
        x, y = spec["position"]
        objects.append(
            SceneObject(
                object_id=f"obj_{index}",
                color=Color(str(spec["color"])),
                shape=Shape(str(spec["shape"])),
                size=Size(str(spec["size"])),
                initial_position=Position(float(x), float(y)),
                visible_start=0,
                visible_end=0,
            )
        )
    return Scene(
        scene_id=scene_id,
        seed=seed,
        canvas_width=32,
        canvas_height=32,
        frame_count=1,
        fps=8,
        objects=tuple(objects),
    )


def _parse_pair(pair: str) -> tuple[str, str]:
    color, shape = pair.split("|", 1)
    return color, shape


def _record_for_scene(
    root: Path,
    output: Path,
    scene: Scene,
    *,
    subset: str,
    mask_index: int,
    composition_pair: str | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    rendered = render_scene(scene, frame_index=0)
    image_dir = output / "source_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{scene.scene_id}.png"
    rendered.image.save(image_path, format="PNG", optimize=False, compress_level=9)
    amodal = np.zeros((3, 64, 64), dtype=np.bool_)
    visible = np.zeros((3, 64, 64), dtype=np.bool_)
    for index, obj in enumerate(scene.objects):
        amodal[index] = rendered.amodal_masks[obj.object_id]
        visible[index] = rendered.visible_masks[obj.object_id]
    target_objects = []
    annotations = {item.object_id: item for item in rendered.annotation.objects}
    for obj in scene.objects:
        item = _object_record(obj)
        annotation = annotations[obj.object_id]
        item.update({
            "rendered_center_xy": [
                annotation.center_xy[0] / 63.0,
                annotation.center_xy[1] / 63.0,
            ],
            "bbox_xywh": list(annotation.bbox_xywh),
            "area_pixels": annotation.area_pixels,
        })
        target_objects.append(item)
    row = {
        "record_id": f"eval:{scene.scene_id}",
        "scene_id": scene.scene_id,
        "subset": subset,
        "prompt": ChineseTextAdapter().describe_scene(scene, 0),
        "objects": target_objects,
        "object_count": len(scene.objects),
        "composition_pair": composition_pair,
        "image_path": image_path.relative_to(root).as_posix(),
        "image_sha256": _sha(image_path),
        "ideal_mask_index": mask_index,
        "scene": scene_to_dict(scene),
    }
    return row, amodal, visible


def _unique_color_specs(
    allowed_pairs: Sequence[str], start: int, count: int, positions: Sequence[str]
) -> list[dict[str, Any]]:
    chosen: list[str] = []
    used_colors: set[str] = set()
    for offset in range(len(allowed_pairs) * 2):
        pair = allowed_pairs[(start + offset) % len(allowed_pairs)]
        color, _ = _parse_pair(pair)
        if color in used_colors:
            continue
        chosen.append(pair)
        used_colors.add(color)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError("could not choose distinct colors for multi-object fixture")
    result = []
    for index, (pair, region) in enumerate(zip(chosen, positions, strict=True)):
        color, shape = _parse_pair(pair)
        result.append({
            "color": color,
            "shape": shape,
            "size": SIZE_ORDER[(start + index) % 3].value,
            "position": REGION_POSITIONS[region],
        })
    return result


def _intervention_row(
    scene: Scene, *, group_id: str, changed_attribute: str, sample_seed: int
) -> dict[str, Any]:
    return {
        "record_id": f"intervention:{scene.scene_id}",
        "scene_id": scene.scene_id,
        "group_id": group_id,
        "changed_attribute": changed_attribute,
        "shared_sample_seed": sample_seed,
        "prompt": ChineseTextAdapter().describe_scene(scene, 0),
        "objects": [_object_record(obj) for obj in scene.objects],
        "scene": scene_to_dict(scene),
    }


def prepare_fixed_evaluation_data(
    project_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = root / str(config["output_eval_data"])
    output.mkdir(parents=True, exist_ok=True)
    policy_path = root / str(config["data"]["holdout_policy"])
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    allowed = sorted(str(value) for value in policy["allowed_training_pairs"])
    holdouts = sorted(str(value) for value in policy["holdout_pairs"])
    seed = int(config["data"]["evaluation_seed"])
    blocked_hashes: set[str] = set()
    training_records = root / str(config["output_data"]) / "records.jsonl"
    if training_records.is_file():
        blocked_hashes = {
            row["image_sha256"]
            for row in (
                json.loads(line)
                for line in training_records.read_text(encoding="utf-8").splitlines()
            )
            if row["split"] == "train"
        }
    used_hashes = set(blocked_hashes)

    rows: list[dict[str, Any]] = []
    amodal_masks: list[np.ndarray] = []
    visible_masks: list[np.ndarray] = []

    for pair_index, pair in enumerate(allowed):
        color, shape = _parse_pair(pair)
        for region_index, region in enumerate(REGION_ORDER):
            size = SIZE_ORDER[(pair_index + region_index) % 3]
            for attempt in range(100):
                scene = _scene(
                    f"ldmeval_seen_{pair_index:02d}_{region_index:02d}",
                    seed + len(rows),
                    [{
                        "color": color,
                        "shape": shape,
                        "size": size.value,
                        "position": _jittered_position(region, pair_index * 9 + region_index, attempt),
                    }],
                )
                if _png_digest(render_scene(scene, frame_index=0).image) not in used_hashes:
                    break
            else:
                raise RuntimeError("could not isolate a seen evaluation image from training")
            row, amodal, visible = _record_for_scene(
                root, output, scene, subset="seen_single", mask_index=len(rows), composition_pair=pair
            )
            used_hashes.add(row["image_sha256"])
            rows.append(row); amodal_masks.append(amodal); visible_masks.append(visible)

    for pair_index, pair in enumerate(holdouts):
        color, shape = _parse_pair(pair)
        for region_index, region in enumerate(REGION_ORDER):
            for size_index, size in enumerate(SIZE_ORDER):
                for attempt in range(100):
                    scene = _scene(
                        f"ldmeval_unseen_{pair_index:02d}_{region_index:02d}_{size_index}",
                        seed + len(rows),
                        [{
                            "color": color,
                            "shape": shape,
                            "size": size.value,
                            "position": _jittered_position(
                                region,
                                225 + pair_index * 27 + region_index * 3 + size_index,
                                attempt,
                            ),
                        }],
                    )
                    if _png_digest(render_scene(scene, frame_index=0).image) not in used_hashes:
                        break
                else:
                    raise RuntimeError("could not isolate an unseen evaluation image from training")
                row, amodal, visible = _record_for_scene(
                    root, output, scene, subset="unseen_single", mask_index=len(rows), composition_pair=pair
                )
                used_hashes.add(row["image_sha256"])
                rows.append(row); amodal_masks.append(amodal); visible_masks.append(visible)

    two_patterns = (
        ("upper_left", "lower_right"),
        ("upper_right", "lower_left"),
        ("center_left", "center_right"),
        ("upper_center", "lower_center"),
        ("upper_left", "center_right"),
    )
    three_patterns = (
        ("upper_left", "upper_right", "lower_center"),
        ("upper_center", "lower_left", "lower_right"),
        ("center_left", "upper_right", "lower_right"),
        ("upper_left", "center_right", "lower_left"),
        ("upper_center", "center_left", "lower_right"),
    )
    for index in range(60):
        count = 2 if index < 30 else 3
        patterns = two_patterns if count == 2 else three_patterns
        for attempt in range(100):
            positions = patterns[(index + attempt) % len(patterns)]
            specs = _unique_color_specs(allowed, index * 11 + attempt, count, positions)
            scene = _scene(f"ldmeval_multi_{index:02d}", seed + len(rows), specs)
            image_digest = _png_digest(render_scene(scene, frame_index=0).image)
            if image_digest not in used_hashes:
                break
        else:
            raise RuntimeError("could not construct a unique multi-object fixture")
        row, amodal, visible = _record_for_scene(
            root, output, scene, subset=f"multi_{count}", mask_index=len(rows)
        )
        used_hashes.add(row["image_sha256"])
        rows.append(row); amodal_masks.append(amodal); visible_masks.append(visible)

    interventions: list[dict[str, Any]] = []
    allowed_cycle = cycle(allowed)
    for group_index in range(12):
        pair = next(allowed_cycle)
        color, shape = _parse_pair(pair)
        horizontal = group_index < 6
        regions = (
            ("center_left", "center", "center_right")
            if horizontal
            else ("upper_center", "center", "lower_center")
        )
        group_id = f"position_{group_index:02d}"
        for item_index, region in enumerate(regions):
            scene = _scene(
                f"ldmint_{group_id}_{item_index}", seed + 10000 + len(interventions),
                [{
                    "color": color,
                    "shape": shape,
                    "size": SIZE_ORDER[group_index % 3].value,
                    "position": REGION_POSITIONS[region],
                }],
            )
            interventions.append(_intervention_row(
                scene, group_id=group_id, changed_attribute="position", sample_seed=seed + 20000 + group_index
            ))

    colors = [color.value for color in Color]
    shapes = [shape.value for shape in Shape]
    for group_index, color in enumerate(colors):
        group_id = f"shape_{group_index:02d}"
        for item_index, shape in enumerate(shapes):
            scene = _scene(
                f"ldmint_{group_id}_{item_index}", seed + 10000 + len(interventions),
                [{
                    "color": color,
                    "shape": shape,
                    "size": SIZE_ORDER[group_index % 3].value,
                    "position": REGION_POSITIONS[REGION_ORDER[(group_index * 2) % 9]],
                }],
            )
            interventions.append(_intervention_row(
                scene, group_id=group_id, changed_attribute="shape", sample_seed=seed + 21000 + group_index
            ))

    for group_index, shape in enumerate(shapes):
        group_id = f"color_{group_index:02d}"
        for item_index, color in enumerate(colors):
            scene = _scene(
                f"ldmint_{group_id}_{item_index}", seed + 10000 + len(interventions),
                [{
                    "color": color,
                    "shape": shape,
                    "size": SIZE_ORDER[group_index % 3].value,
                    "position": REGION_POSITIONS[REGION_ORDER[(group_index * 2 + 1) % 9]],
                }],
            )
            interventions.append(_intervention_row(
                scene, group_id=group_id, changed_attribute="color", sample_seed=seed + 22000 + group_index
            ))

    if len(interventions) != 96:
        raise AssertionError(f"expected 96 interventions, got {len(interventions)}")

    _write_jsonl(output / "records.jsonl", rows)
    _write_jsonl(output / "interventions.jsonl", interventions)
    np.savez_compressed(
        output / "ideal_masks.npz",
        amodal=np.stack(amodal_masks),
        visible=np.stack(visible_masks),
        record_ids=np.asarray([row["record_id"] for row in rows]),
        object_counts=np.asarray([row["object_count"] for row in rows], dtype=np.int8),
    )
    stable_ids = [row["record_id"] for row in rows[:15]] + [
        row["record_id"] for row in rows[225:240]
    ]
    subset_counts: dict[str, int] = {}
    for row in rows:
        subset_counts[row["subset"]] = subset_counts.get(row["subset"], 0) + 1
    source_hashes = [row["image_sha256"] for row in rows]
    files = ("records.jsonl", "interventions.jsonl", "ideal_masks.npz")
    manifest = {
        "artifact_version": 2,
        "seed": seed,
        "holdout_policy": policy_path.relative_to(root).as_posix(),
        "holdout_policy_sha256": _sha(policy_path),
        "holdout_pairs": holdouts,
        "allowed_training_pairs": allowed,
        "record_count": len(rows),
        "subset_counts": subset_counts,
        "intervention_count": len(interventions),
        "intervention_attribute_counts": {
            name: sum(row["changed_attribute"] == name for row in interventions)
            for name in ("position", "shape", "color")
        },
        "stability_subset": {
            "record_ids": stable_ids,
            "seeds": [seed + 30001, seed + 30002, seed + 30003],
        },
        "source_image_unique_count": len(set(source_hashes)),
        "source_image_tree_sha256": hashlib.sha256("".join(source_hashes).encode("ascii")).hexdigest(),
        "files": {
            name: {"bytes": (output / name).stat().st_size, "sha256": _sha(output / name)}
            for name in files
        },
        "checks": {
            "seen_single_count": subset_counts.get("seen_single") == 225,
            "unseen_single_count": subset_counts.get("unseen_single") == 135,
            "multi_count": subset_counts.get("multi_2", 0) + subset_counts.get("multi_3", 0) == 60,
            "intervention_count": len(interventions) == 96,
            "intervention_attribute_counts": {
                name: sum(row["changed_attribute"] == name for row in interventions)
                for name in ("position", "shape", "color")
            } == {"position": 36, "shape": 30, "color": 30},
            "source_images_unique": len(set(source_hashes)) == len(rows),
            "stability_subset_count": len(stable_ids) == 30,
        },
    }
    manifest["verified"] = all(manifest["checks"].values())
    _write_json(output / "manifest.json", manifest)
    return manifest


def verify_fixed_evaluation_data(
    project_root: str | Path, data_dir: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    directory = Path(data_dir)
    if not directory.is_absolute():
        directory = root / directory
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != expected["bytes"] or _sha(path) != expected["sha256"]:
            raise ValueError(f"fixed evaluation file differs from manifest: {name}")
    if _sha(root / manifest["holdout_policy"]) != manifest["holdout_policy_sha256"]:
        raise ValueError("fixed evaluation holdout policy changed")
    if not manifest["verified"]:
        raise ValueError("fixed evaluation manifest is not verified")
    return manifest
