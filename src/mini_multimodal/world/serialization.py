"""JSON-compatible serialization for semantic scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    Color,
    MotionDirection,
    MotionSpec,
    Position,
    Scene,
    SceneObject,
    Shape,
    Size,
)


def scene_to_dict(scene: Scene) -> dict[str, Any]:
    return {
        "schema_version": scene.schema_version,
        "scene_id": scene.scene_id,
        "seed": scene.seed,
        "canvas_width": scene.canvas_width,
        "canvas_height": scene.canvas_height,
        "frame_count": scene.frame_count,
        "fps": scene.fps,
        "objects": [
            {
                "object_id": obj.object_id,
                "shape": obj.shape.value,
                "color": obj.color.value,
                "size": obj.size.value,
                "initial_position": {
                    "x": obj.initial_position.x,
                    "y": obj.initial_position.y,
                },
                "motion": {
                    "direction": obj.motion.direction.value,
                    "speed": obj.motion.speed,
                    "start_frame": obj.motion.start_frame,
                    "end_frame": obj.motion.end_frame,
                },
                "visible_start": obj.visible_start,
                "visible_end": obj.visible_end,
            }
            for obj in scene.objects
        ],
    }


def scene_from_dict(data: Mapping[str, Any]) -> Scene:
    objects = []
    for raw in data["objects"]:
        position = raw["initial_position"]
        motion = raw["motion"]
        objects.append(
            SceneObject(
                object_id=str(raw["object_id"]),
                shape=Shape(raw["shape"]),
                color=Color(raw["color"]),
                size=Size(raw["size"]),
                initial_position=Position(float(position["x"]), float(position["y"])),
                motion=MotionSpec(
                    direction=MotionDirection(motion["direction"]),
                    speed=float(motion["speed"]),
                    start_frame=int(motion["start_frame"]),
                    end_frame=int(motion["end_frame"]),
                ),
                visible_start=int(raw["visible_start"]),
                visible_end=int(raw["visible_end"]),
            )
        )

    return Scene(
        schema_version=int(data["schema_version"]),
        scene_id=str(data["scene_id"]),
        seed=int(data["seed"]),
        canvas_width=int(data["canvas_width"]),
        canvas_height=int(data["canvas_height"]),
        frame_count=int(data["frame_count"]),
        fps=int(data["fps"]),
        objects=tuple(objects),
    )


def save_scene_json(scene: Scene, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(scene_to_dict(scene), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_scene_json(path: str | Path) -> Scene:
    source = Path(path)
    return scene_from_dict(json.loads(source.read_text(encoding="utf-8")))

