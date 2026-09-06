"""Appearance-anchored fallback for the bounded P3-W04 teaching language."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from mini_multimodal.vision.renderer import render_scene
from mini_multimodal.world.schema import Color, MotionSpec, Position, Scene, SceneObject, Shape, Size

from .motion_plan import MotionPlan, parse_motion_plan


_COLOR = {"红色": "red", "蓝色": "blue", "绿色": "green", "黄色": "yellow", "紫色": "purple", "橙色": "orange"}
_SHAPE = {"圆形": "circle", "正方形": "square", "三角形": "triangle", "矩形": "rectangle", "五边形": "pentagon"}
_APPEARANCE = re.compile(r"对象1是(红色|蓝色|绿色|黄色|紫色|橙色)的(小|中等大小|大)(圆形|正方形|三角形|矩形|五边形)")


@dataclass(frozen=True)
class ObjectAppearance:
    color: str
    size: str
    shape: str


def parse_object_appearance(prompt: str) -> ObjectAppearance:
    match = _APPEARANCE.search(prompt.split("；", 1)[0])
    if match is None:
        raise ValueError(f"unsupported object-1 appearance: {prompt}")
    color, size, shape = match.groups()
    return ObjectAppearance(_COLOR[color], {"小": "small", "中等大小": "medium", "大": "large"}[size], _SHAPE[shape])


def render_planned_video(prompt: str) -> tuple[np.ndarray, dict, MotionPlan]:
    """Render a canonical first-object anchor along the semantic motion plan."""
    appearance = parse_object_appearance(prompt)
    plan = parse_motion_plan(prompt)
    obj = SceneObject("obj_0", Shape(appearance.shape), Color(appearance.color), Size(appearance.size),
                      Position(0.5, 0.5), MotionSpec(), visible_start=0, visible_end=7)
    scene = Scene("planned_anchor", 0, 32, 32, 8, 4, (obj,))
    frames = []
    for frame, point in enumerate(plan.trajectory.tolist()):
        center = (round(10 + point[0] * 43), round(10 + point[1] * 43))
        rendered = render_scene(scene, frame_index=frame, center_overrides={"obj_0": center})
        frames.append(np.asarray(rendered.image, dtype=np.uint8))
    target = {"object_id": "obj_0", "shape": appearance.shape, "color": appearance.color,
              "size": appearance.size, "trajectory": plan.trajectory.tolist()}
    return np.stack(frames), target, plan
