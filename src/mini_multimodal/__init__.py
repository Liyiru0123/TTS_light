"""Core package for the multimodal teaching system prototype."""

from .world import (
    Color,
    MotionDirection,
    MotionSpec,
    Position,
    SamplerConfig,
    Scene,
    SceneObject,
    SceneSampler,
    Shape,
    Size,
    load_scene_json,
    save_scene_json,
    scene_from_dict,
    scene_to_dict,
)

__all__ = [
    "Color",
    "MotionDirection",
    "MotionSpec",
    "Position",
    "SamplerConfig",
    "Scene",
    "SceneObject",
    "SceneSampler",
    "Shape",
    "Size",
    "load_scene_json",
    "save_scene_json",
    "scene_from_dict",
    "scene_to_dict",
]

