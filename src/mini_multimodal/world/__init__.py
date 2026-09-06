"""Language- and modality-neutral scene specifications."""

from .sampling import SamplerConfig, SceneSampler
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
from .serialization import (
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

