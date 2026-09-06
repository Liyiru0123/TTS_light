"""Validated, immutable semantic scene objects for the teaching world."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class Shape(str, Enum):
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"
    PENTAGON = "pentagon"


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    PURPLE = "purple"
    ORANGE = "orange"


class Size(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class MotionDirection(str, Enum):
    STATIC = "static"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

    @property
    def delta(self) -> tuple[int, int]:
        """Return a normalized direction vector; y grows downward."""

        return {
            MotionDirection.STATIC: (0, 0),
            MotionDirection.LEFT: (-1, 0),
            MotionDirection.RIGHT: (1, 0),
            MotionDirection.UP: (0, -1),
            MotionDirection.DOWN: (0, 1),
        }[self]


@dataclass(frozen=True, slots=True)
class Position:
    x: float
    y: float

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y)):
            if not isfinite(value):
                raise ValueError(f"Position {name} must be finite, got {value!r}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Position {name} must be within [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class MotionSpec:
    direction: MotionDirection = MotionDirection.STATIC
    speed: float = 0.0
    start_frame: int = 0
    end_frame: int = 0

    def __post_init__(self) -> None:
        if not isfinite(self.speed) or self.speed < 0.0:
            raise ValueError(f"Motion speed must be finite and non-negative, got {self.speed!r}")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("Motion frame interval must satisfy 0 <= start_frame <= end_frame")
        if self.direction is MotionDirection.STATIC and self.speed != 0.0:
            raise ValueError("Static motion must have zero speed")
        if self.direction is not MotionDirection.STATIC and self.speed <= 0.0:
            raise ValueError("Non-static motion must have positive speed")

    def active_steps_at(self, frame: int) -> int:
        if frame < 0:
            raise ValueError("Frame must be non-negative")
        return min(max(frame - self.start_frame, 0), self.end_frame - self.start_frame)


@dataclass(frozen=True, slots=True)
class SceneObject:
    object_id: str
    shape: Shape
    color: Color
    size: Size
    initial_position: Position
    motion: MotionSpec = MotionSpec()
    visible_start: int = 0
    visible_end: int = 0

    def __post_init__(self) -> None:
        if not self.object_id or not self.object_id.strip():
            raise ValueError("Object id must be a non-empty string")
        if self.visible_start < 0 or self.visible_end < self.visible_start:
            raise ValueError("Visibility interval must satisfy 0 <= visible_start <= visible_end")

    def position_at(self, frame: int) -> Position:
        steps = self.motion.active_steps_at(frame)
        dx, dy = self.motion.direction.delta
        return Position(
            x=self.initial_position.x + dx * self.motion.speed * steps,
            y=self.initial_position.y + dy * self.motion.speed * steps,
        )

    def is_visible_at(self, frame: int) -> bool:
        if frame < 0:
            raise ValueError("Frame must be non-negative")
        return self.visible_start <= frame <= self.visible_end


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    seed: int
    canvas_width: int
    canvas_height: int
    frame_count: int
    fps: int
    objects: tuple[SceneObject, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        if not self.scene_id or not self.scene_id.strip():
            raise ValueError("Scene id must be a non-empty string")
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        if self.frame_count <= 0 or self.fps <= 0:
            raise ValueError("Frame count and fps must be positive")
        if not self.objects:
            raise ValueError("A scene must contain at least one object")

        object_ids = [obj.object_id for obj in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("Object ids must be unique within a scene")

        last_frame = self.frame_count - 1
        for obj in self.objects:
            if obj.visible_end > last_frame:
                raise ValueError(f"Visibility interval for {obj.object_id} exceeds scene frames")
            if obj.motion.end_frame > last_frame:
                raise ValueError(f"Motion interval for {obj.object_id} exceeds scene frames")
            # Linear motion is monotonic, so validating both ends is sufficient.
            obj.position_at(0)
            obj.position_at(last_frame)

