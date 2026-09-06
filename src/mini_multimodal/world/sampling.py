"""Deterministic sampling for semantic scenes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import dist
from random import Random

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


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    canvas_width: int = 32
    canvas_height: int = 32
    frame_count: int = 8
    fps: int = 8
    min_objects: int = 1
    max_objects: int = 3
    min_center_distance: float = 0.15
    min_speed: float = 0.02
    max_speed: float = 0.08
    max_position_attempts: int = 200
    shapes: tuple[Shape, ...] = tuple(Shape)
    colors: tuple[Color, ...] = tuple(Color)
    sizes: tuple[Size, ...] = tuple(Size)
    motions: tuple[MotionDirection, ...] = tuple(MotionDirection)

    def __post_init__(self) -> None:
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        if self.frame_count <= 0 or self.fps <= 0:
            raise ValueError("Frame count and fps must be positive")
        if self.min_objects <= 0 or self.max_objects < self.min_objects:
            raise ValueError("Object count bounds are invalid")
        if not 0.0 <= self.min_center_distance <= 1.0:
            raise ValueError("Minimum center distance must be within [0, 1]")
        if self.min_speed <= 0.0 or self.max_speed < self.min_speed:
            raise ValueError("Speed bounds must be positive and ordered")
        if self.max_speed * max(self.frame_count - 1, 0) > 1.0:
            raise ValueError("Maximum speed can move an object beyond the normalized canvas")
        if self.max_position_attempts <= 0:
            raise ValueError("max_position_attempts must be positive")
        if not self.shapes or not self.colors or not self.sizes or not self.motions:
            raise ValueError("Sampling vocabularies must not be empty")


class SceneSampler:
    def __init__(self, base_seed: int, config: SamplerConfig | None = None) -> None:
        self.base_seed = int(base_seed)
        self.config = config or SamplerConfig()

    def _seed_for_index(self, index: int) -> int:
        if index < 0:
            raise ValueError("Scene index must be non-negative")
        payload = f"mini-multimodal-world:{self.base_seed}:{index}".encode("utf-8")
        return int.from_bytes(sha256(payload).digest()[:8], "big", signed=False)

    def sample(self, index: int) -> Scene:
        scene_seed = self._seed_for_index(index)
        rng = Random(scene_seed)
        cfg = self.config
        object_count = rng.randint(cfg.min_objects, cfg.max_objects)
        positions: list[Position] = []
        objects: list[SceneObject] = []

        for object_index in range(object_count):
            direction = rng.choice(cfg.motions)
            if direction is MotionDirection.STATIC:
                speed = 0.0
                motion = MotionSpec()
            else:
                speed = rng.uniform(cfg.min_speed, cfg.max_speed)
                motion = MotionSpec(
                    direction=direction,
                    speed=speed,
                    start_frame=0,
                    end_frame=cfg.frame_count - 1,
                )

            position = self._sample_position(rng, motion, positions)
            positions.append(position)
            objects.append(
                SceneObject(
                    object_id=f"obj_{object_index}",
                    shape=rng.choice(cfg.shapes),
                    color=rng.choice(cfg.colors),
                    size=rng.choice(cfg.sizes),
                    initial_position=position,
                    motion=motion,
                    visible_start=0,
                    visible_end=cfg.frame_count - 1,
                )
            )

        return Scene(
            scene_id=f"scene_{index:06d}",
            seed=scene_seed,
            canvas_width=cfg.canvas_width,
            canvas_height=cfg.canvas_height,
            frame_count=cfg.frame_count,
            fps=cfg.fps,
            objects=tuple(objects),
        )

    def _sample_position(
        self,
        rng: Random,
        motion: MotionSpec,
        existing: list[Position],
    ) -> Position:
        cfg = self.config
        steps = motion.end_frame - motion.start_frame
        travel = motion.speed * steps
        x_low, x_high = 0.0, 1.0
        y_low, y_high = 0.0, 1.0

        if motion.direction is MotionDirection.RIGHT:
            x_high -= travel
        elif motion.direction is MotionDirection.LEFT:
            x_low += travel
        elif motion.direction is MotionDirection.DOWN:
            y_high -= travel
        elif motion.direction is MotionDirection.UP:
            y_low += travel

        if x_low > x_high or y_low > y_high:
            raise ValueError("Motion configuration leaves no valid starting position")

        for _ in range(cfg.max_position_attempts):
            candidate = Position(rng.uniform(x_low, x_high), rng.uniform(y_low, y_high))
            if all(
                dist((candidate.x, candidate.y), (other.x, other.y))
                >= cfg.min_center_distance
                for other in existing
            ):
                return candidate

        raise RuntimeError(
            "Unable to sample non-overlapping object centers; reduce object count or distance"
        )

