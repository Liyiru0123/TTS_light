"""Canonical edge-case scenes used by task tests and persistent fixtures."""

from mini_multimodal.world.schema import (
    Color,
    MotionDirection,
    MotionSpec,
    Position,
    Scene,
    SceneObject,
    Shape,
    Size,
)


def make_temporal_example_scene() -> Scene:
    """Return a scene with delayed appearance, motion, and early disappearance."""

    return Scene(
        scene_id="scene_temporal_edge",
        seed=20260807,
        canvas_width=32,
        canvas_height=32,
        frame_count=8,
        fps=8,
        objects=(
            SceneObject(
                object_id="obj_0",
                shape=Shape.CIRCLE,
                color=Color.RED,
                size=Size.SMALL,
                initial_position=Position(0.2, 0.2),
                motion=MotionSpec(MotionDirection.RIGHT, 0.1, 1, 4),
                visible_start=1,
                visible_end=7,
            ),
            SceneObject(
                object_id="obj_1",
                shape=Shape.SQUARE,
                color=Color.BLUE,
                size=Size.LARGE,
                initial_position=Position(0.75, 0.75),
                motion=MotionSpec(),
                visible_start=0,
                visible_end=5,
            ),
        ),
    )
