"""Deterministic reasoning over validated semantic scenes."""

from __future__ import annotations

from math import isclose

from mini_multimodal.world.schema import MotionDirection, Position, Scene, SceneObject

from .schema import (
    AnswerSpec,
    EventRef,
    EventType,
    HorizontalRelation,
    PositionRegion,
    QuestionKind,
    QuestionSpec,
    SceneEvent,
    VerticalRelation,
)


def object_by_id(scene: Scene, object_id: str) -> SceneObject:
    for obj in scene.objects:
        if obj.object_id == object_id:
            return obj
    raise ValueError(f"Unknown object id {object_id!r} in scene {scene.scene_id!r}")


def _validate_frame(scene: Scene, frame: int) -> None:
    if not 0 <= frame < scene.frame_count:
        raise ValueError(
            f"Frame {frame} is outside scene range [0, {scene.frame_count - 1}]"
        )


def visible_objects(scene: Scene, frame: int) -> tuple[SceneObject, ...]:
    _validate_frame(scene, frame)
    return tuple(obj for obj in scene.objects if obj.is_visible_at(frame))


def horizontal_relation(
    target: SceneObject,
    reference: SceneObject,
    frame: int,
    *,
    tolerance: float = 1e-9,
) -> HorizontalRelation:
    target_x = target.position_at(frame).x
    reference_x = reference.position_at(frame).x
    if isclose(target_x, reference_x, abs_tol=tolerance):
        return HorizontalRelation.ALIGNED
    return (
        HorizontalRelation.LEFT_OF
        if target_x < reference_x
        else HorizontalRelation.RIGHT_OF
    )


def vertical_relation(
    target: SceneObject,
    reference: SceneObject,
    frame: int,
    *,
    tolerance: float = 1e-9,
) -> VerticalRelation:
    target_y = target.position_at(frame).y
    reference_y = reference.position_at(frame).y
    if isclose(target_y, reference_y, abs_tol=tolerance):
        return VerticalRelation.ALIGNED
    return VerticalRelation.ABOVE if target_y < reference_y else VerticalRelation.BELOW


def position_region(position: Position) -> PositionRegion:
    horizontal = "left" if position.x < 1 / 3 else "right" if position.x >= 2 / 3 else "center"
    vertical = "upper" if position.y < 1 / 3 else "lower" if position.y >= 2 / 3 else "center"
    value = "center" if horizontal == vertical == "center" else f"{vertical}_{horizontal}"
    return PositionRegion(value)


def derive_events(scene: Scene) -> tuple[SceneEvent, ...]:
    events: list[SceneEvent] = []
    for obj in scene.objects:
        events.append(SceneEvent(obj.object_id, EventType.APPEAR, obj.visible_start))
        if obj.visible_end < scene.frame_count - 1:
            events.append(
                SceneEvent(obj.object_id, EventType.DISAPPEAR, obj.visible_end + 1)
            )
        if obj.motion.direction is not MotionDirection.STATIC:
            events.extend(
                (
                    SceneEvent(obj.object_id, EventType.MOVE_START, obj.motion.start_frame),
                    SceneEvent(obj.object_id, EventType.MOVE_END, obj.motion.end_frame),
                )
            )

    object_order = {obj.object_id: index for index, obj in enumerate(scene.objects)}
    event_order = {event_type: index for index, event_type in enumerate(EventType)}
    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.frame,
                object_order[event.object_id],
                event_order[event.event_type],
            ),
        )
    )


def resolve_event(scene: Scene, event_ref: EventRef) -> SceneEvent:
    for event in derive_events(scene):
        if event.ref == event_ref:
            return event
    raise ValueError(
        f"Scene {scene.scene_id!r} has no event {event_ref.event_type.value!r} "
        f"for object {event_ref.object_id!r}"
    )


def _require_visible(obj: SceneObject, frame: int) -> None:
    if not obj.is_visible_at(frame):
        raise ValueError(f"Object {obj.object_id!r} is not visible at frame {frame}")


def answer_question(scene: Scene, question: QuestionSpec) -> AnswerSpec:
    if question.scene_id != scene.scene_id:
        raise ValueError(
            f"Question scene {question.scene_id!r} does not match {scene.scene_id!r}"
        )

    if question.kind is QuestionKind.EVENT_ORDER:
        assert question.first_event is not None and question.second_event is not None
        first = resolve_event(scene, question.first_event)
        second = resolve_event(scene, question.second_event)
        value = "same_time"
        if first.frame < second.frame:
            value = "first_event"
        elif first.frame > second.frame:
            value = "second_event"
        return AnswerSpec(
            value=value,
            evidence_object_ids=(first.object_id, second.object_id),
            evidence_frames=(first.frame, second.frame),
            derivation="compare_event_frames",
        )

    assert question.frame is not None
    _validate_frame(scene, question.frame)

    if question.kind is QuestionKind.OBJECT_COUNT:
        objects = visible_objects(scene, question.frame)
        return AnswerSpec(
            value=len(objects),
            evidence_object_ids=tuple(obj.object_id for obj in objects),
            evidence_frames=(question.frame,),
            derivation="count_visible_objects",
        )

    assert question.target_object_id is not None
    target = object_by_id(scene, question.target_object_id)

    if question.kind in {
        QuestionKind.OBJECT_COLOR,
        QuestionKind.OBJECT_SHAPE,
        QuestionKind.OBJECT_SIZE,
    }:
        _require_visible(target, question.frame)

    attribute_values = {
        QuestionKind.OBJECT_COLOR: target.color.value,
        QuestionKind.OBJECT_SHAPE: target.shape.value,
        QuestionKind.OBJECT_SIZE: target.size.value,
        QuestionKind.MOTION_DIRECTION: target.motion.direction.value,
    }
    if question.kind in attribute_values:
        return AnswerSpec(
            value=attribute_values[question.kind],
            evidence_object_ids=(target.object_id,),
            evidence_frames=(question.frame,),
            derivation=f"read_{question.kind.value}",
        )

    assert question.reference_object_id is not None
    reference = object_by_id(scene, question.reference_object_id)
    _require_visible(target, question.frame)
    _require_visible(reference, question.frame)

    if question.kind is QuestionKind.HORIZONTAL_RELATION:
        value = horizontal_relation(target, reference, question.frame).value
        derivation = "compare_x_coordinates"
    elif question.kind is QuestionKind.VERTICAL_RELATION:
        value = vertical_relation(target, reference, question.frame).value
        derivation = "compare_y_coordinates"
    else:  # Guard against future enum additions without reasoning support.
        raise ValueError(f"Unsupported question kind: {question.kind}")

    return AnswerSpec(
        value=value,
        evidence_object_ids=(target.object_id, reference.object_id),
        evidence_frames=(question.frame,),
        derivation=derivation,
    )
