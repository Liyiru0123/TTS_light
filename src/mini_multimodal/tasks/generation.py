"""Deterministic construction of QA tasks and generation conditions."""

from __future__ import annotations

from itertools import combinations

from mini_multimodal.world.schema import Scene

from .reasoning import answer_question, derive_events, visible_objects
from .schema import (
    EventRef,
    GenerationCondition,
    QATask,
    QuestionKind,
    QuestionSpec,
    RenderedGenerationCondition,
    TargetModality,
)
from .text_zh import ChineseTextAdapter


def build_question_specs(scene: Scene, *, frame: int = 0) -> tuple[QuestionSpec, ...]:
    objects = visible_objects(scene, frame)
    questions: list[QuestionSpec] = [
        QuestionSpec(
            question_id=f"{scene.scene_id}:q:count:f{frame}",
            scene_id=scene.scene_id,
            kind=QuestionKind.OBJECT_COUNT,
            frame=frame,
        )
    ]

    attribute_kinds = (
        QuestionKind.OBJECT_COLOR,
        QuestionKind.OBJECT_SHAPE,
        QuestionKind.OBJECT_SIZE,
        QuestionKind.MOTION_DIRECTION,
    )
    for obj in objects:
        for kind in attribute_kinds:
            questions.append(
                QuestionSpec(
                    question_id=f"{scene.scene_id}:q:{kind.value}:{obj.object_id}:f{frame}",
                    scene_id=scene.scene_id,
                    kind=kind,
                    frame=frame,
                    target_object_id=obj.object_id,
                )
            )

    for target, reference in combinations(objects, 2):
        for kind in (
            QuestionKind.HORIZONTAL_RELATION,
            QuestionKind.VERTICAL_RELATION,
        ):
            questions.append(
                QuestionSpec(
                    question_id=(
                        f"{scene.scene_id}:q:{kind.value}:{target.object_id}:"
                        f"{reference.object_id}:f{frame}"
                    ),
                    scene_id=scene.scene_id,
                    kind=kind,
                    frame=frame,
                    target_object_id=target.object_id,
                    reference_object_id=reference.object_id,
                )
            )

    events = derive_events(scene)
    if len(events) >= 2 and events[0].ref != events[-1].ref:
        questions.append(
            QuestionSpec(
                question_id=f"{scene.scene_id}:q:event_order:first_last",
                scene_id=scene.scene_id,
                kind=QuestionKind.EVENT_ORDER,
                frame=None,
                first_event=events[0].ref,
                second_event=events[-1].ref,
            )
        )
    return tuple(questions)


def build_qa_tasks(
    scene: Scene,
    adapter: ChineseTextAdapter | None = None,
    *,
    frame: int = 0,
) -> tuple[QATask, ...]:
    renderer = adapter or ChineseTextAdapter()
    tasks: list[QATask] = []
    for question in build_question_specs(scene, frame=frame):
        answer = answer_question(scene, question)
        tasks.append(
            QATask(
                question=question,
                answer=answer,
                question_zh=renderer.render_question(scene, question),
                answer_zh=renderer.render_answer(question, answer),
            )
        )
    return tuple(tasks)


def build_generation_conditions(
    scene: Scene,
    adapter: ChineseTextAdapter | None = None,
    *,
    frame: int = 0,
) -> tuple[RenderedGenerationCondition, ...]:
    renderer = adapter or ChineseTextAdapter()
    visible_ids = tuple(obj.object_id for obj in visible_objects(scene, frame))
    all_ids = tuple(obj.object_id for obj in scene.objects)
    conditions = (
        GenerationCondition(
            condition_id=f"{scene.scene_id}:condition:image:f{frame}",
            scene_id=scene.scene_id,
            target_modality=TargetModality.IMAGE,
            object_ids=visible_ids,
            frame=frame,
        ),
        GenerationCondition(
            condition_id=f"{scene.scene_id}:condition:speech:f{frame}",
            scene_id=scene.scene_id,
            target_modality=TargetModality.SPEECH,
            object_ids=visible_ids,
            frame=frame,
        ),
        GenerationCondition(
            condition_id=f"{scene.scene_id}:condition:video",
            scene_id=scene.scene_id,
            target_modality=TargetModality.VIDEO,
            object_ids=all_ids,
            frame=None,
        ),
    )
    return tuple(
        RenderedGenerationCondition(condition, renderer.render_condition(scene, condition))
        for condition in conditions
    )
