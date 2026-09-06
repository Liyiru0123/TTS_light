"""JSON-compatible serialization for structured task records."""

from __future__ import annotations

from typing import Any, Mapping

from .schema import (
    AnswerSpec,
    EventRef,
    EventType,
    GenerationCondition,
    QATask,
    QuestionKind,
    QuestionSpec,
    RenderedGenerationCondition,
    TargetModality,
)


def _event_ref_to_dict(event: EventRef | None) -> dict[str, str] | None:
    if event is None:
        return None
    return {"object_id": event.object_id, "event_type": event.event_type.value}


def _event_ref_from_dict(data: Mapping[str, Any] | None) -> EventRef | None:
    if data is None:
        return None
    return EventRef(str(data["object_id"]), EventType(data["event_type"]))


def question_to_dict(question: QuestionSpec) -> dict[str, Any]:
    return {
        "schema_version": question.schema_version,
        "question_id": question.question_id,
        "scene_id": question.scene_id,
        "kind": question.kind.value,
        "frame": question.frame,
        "target_object_id": question.target_object_id,
        "reference_object_id": question.reference_object_id,
        "first_event": _event_ref_to_dict(question.first_event),
        "second_event": _event_ref_to_dict(question.second_event),
    }


def question_from_dict(data: Mapping[str, Any]) -> QuestionSpec:
    frame = data.get("frame")
    return QuestionSpec(
        schema_version=int(data["schema_version"]),
        question_id=str(data["question_id"]),
        scene_id=str(data["scene_id"]),
        kind=QuestionKind(data["kind"]),
        frame=None if frame is None else int(frame),
        target_object_id=data.get("target_object_id"),
        reference_object_id=data.get("reference_object_id"),
        first_event=_event_ref_from_dict(data.get("first_event")),
        second_event=_event_ref_from_dict(data.get("second_event")),
    )


def answer_to_dict(answer: AnswerSpec) -> dict[str, Any]:
    return {
        "schema_version": answer.schema_version,
        "value": answer.value,
        "evidence_object_ids": list(answer.evidence_object_ids),
        "evidence_frames": list(answer.evidence_frames),
        "derivation": answer.derivation,
    }


def answer_from_dict(data: Mapping[str, Any]) -> AnswerSpec:
    value = data["value"]
    return AnswerSpec(
        schema_version=int(data["schema_version"]),
        value=int(value) if isinstance(value, int) else str(value),
        evidence_object_ids=tuple(str(item) for item in data["evidence_object_ids"]),
        evidence_frames=tuple(int(item) for item in data["evidence_frames"]),
        derivation=str(data["derivation"]),
    )


def qa_task_to_dict(task: QATask) -> dict[str, Any]:
    return {
        "record_type": "qa",
        "question": question_to_dict(task.question),
        "answer": answer_to_dict(task.answer),
        "question_zh": task.question_zh,
        "answer_zh": task.answer_zh,
    }


def qa_task_from_dict(data: Mapping[str, Any]) -> QATask:
    if data.get("record_type") != "qa":
        raise ValueError("Expected a QA task record")
    return QATask(
        question=question_from_dict(data["question"]),
        answer=answer_from_dict(data["answer"]),
        question_zh=str(data["question_zh"]),
        answer_zh=str(data["answer_zh"]),
    )


def condition_to_dict(condition: RenderedGenerationCondition) -> dict[str, Any]:
    spec = condition.condition
    return {
        "record_type": "generation_condition",
        "condition": {
            "schema_version": spec.schema_version,
            "condition_id": spec.condition_id,
            "scene_id": spec.scene_id,
            "target_modality": spec.target_modality.value,
            "object_ids": list(spec.object_ids),
            "frame": spec.frame,
        },
        "text_zh": condition.text_zh,
    }


def condition_from_dict(data: Mapping[str, Any]) -> RenderedGenerationCondition:
    if data.get("record_type") != "generation_condition":
        raise ValueError("Expected a generation-condition record")
    raw = data["condition"]
    frame = raw.get("frame")
    return RenderedGenerationCondition(
        condition=GenerationCondition(
            schema_version=int(raw["schema_version"]),
            condition_id=str(raw["condition_id"]),
            scene_id=str(raw["scene_id"]),
            target_modality=TargetModality(raw["target_modality"]),
            object_ids=tuple(str(item) for item in raw["object_ids"]),
            frame=None if frame is None else int(frame),
        ),
        text_zh=str(data["text_zh"]),
    )
