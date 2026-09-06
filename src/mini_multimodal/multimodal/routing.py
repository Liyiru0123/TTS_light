"""Explicit evidence-modality routing for structured QA tasks."""

from __future__ import annotations

from dataclasses import dataclass

from mini_multimodal.tasks.schema import QuestionKind


STATIC_VISUAL_KINDS = frozenset(
    {
        QuestionKind.OBJECT_COLOR,
        QuestionKind.OBJECT_SHAPE,
        QuestionKind.OBJECT_SIZE,
        QuestionKind.OBJECT_COUNT,
        QuestionKind.HORIZONTAL_RELATION,
        QuestionKind.VERTICAL_RELATION,
    }
)
TEMPORAL_VISUAL_KINDS = frozenset(
    {QuestionKind.MOTION_DIRECTION, QuestionKind.EVENT_ORDER}
)


@dataclass(frozen=True, slots=True)
class TaskRoute:
    capability: str
    supported_modalities: tuple[str, ...]

    def supports(self, modality: str) -> bool:
        return modality in self.supported_modalities

    def to_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "supported_modalities": list(self.supported_modalities),
        }


def route_question_kind(kind: QuestionKind | str) -> TaskRoute:
    question_kind = kind if isinstance(kind, QuestionKind) else QuestionKind(str(kind))
    if question_kind in STATIC_VISUAL_KINDS:
        return TaskRoute("static_visual", ("image", "video"))
    if question_kind in TEMPORAL_VISUAL_KINDS:
        return TaskRoute("temporal_visual", ("video",))
    raise ValueError(f"no modality route for question kind: {question_kind.value}")


def route_table() -> dict[str, dict[str, object]]:
    return {kind.value: route_question_kind(kind).to_dict() for kind in QuestionKind}

