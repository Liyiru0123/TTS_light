"""Structured tasks and deterministic text adapters for the teaching world."""

from .generation import build_generation_conditions, build_qa_tasks
from .reasoning import (
    answer_question,
    derive_events,
    horizontal_relation,
    position_region,
    vertical_relation,
)
from .schema import (
    AnswerSpec,
    EventRef,
    EventType,
    GenerationCondition,
    HorizontalRelation,
    PositionRegion,
    QATask,
    QuestionKind,
    QuestionSpec,
    RenderedGenerationCondition,
    SceneEvent,
    TargetModality,
    VerticalRelation,
)
from .text_zh import ChineseTextAdapter

__all__ = [
    "AnswerSpec",
    "ChineseTextAdapter",
    "EventRef",
    "EventType",
    "GenerationCondition",
    "HorizontalRelation",
    "PositionRegion",
    "QATask",
    "QuestionKind",
    "QuestionSpec",
    "RenderedGenerationCondition",
    "SceneEvent",
    "TargetModality",
    "VerticalRelation",
    "answer_question",
    "build_generation_conditions",
    "build_qa_tasks",
    "derive_events",
    "horizontal_relation",
    "position_region",
    "vertical_relation",
]
