"""Language-neutral schemas for questions, answers, events, and conditions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HorizontalRelation(str, Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ALIGNED = "aligned"


class VerticalRelation(str, Enum):
    ABOVE = "above"
    BELOW = "below"
    ALIGNED = "aligned"


class PositionRegion(str, Enum):
    UPPER_LEFT = "upper_left"
    UPPER_CENTER = "upper_center"
    UPPER_RIGHT = "upper_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    LOWER_LEFT = "lower_left"
    LOWER_CENTER = "lower_center"
    LOWER_RIGHT = "lower_right"


class EventType(str, Enum):
    APPEAR = "appear"
    DISAPPEAR = "disappear"
    MOVE_START = "move_start"
    MOVE_END = "move_end"


class QuestionKind(str, Enum):
    OBJECT_COLOR = "object_color"
    OBJECT_SHAPE = "object_shape"
    OBJECT_SIZE = "object_size"
    MOTION_DIRECTION = "motion_direction"
    OBJECT_COUNT = "object_count"
    HORIZONTAL_RELATION = "horizontal_relation"
    VERTICAL_RELATION = "vertical_relation"
    EVENT_ORDER = "event_order"


class TargetModality(str, Enum):
    IMAGE = "image"
    SPEECH = "speech"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class EventRef:
    object_id: str
    event_type: EventType

    def __post_init__(self) -> None:
        if not self.object_id or not self.object_id.strip():
            raise ValueError("Event object id must be non-empty")


@dataclass(frozen=True, slots=True)
class SceneEvent:
    object_id: str
    event_type: EventType
    frame: int

    def __post_init__(self) -> None:
        if not self.object_id or not self.object_id.strip():
            raise ValueError("Event object id must be non-empty")
        if self.frame < 0:
            raise ValueError("Event frame must be non-negative")

    @property
    def ref(self) -> EventRef:
        return EventRef(self.object_id, self.event_type)


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    question_id: str
    scene_id: str
    kind: QuestionKind
    frame: int | None = 0
    target_object_id: str | None = None
    reference_object_id: str | None = None
    first_event: EventRef | None = None
    second_event: EventRef | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported question schema version: {self.schema_version}")
        if not self.question_id or not self.question_id.strip():
            raise ValueError("Question id must be non-empty")
        if not self.scene_id or not self.scene_id.strip():
            raise ValueError("Question scene id must be non-empty")

        attribute_kinds = {
            QuestionKind.OBJECT_COLOR,
            QuestionKind.OBJECT_SHAPE,
            QuestionKind.OBJECT_SIZE,
            QuestionKind.MOTION_DIRECTION,
        }
        relation_kinds = {
            QuestionKind.HORIZONTAL_RELATION,
            QuestionKind.VERTICAL_RELATION,
        }

        if self.kind is QuestionKind.EVENT_ORDER:
            if self.frame is not None:
                raise ValueError("Event-order questions must not set a query frame")
            if self.first_event is None or self.second_event is None:
                raise ValueError("Event-order questions require two event references")
            if self.first_event == self.second_event:
                raise ValueError("Event-order questions require two distinct events")
            if self.target_object_id is not None or self.reference_object_id is not None:
                raise ValueError("Event-order questions must not set object query fields")
            return

        if self.frame is None or self.frame < 0:
            raise ValueError("Non-event questions require a non-negative query frame")
        if self.first_event is not None or self.second_event is not None:
            raise ValueError("Only event-order questions may set event references")

        if self.kind in attribute_kinds:
            if not self.target_object_id:
                raise ValueError("Attribute questions require a target object")
            if self.reference_object_id is not None:
                raise ValueError("Attribute questions must not set a reference object")
        elif self.kind is QuestionKind.OBJECT_COUNT:
            if self.target_object_id is not None or self.reference_object_id is not None:
                raise ValueError("Count questions must not set object query fields")
        elif self.kind in relation_kinds:
            if not self.target_object_id or not self.reference_object_id:
                raise ValueError("Relation questions require target and reference objects")
            if self.target_object_id == self.reference_object_id:
                raise ValueError("Relation questions require two distinct objects")
        else:
            raise ValueError(f"Unsupported question kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    value: str | int
    evidence_object_ids: tuple[str, ...]
    evidence_frames: tuple[int, ...]
    derivation: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported answer schema version: {self.schema_version}")
        if isinstance(self.value, str) and not self.value:
            raise ValueError("String answer value must be non-empty")
        if any(frame < 0 for frame in self.evidence_frames):
            raise ValueError("Evidence frames must be non-negative")
        if not self.derivation or not self.derivation.strip():
            raise ValueError("Answer derivation must be non-empty")


@dataclass(frozen=True, slots=True)
class QATask:
    question: QuestionSpec
    answer: AnswerSpec
    question_zh: str
    answer_zh: str

    def __post_init__(self) -> None:
        if not self.question_zh.strip() or not self.answer_zh.strip():
            raise ValueError("Rendered question and answer text must be non-empty")


@dataclass(frozen=True, slots=True)
class GenerationCondition:
    condition_id: str
    scene_id: str
    target_modality: TargetModality
    object_ids: tuple[str, ...]
    frame: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported condition schema version: {self.schema_version}")
        if not self.condition_id or not self.condition_id.strip():
            raise ValueError("Condition id must be non-empty")
        if not self.scene_id or not self.scene_id.strip():
            raise ValueError("Condition scene id must be non-empty")
        if len(self.object_ids) != len(set(self.object_ids)):
            raise ValueError("Condition object ids must be unique")
        if self.target_modality in {TargetModality.IMAGE, TargetModality.SPEECH}:
            if self.frame is None or self.frame < 0:
                raise ValueError("Image and speech conditions require a non-negative frame")
        elif self.target_modality is TargetModality.VIDEO:
            if self.frame is not None:
                raise ValueError("Video conditions describe the full sequence and must not set a frame")
            if not self.object_ids:
                raise ValueError("Video conditions require at least one scene object")


@dataclass(frozen=True, slots=True)
class RenderedGenerationCondition:
    condition: GenerationCondition
    text_zh: str

    def __post_init__(self) -> None:
        if not self.text_zh.strip():
            raise ValueError("Rendered generation condition must be non-empty")
