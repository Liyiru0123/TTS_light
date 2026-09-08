"""Schemas for scene-level splits and controlled dataset variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mini_multimodal.tasks.schema import QATask
from mini_multimodal.world.schema import Color, Scene, Shape


class DataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetVariant(str, Enum):
    NORMAL = "normal"
    SHORTCUT_IID = "shortcut_iid"
    COUNTERFACTUAL = "counterfactual"
    COMPOSITIONAL = "compositional"


DEFAULT_HOLDOUT_PAIRS: tuple[tuple[str, str], ...] = (
    (Color.RED.value, Shape.CIRCLE.value),
    (Color.BLUE.value, Shape.SQUARE.value),
    (Color.GREEN.value, Shape.TRIANGLE.value),
    (Color.YELLOW.value, Shape.RECTANGLE.value),
    (Color.PURPLE.value, Shape.PENTAGON.value),
)


@dataclass(frozen=True, slots=True)
class DatasetBuildConfig:
    base_seed: int = 20260807
    core_train_scenes: int = 600
    core_validation_scenes: int = 100
    core_test_scenes: int = 100
    shortcut_train_scenes: int = 300
    shortcut_validation_scenes: int = 60
    shortcut_test_scenes: int = 100
    compositional_train_scenes: int = 300
    compositional_validation_scenes: int = 60
    compositional_test_scenes: int = 100
    shortcut_strength: float = 0.9
    holdout_pairs: tuple[tuple[str, str], ...] = DEFAULT_HOLDOUT_PAIRS
    max_scan_count: int = 100_000
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported dataset config version: {self.schema_version}")
        counts = (
            self.core_train_scenes,
            self.core_validation_scenes,
            self.core_test_scenes,
            self.shortcut_train_scenes,
            self.shortcut_validation_scenes,
            self.shortcut_test_scenes,
            self.compositional_train_scenes,
            self.compositional_validation_scenes,
            self.compositional_test_scenes,
        )
        if any(count <= 0 for count in counts):
            raise ValueError("All configured scene counts must be positive")
        class_count = len(Shape)
        stratified_counts = counts[3:]
        if any(count % class_count != 0 for count in stratified_counts):
            raise ValueError(
                f"Shortcut and compositional counts must be divisible by {class_count}"
            )
        if not 0.0 < self.shortcut_strength < 1.0:
            raise ValueError("Shortcut strength must be strictly between 0 and 1")
        if self.max_scan_count <= 0:
            raise ValueError("max_scan_count must be positive")

        colors = {color.value for color in Color}
        shapes = {shape.value for shape in Shape}
        if len(self.holdout_pairs) != len(set(self.holdout_pairs)):
            raise ValueError("Holdout pairs must be unique")
        if {shape for _, shape in self.holdout_pairs} != shapes:
            raise ValueError("Holdout pairs must contain exactly one pair for every shape")
        if any(color not in colors or shape not in shapes for color, shape in self.holdout_pairs):
            raise ValueError("Holdout pairs contain an unknown color or shape")


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    record_id: str
    split: DataSplit
    variant: DatasetVariant
    task: QATask
    cue_token: str | None = None
    cue_aligned: bool | None = None
    composition_key: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported dataset record version: {self.schema_version}")
        if not self.record_id or not self.record_id.strip():
            raise ValueError("Dataset record id must be non-empty")
        if self.variant in {DatasetVariant.SHORTCUT_IID, DatasetVariant.COUNTERFACTUAL}:
            if not self.cue_token or self.cue_aligned is None:
                raise ValueError("Shortcut records require cue_token and cue_aligned")
            if self.composition_key is not None:
                raise ValueError("Shortcut records must not set a composition key")
        elif self.cue_token is not None or self.cue_aligned is not None:
            raise ValueError("Normal and compositional records must not carry shortcut cues")

        if self.variant is DatasetVariant.COMPOSITIONAL:
            if not self.composition_key:
                raise ValueError("Compositional records require a composition key")
        elif self.composition_key is not None:
            raise ValueError("Only compositional records may set a composition key")

    @property
    def scene_id(self) -> str:
        return self.task.question.scene_id


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    config: DatasetBuildConfig
    core_scenes: tuple[Scene, ...]
    core_records: tuple[DatasetRecord, ...]
    shortcut_scenes: tuple[Scene, ...]
    shortcut_records: tuple[DatasetRecord, ...]
    compositional_scenes: tuple[Scene, ...]
    compositional_records: tuple[DatasetRecord, ...]

    @property
    def all_scenes(self) -> tuple[Scene, ...]:
        return self.core_scenes + self.shortcut_scenes + self.compositional_scenes

    @property
    def all_records(self) -> tuple[DatasetRecord, ...]:
        return self.core_records + self.shortcut_records + self.compositional_records
