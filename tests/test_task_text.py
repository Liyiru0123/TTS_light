from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mini_multimodal.tasks import (  # noqa: E402
    ChineseTextAdapter,
    EventRef,
    EventType,
    HorizontalRelation,
    PositionRegion,
    QuestionKind,
    QuestionSpec,
    TargetModality,
    VerticalRelation,
    answer_question,
    build_generation_conditions,
    build_qa_tasks,
    derive_events,
    horizontal_relation,
    position_region,
    vertical_relation,
)
from mini_multimodal.tasks.artifacts import (  # noqa: E402
    export_task_fixture,
    load_task_fixture,
    verify_task_fixture,
)
from mini_multimodal.tasks.examples import make_temporal_example_scene  # noqa: E402
from mini_multimodal.tasks.serialization import (  # noqa: E402
    condition_from_dict,
    condition_to_dict,
    qa_task_from_dict,
    qa_task_to_dict,
)
from mini_multimodal.tasks.text_zh import (  # noqa: E402
    COLOR_ZH,
    MOTION_ZH,
    REGION_ZH,
    SHAPE_ZH,
    SIZE_ZH,
)
from mini_multimodal.world import (  # noqa: E402
    Color,
    MotionDirection,
    MotionSpec,
    Position,
    SamplerConfig,
    Scene,
    SceneObject,
    SceneSampler,
    Shape,
    Size,
)


def make_relation_scene() -> Scene:
    return Scene(
        scene_id="scene_relation",
        seed=101,
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
                visible_start=0,
                visible_end=7,
            ),
            SceneObject(
                object_id="obj_1",
                shape=Shape.SQUARE,
                color=Color.BLUE,
                size=Size.LARGE,
                initial_position=Position(0.45, 0.7),
                motion=MotionSpec(),
                visible_start=0,
                visible_end=7,
            ),
        ),
    )


def make_temporal_scene() -> Scene:
    return make_temporal_example_scene()


def question(scene: Scene, kind: QuestionKind, **kwargs: object) -> QuestionSpec:
    return QuestionSpec(
        question_id=f"test:{kind.value}:{len(kwargs)}",
        scene_id=scene.scene_id,
        kind=kind,
        **kwargs,
    )


class TaskReasoningTests(unittest.TestCase):
    def test_task_unit_001_horizontal_relation(self) -> None:
        scene = make_relation_scene()
        left, right = scene.objects
        self.assertEqual(horizontal_relation(left, right, 0), HorizontalRelation.LEFT_OF)
        self.assertEqual(horizontal_relation(right, left, 0), HorizontalRelation.RIGHT_OF)

    def test_task_unit_002_vertical_relation(self) -> None:
        scene = make_relation_scene()
        upper, lower = scene.objects
        self.assertEqual(vertical_relation(upper, lower, 0), VerticalRelation.ABOVE)
        self.assertEqual(vertical_relation(lower, upper, 0), VerticalRelation.BELOW)

    def test_task_unit_003_relation_uses_queried_frame(self) -> None:
        scene = make_relation_scene()
        spec = question(
            scene,
            QuestionKind.HORIZONTAL_RELATION,
            frame=4,
            target_object_id="obj_0",
            reference_object_id="obj_1",
        )
        self.assertEqual(answer_question(scene, spec).value, "right_of")

    def test_task_unit_004_position_regions(self) -> None:
        self.assertEqual(position_region(Position(0.0, 0.0)), PositionRegion.UPPER_LEFT)
        self.assertEqual(position_region(Position(0.5, 0.5)), PositionRegion.CENTER)
        self.assertEqual(position_region(Position(1.0, 1.0)), PositionRegion.LOWER_RIGHT)

    def test_task_unit_005_event_derivation(self) -> None:
        actual = {
            (event.object_id, event.event_type, event.frame)
            for event in derive_events(make_temporal_scene())
        }
        expected = {
            ("obj_0", EventType.APPEAR, 1),
            ("obj_0", EventType.MOVE_START, 1),
            ("obj_0", EventType.MOVE_END, 4),
            ("obj_1", EventType.APPEAR, 0),
            ("obj_1", EventType.DISAPPEAR, 6),
        }
        self.assertEqual(actual, expected)

    def test_task_unit_006_event_order_first_second_and_same(self) -> None:
        scene = make_temporal_scene()
        cases = (
            (EventRef("obj_1", EventType.APPEAR), EventRef("obj_0", EventType.APPEAR), "first_event"),
            (EventRef("obj_0", EventType.APPEAR), EventRef("obj_1", EventType.APPEAR), "second_event"),
            (EventRef("obj_0", EventType.APPEAR), EventRef("obj_0", EventType.MOVE_START), "same_time"),
        )
        for first, second, expected in cases:
            with self.subTest(expected=expected):
                spec = question(
                    scene,
                    QuestionKind.EVENT_ORDER,
                    frame=None,
                    first_event=first,
                    second_event=second,
                )
                self.assertEqual(answer_question(scene, spec).value, expected)

    def test_task_unit_007_attribute_answers_and_evidence(self) -> None:
        scene = make_relation_scene()
        cases = {
            QuestionKind.OBJECT_COLOR: "red",
            QuestionKind.OBJECT_SHAPE: "circle",
            QuestionKind.OBJECT_SIZE: "small",
            QuestionKind.MOTION_DIRECTION: "right",
        }
        for kind, expected in cases.items():
            with self.subTest(kind=kind):
                spec = question(scene, kind, frame=0, target_object_id="obj_0")
                answer = answer_question(scene, spec)
                self.assertEqual(answer.value, expected)
                self.assertEqual(answer.evidence_object_ids, ("obj_0",))

    def test_task_unit_008_count_respects_visibility(self) -> None:
        scene = make_temporal_scene()
        for frame, expected in ((0, 1), (1, 2), (6, 1)):
            with self.subTest(frame=frame):
                spec = question(scene, QuestionKind.OBJECT_COUNT, frame=frame)
                self.assertEqual(answer_question(scene, spec).value, expected)

    def test_task_unit_009_relation_answer_evidence(self) -> None:
        scene = make_relation_scene()
        spec = question(
            scene,
            QuestionKind.VERTICAL_RELATION,
            frame=0,
            target_object_id="obj_0",
            reference_object_id="obj_1",
        )
        answer = answer_question(scene, spec)
        self.assertEqual(answer.value, "above")
        self.assertEqual(answer.evidence_object_ids, ("obj_0", "obj_1"))
        self.assertEqual(answer.evidence_frames, (0,))

    def test_task_unit_010_rejects_invalid_questions(self) -> None:
        scene = make_relation_scene()
        unknown = question(
            scene, QuestionKind.OBJECT_COLOR, frame=0, target_object_id="missing"
        )
        with self.assertRaisesRegex(ValueError, "Unknown object"):
            answer_question(scene, unknown)
        with self.assertRaisesRegex(ValueError, "distinct"):
            question(
                scene,
                QuestionKind.HORIZONTAL_RELATION,
                frame=0,
                target_object_id="obj_0",
                reference_object_id="obj_0",
            )
        outside = question(scene, QuestionKind.OBJECT_COUNT, frame=8)
        with self.assertRaisesRegex(ValueError, "outside"):
            answer_question(scene, outside)


class ChineseTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = make_relation_scene()
        self.adapter = ChineseTextAdapter()

    def test_text_unit_001_all_fixed_vocabularies_are_translated(self) -> None:
        self.assertEqual(set(COLOR_ZH), {item.value for item in Color})
        self.assertEqual(set(SHAPE_ZH), {item.value for item in Shape})
        self.assertEqual(set(SIZE_ZH), {item.value for item in Size})
        self.assertEqual(set(MOTION_ZH), {item.value for item in MotionDirection})
        self.assertEqual(set(REGION_ZH), {item.value for item in PositionRegion})
        self.assertEqual([ord(char) for char in COLOR_ZH["red"]], [0x7EA2, 0x8272])
        rendered = ChineseTextAdapter().describe_scene(self.scene, 0)
        self.assertFalse(any(marker in rendered for marker in ("Ã", "Â", "ç", "å", "æ")))

    def test_text_unit_002_object_numbers_disambiguate_same_attributes(self) -> None:
        duplicate_scene = Scene(
            scene_id="duplicates",
            seed=1,
            canvas_width=32,
            canvas_height=32,
            frame_count=1,
            fps=1,
            objects=(
                SceneObject("a", Shape.CIRCLE, Color.RED, Size.SMALL, Position(0.2, 0.2)),
                SceneObject("b", Shape.CIRCLE, Color.RED, Size.SMALL, Position(0.8, 0.8)),
            ),
        )
        self.assertIn("对象1", self.adapter.object_ref(duplicate_scene, "a", with_attributes=True))
        self.assertIn("对象2", self.adapter.object_ref(duplicate_scene, "b", with_attributes=True))

    def test_text_unit_003_question_and_answer_match_structure(self) -> None:
        spec = question(
            self.scene, QuestionKind.OBJECT_COLOR, frame=0, target_object_id="obj_0"
        )
        answer = answer_question(self.scene, spec)
        self.assertEqual(self.adapter.render_question(self.scene, spec), "对象1是什么颜色？")
        self.assertEqual(self.adapter.render_answer(spec, answer), "红色")

    def test_text_unit_004_internal_frame_zero_displays_as_frame_one(self) -> None:
        spec = question(self.scene, QuestionKind.OBJECT_COUNT, frame=0)
        self.assertEqual(self.adapter.render_question(self.scene, spec), "第1帧中有几个可见对象？")


class GenerationConditionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = make_temporal_scene()
        self.conditions = {
            item.condition.target_modality: item
            for item in build_generation_conditions(self.scene)
        }

    def test_gen_unit_001_image_condition_is_frame_specific(self) -> None:
        item = self.conditions[TargetModality.IMAGE]
        self.assertEqual(item.condition.object_ids, ("obj_1",))
        self.assertIn("1个可见对象", item.text_zh)
        self.assertNotIn("移动", item.text_zh)
        hidden_scene = Scene(
            scene_id="hidden_at_start",
            seed=3,
            canvas_width=32,
            canvas_height=32,
            frame_count=2,
            fps=1,
            objects=(
                SceneObject(
                    "late",
                    Shape.CIRCLE,
                    Color.RED,
                    Size.SMALL,
                    Position(0.5, 0.5),
                    visible_start=1,
                    visible_end=1,
                ),
            ),
        )
        blank = {
            value.condition.target_modality: value
            for value in build_generation_conditions(hidden_scene)
        }[TargetModality.IMAGE]
        self.assertEqual(blank.condition.object_ids, ())
        self.assertIn("0个可见对象", blank.text_zh)

    def test_gen_unit_002_speech_condition_is_traceable_utterance(self) -> None:
        item = self.conditions[TargetModality.SPEECH]
        self.assertEqual(item.condition.scene_id, self.scene.scene_id)
        self.assertEqual(item.condition.object_ids, ("obj_1",))
        self.assertTrue(item.text_zh.endswith("。"))

    def test_gen_unit_003_video_condition_contains_dynamics(self) -> None:
        item = self.conditions[TargetModality.VIDEO]
        self.assertIn("向右移动", item.text_zh)
        self.assertIn("第2帧至第8帧可见", item.text_zh)
        self.assertIsNone(item.condition.frame)


class TaskIntegrationTests(unittest.TestCase):
    def test_task_data_001_generation_is_deterministic(self) -> None:
        scene = make_relation_scene()
        self.assertEqual(build_qa_tasks(scene), build_qa_tasks(scene))
        self.assertEqual(
            build_generation_conditions(scene), build_generation_conditions(scene)
        )

    def test_task_data_002_batch_covers_tasks_and_modalities(self) -> None:
        sampler = SceneSampler(2026, SamplerConfig())
        kinds: set[QuestionKind] = set()
        modalities: set[TargetModality] = set()
        for index in range(200):
            scene = sampler.sample(index)
            kinds.update(task.question.kind for task in build_qa_tasks(scene))
            modalities.update(
                item.condition.target_modality
                for item in build_generation_conditions(scene)
            )
        self.assertEqual(kinds, set(QuestionKind))
        self.assertEqual(modalities, set(TargetModality))

    def test_task_int_001_json_compatible_round_trip(self) -> None:
        scene = make_relation_scene()
        task = build_qa_tasks(scene)[0]
        condition = build_generation_conditions(scene)[0]
        self.assertEqual(qa_task_from_dict(qa_task_to_dict(task)), task)
        self.assertEqual(condition_from_dict(condition_to_dict(condition)), condition)

    def test_task_int_002_fixture_export_and_verification(self) -> None:
        scenes = [SceneSampler(88).sample(index) for index in range(10)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = export_task_fixture(tmp_dir, scenes)
            verification = verify_task_fixture(tmp_dir)
            tasks, conditions = load_task_fixture(tmp_dir)
            self.assertTrue(verification["verified"])
            self.assertEqual(len(tasks), manifest["statistics"]["qa_task_count"])
            self.assertEqual(
                len(conditions), manifest["statistics"]["generation_condition_count"]
            )

    def test_task_int_003_fixture_tampering_is_detected(self) -> None:
        scenes = [SceneSampler(99).sample(index) for index in range(3)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_task_fixture(tmp_dir, scenes)
            path = Path(tmp_dir) / "qa_tasks.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_task_fixture(tmp_dir)


if __name__ == "__main__":
    unittest.main()
