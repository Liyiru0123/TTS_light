from __future__ import annotations

import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

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
    load_scene_json,
    save_scene_json,
    scene_from_dict,
    scene_to_dict,
)
from mini_multimodal.world.artifacts import (  # noqa: E402
    export_scene_fixture,
    load_scene_fixture,
    verify_scene_fixture,
)


def make_object(**changes: object) -> SceneObject:
    base = SceneObject(
        object_id="obj_0",
        shape=Shape.CIRCLE,
        color=Color.RED,
        size=Size.SMALL,
        initial_position=Position(0.25, 0.5),
        motion=MotionSpec(),
        visible_start=0,
        visible_end=7,
    )
    return replace(base, **changes)


def make_scene(*objects: SceneObject) -> Scene:
    return Scene(
        scene_id="scene_test",
        seed=7,
        canvas_width=32,
        canvas_height=32,
        frame_count=8,
        fps=8,
        objects=objects or (make_object(),),
    )


class SchemaTests(unittest.TestCase):
    def test_unit_001_valid_static_object(self) -> None:
        scene = make_scene(make_object())
        self.assertEqual(scene.objects[0].position_at(7), Position(0.25, 0.5))

    def test_unit_002_rejects_invalid_position(self) -> None:
        for x, y in ((-0.01, 0.5), (1.01, 0.5), (0.5, math.inf)):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                Position(x, y)

    def test_unit_003_rejects_inconsistent_speed(self) -> None:
        with self.assertRaises(ValueError):
            MotionSpec(direction=MotionDirection.STATIC, speed=0.1)
        with self.assertRaises(ValueError):
            MotionSpec(direction=MotionDirection.RIGHT, speed=0.0, end_frame=7)

    def test_unit_004_position_before_during_after_motion(self) -> None:
        obj = make_object(
            motion=MotionSpec(
                direction=MotionDirection.RIGHT,
                speed=0.1,
                start_frame=2,
                end_frame=5,
            )
        )
        self.assertEqual(obj.position_at(1), Position(0.25, 0.5))
        self.assertAlmostEqual(obj.position_at(4).x, 0.45)
        self.assertAlmostEqual(obj.position_at(7).x, 0.55)

    def test_unit_005_rejects_out_of_bounds_trajectory(self) -> None:
        obj = make_object(
            initial_position=Position(0.9, 0.5),
            motion=MotionSpec(
                direction=MotionDirection.RIGHT,
                speed=0.05,
                start_frame=0,
                end_frame=7,
            ),
        )
        with self.assertRaises(ValueError):
            make_scene(obj)

    def test_unit_006_rejects_duplicate_object_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            make_scene(make_object(), make_object())

    def test_unit_007_rejects_visibility_outside_scene(self) -> None:
        with self.assertRaises(ValueError):
            make_scene(make_object(visible_end=8))


class SamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SamplerConfig(min_objects=2, max_objects=3)
        self.sampler = SceneSampler(base_seed=2026, config=self.config)

    def test_data_001_same_seed_and_index_are_identical(self) -> None:
        self.assertEqual(self.sampler.sample(42), self.sampler.sample(42))

    def test_data_002_random_access_order_is_stable(self) -> None:
        forward = {i: self.sampler.sample(i) for i in (2, 9, 1, 7)}
        reverse = {i: self.sampler.sample(i) for i in (7, 1, 9, 2)}
        self.assertEqual(forward, reverse)

    def test_data_003_different_indices_are_not_all_identical(self) -> None:
        scenes = [self.sampler.sample(i) for i in range(10)]
        serialized = {str(scene_to_dict(scene)) for scene in scenes}
        self.assertGreater(len(serialized), 1)

    def test_data_004_minimum_center_distance(self) -> None:
        for index in range(50):
            scene = self.sampler.sample(index)
            positions = [obj.initial_position for obj in scene.objects]
            for i, left in enumerate(positions):
                for right in positions[i + 1 :]:
                    actual = math.dist((left.x, left.y), (right.x, right.y))
                    self.assertGreaterEqual(actual, self.config.min_center_distance)

    def test_data_005_batch_validity_and_vocabulary_coverage(self) -> None:
        shapes: set[Shape] = set()
        colors: set[Color] = set()
        motions: set[MotionDirection] = set()
        for index in range(200):
            scene = self.sampler.sample(index)
            self.assertGreaterEqual(len(scene.objects), self.config.min_objects)
            self.assertLessEqual(len(scene.objects), self.config.max_objects)
            for obj in scene.objects:
                shapes.add(obj.shape)
                colors.add(obj.color)
                motions.add(obj.motion.direction)
                obj.position_at(scene.frame_count - 1)
        self.assertEqual(shapes, set(Shape))
        self.assertEqual(colors, set(Color))
        self.assertEqual(motions, set(MotionDirection))


class SerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = SceneSampler(base_seed=9, config=SamplerConfig(min_objects=2)).sample(3)

    def test_int_001_dictionary_round_trip(self) -> None:
        self.assertEqual(scene_from_dict(scene_to_dict(self.scene)), self.scene)

    def test_int_002_json_file_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "scene.json"
            save_scene_json(self.scene, path)
            self.assertEqual(load_scene_json(path), self.scene)

    def test_int_003_rejects_unknown_enum_value(self) -> None:
        data = scene_to_dict(self.scene)
        data["objects"][0]["shape"] = "hexagon"
        with self.assertRaises(ValueError):
            scene_from_dict(data)


class ArtifactTests(unittest.TestCase):
    def test_int_004_fixture_export_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = export_scene_fixture(tmp_dir, count=10, base_seed=88)
            verification = verify_scene_fixture(tmp_dir)
            scenes = load_scene_fixture(Path(tmp_dir) / "scenes.jsonl")
            self.assertEqual(manifest["count"], 10)
            self.assertTrue(verification["verified"])
            self.assertEqual(len(scenes), 10)

    def test_int_005_fixture_verification_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_scene_fixture(tmp_dir, count=4, base_seed=99)
            scenes_path = Path(tmp_dir) / "scenes.jsonl"
            scenes_path.write_text(
                scenes_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_scene_fixture(tmp_dir)


if __name__ == "__main__":
    unittest.main()
