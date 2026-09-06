from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mini_multimodal.data.artifacts import load_dataset_bundle  # noqa: E402
from mini_multimodal.data.schema import DatasetVariant  # noqa: E402
from mini_multimodal.vision import (  # noqa: E402
    COLOR_RGB,
    CUE_BACKGROUNDS,
    NEUTRAL_BACKGROUND,
    ImageRenderConfig,
    decode_rle,
    mask_bbox,
    render_scene,
    validate_image_bundle,
)
from mini_multimodal.vision.artifacts import (  # noqa: E402
    export_image_artifacts,
    verify_image_artifacts,
)
from mini_multimodal.vision.dataset import iter_image_specs  # noqa: E402
from mini_multimodal.vision.serialization import (  # noqa: E402
    image_annotation_from_dict,
    image_annotation_to_dict,
)
from mini_multimodal.world import (  # noqa: E402
    Color,
    MotionDirection,
    MotionSpec,
    Position,
    Scene,
    SceneObject,
    Shape,
    Size,
)


_BUNDLE = None


def full_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_dataset_bundle(PROJECT_ROOT / "data" / "datasets" / "p1_w03_v1")
    return _BUNDLE


def one_object_scene(
    *,
    shape: Shape = Shape.CIRCLE,
    color: Color = Color.RED,
    size: Size = Size.MEDIUM,
    position: Position = Position(0.5, 0.5),
    motion: MotionSpec = MotionSpec(),
    visible_start: int = 0,
    visible_end: int = 7,
) -> Scene:
    return Scene(
        scene_id="custom_scene",
        seed=1,
        canvas_width=32,
        canvas_height=32,
        frame_count=8,
        fps=8,
        objects=(
            SceneObject(
                object_id="obj_0",
                shape=shape,
                color=color,
                size=size,
                initial_position=position,
                motion=motion,
                visible_start=visible_start,
                visible_end=visible_end,
            ),
        ),
    )


class ImageRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = full_bundle()
        cls.report = validate_image_bundle(cls.bundle)

    def test_image_001_same_input_is_deterministic(self) -> None:
        scene = self.bundle.core_scenes[0]
        first = render_scene(scene)
        second = render_scene(scene)
        self.assertEqual(first.image.tobytes(), second.image.tobytes())
        self.assertEqual(first.annotation, second.annotation)

    def test_image_002_invalid_frame_config_and_cue_are_rejected(self) -> None:
        scene = self.bundle.core_scenes[0]
        cases = (
            lambda: render_scene(scene, frame_index=scene.frame_count),
            lambda: ImageRenderConfig(output_scale=0),
            lambda: render_scene(scene, cue_token="unknown_cue"),
        )
        for call in cases:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_image_003_all_shape_masks_are_distinct(self) -> None:
        signatures = set()
        for shape in Shape:
            rendered = render_scene(one_object_scene(shape=shape))
            signatures.add(rendered.amodal_masks["obj_0"].tobytes())
        self.assertEqual(len(signatures), len(Shape))

    def test_image_004_all_colors_use_the_fixed_palette(self) -> None:
        for color in Color:
            with self.subTest(color=color):
                rendered = render_scene(one_object_scene(color=color))
                pixels = np.asarray(rendered.image)
                mask = rendered.visible_masks["obj_0"]
                self.assertTrue(np.all(pixels[mask] == COLOR_RGB[color]))

    def test_image_005_size_areas_strictly_increase(self) -> None:
        areas = []
        for size in (Size.SMALL, Size.MEDIUM, Size.LARGE):
            annotation = render_scene(one_object_scene(size=size)).annotation.objects[0]
            areas.append(annotation.area_pixels)
        self.assertLess(areas[0], areas[1])
        self.assertLess(areas[1], areas[2])

    def test_image_006_all_fixed_objects_stay_inside_canvas(self) -> None:
        self.assertEqual(self.report["consistency"]["out_of_bounds_count"], 0)

    def test_image_007_pixel_order_matches_all_relation_answers(self) -> None:
        consistency = self.report["consistency"]
        self.assertEqual(consistency["relation_record_count"], 2200)
        self.assertEqual(consistency["relation_order_violation_count"], 0)

    def test_image_008_order_adjustment_is_bounded(self) -> None:
        consistency = self.report["consistency"]
        self.assertGreater(consistency["adjusted_object_count"], 0)
        self.assertLessEqual(consistency["max_quantization_adjustment_pixels"], 1)


class MaskAndFrameTests(unittest.TestCase):
    def test_mask_001_rle_round_trip(self) -> None:
        rendered = render_scene(one_object_scene(shape=Shape.PENTAGON))
        annotation = rendered.annotation.objects[0]
        decoded = decode_rle(
            annotation.amodal_rle,
            rendered.annotation.height,
            rendered.annotation.width,
        )
        self.assertTrue(np.array_equal(decoded, rendered.amodal_masks["obj_0"]))

    def test_mask_002_area_and_bbox_match_masks(self) -> None:
        rendered = render_scene(one_object_scene(shape=Shape.TRIANGLE))
        annotation = rendered.annotation.objects[0]
        mask = rendered.amodal_masks["obj_0"]
        self.assertEqual(annotation.area_pixels, int(mask.sum()))
        self.assertEqual(annotation.bbox_xywh, mask_bbox(mask))

    def test_mask_003_occlusion_respects_z_order(self) -> None:
        base = one_object_scene().objects[0]
        scene = Scene(
            scene_id="occlusion",
            seed=2,
            canvas_width=32,
            canvas_height=32,
            frame_count=8,
            fps=8,
            objects=(
                base,
                SceneObject(
                    object_id="obj_1",
                    shape=Shape.SQUARE,
                    color=Color.BLUE,
                    size=Size.LARGE,
                    initial_position=base.initial_position,
                    visible_start=0,
                    visible_end=7,
                ),
            ),
        )
        rendered = render_scene(scene)
        lower, upper = rendered.annotation.objects
        self.assertEqual(lower.area_pixels, int(rendered.amodal_masks["obj_0"].sum()))
        self.assertLess(lower.visible_area_pixels, lower.area_pixels)
        self.assertEqual(upper.visible_area_pixels, upper.area_pixels)

    def test_mask_004_fixed_data_has_no_fully_hidden_object(self) -> None:
        report = validate_image_bundle(full_bundle())
        self.assertEqual(report["consistency"]["fully_occluded_object_count"], 0)

    def test_frame_001_visibility_interval_controls_rendering(self) -> None:
        scene = one_object_scene(visible_start=2, visible_end=5)
        self.assertEqual(render_scene(scene, frame_index=1).annotation.objects, ())
        self.assertEqual(len(render_scene(scene, frame_index=2).annotation.objects), 1)
        self.assertEqual(render_scene(scene, frame_index=6).annotation.objects, ())

    def test_frame_002_motion_changes_pixel_center_in_the_right_direction(self) -> None:
        motion = MotionSpec(
            direction=MotionDirection.RIGHT,
            speed=0.05,
            start_frame=0,
            end_frame=7,
        )
        scene = one_object_scene(position=Position(0.2, 0.5), motion=motion)
        start = render_scene(scene, frame_index=0).annotation.objects[0].center_xy
        end = render_scene(scene, frame_index=7).annotation.objects[0].center_xy
        self.assertGreater(end[0], start[0])
        self.assertEqual(end[1], start[1])


class VisualShortcutTests(unittest.TestCase):
    def test_cue_001_each_token_maps_to_its_background(self) -> None:
        scene = one_object_scene()
        for cue, background in CUE_BACKGROUNDS.items():
            with self.subTest(cue=cue):
                rendered = render_scene(scene, cue_token=cue)
                self.assertEqual(rendered.annotation.background_rgb, background)

    def test_cue_002_disabling_cue_changes_only_background(self) -> None:
        scene = one_object_scene()
        enabled = render_scene(scene, cue_token="cue_4")
        disabled = render_scene(
            scene,
            cue_token="cue_4",
            config=ImageRenderConfig(enable_cue=False),
        )
        self.assertEqual(disabled.annotation.background_rgb, NEUTRAL_BACKGROUND)
        self.assertTrue(
            np.array_equal(enabled.amodal_masks["obj_0"], disabled.amodal_masks["obj_0"])
        )
        mask = enabled.amodal_masks["obj_0"]
        self.assertTrue(
            np.array_equal(np.asarray(enabled.image)[mask], np.asarray(disabled.image)[mask])
        )

    def test_cue_003_dataset_variants_obey_cue_boundary(self) -> None:
        cue_count = 0
        for spec in iter_image_specs(full_bundle()):
            if spec.variant in {
                DatasetVariant.SHORTCUT_IID,
                DatasetVariant.COUNTERFACTUAL,
            }:
                self.assertIsNotNone(spec.cue_token)
                cue_count += 1
            else:
                self.assertIsNone(spec.cue_token)
        self.assertEqual(cue_count, 460)
        self.assertEqual(
            validate_image_bundle(full_bundle())["shortcut"]["background_mismatch_count"],
            0,
        )


class ImageArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temp_dir.name)
        cls.manifest = export_image_artifacts(cls.output_dir, full_bundle())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_image_int_001_annotation_json_round_trip(self) -> None:
        annotation = render_scene(full_bundle().core_scenes[0]).annotation
        self.assertEqual(
            image_annotation_from_dict(image_annotation_to_dict(annotation)),
            annotation,
        )

    def test_image_int_002_full_export_and_rerender_verify(self) -> None:
        verification = verify_image_artifacts(self.output_dir, full_bundle())
        self.assertTrue(verification["verified"])
        self.assertEqual(verification["image_count"], 1720)
        self.assertEqual(verification["annotation_count"], 1720)
        self.assertEqual(self.manifest["image_count"], 1720)

    def test_image_int_003_png_tampering_is_detected(self) -> None:
        relative_name = next(
            name for name in self.manifest["files"] if name.startswith("images/")
        )
        path = self.output_dir / relative_name
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_image_artifacts(self.output_dir, full_bundle())
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
