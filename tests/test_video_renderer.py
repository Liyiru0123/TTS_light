from __future__ import annotations
import hashlib,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from mini_multimodal.data.artifacts import load_dataset_bundle
from mini_multimodal.video import render_video
from mini_multimodal.video.artifacts import load_frames,save_frames,verify_video_artifacts
from mini_multimodal.video.diagnostics import validate_video_bundle
from mini_multimodal.video.serialization import annotation_from_dict,annotation_to_dict
from mini_multimodal.vision.schema import ImageRenderConfig
from mini_multimodal.video.schema import VideoRenderConfig
from mini_multimodal.world import Color,MotionDirection,MotionSpec,Position,Scene,SceneObject,Shape,Size
BUNDLE=load_dataset_bundle(ROOT/"data"/"datasets"/"p1_w03_v1");VIDEO_DATA=ROOT/"data"/"video"/"p1_w06_v1";REPORT=None
def report():
 global REPORT
 if REPORT is None:REPORT=validate_video_bundle(BUNDLE)
 return REPORT
def boundary_scene():
 return Scene("video_boundary",1,32,32,8,8,(SceneObject("obj_0",Shape.CIRCLE,Color.RED,Size.SMALL,Position(.2,.5),MotionSpec(MotionDirection.RIGHT,.05,2,6),2,5),))

class VideoRendererTests(unittest.TestCase):
 def test_video_001_deterministic(self):
  a=render_video(BUNDLE.core_scenes[0]);b=render_video(BUNDLE.core_scenes[0]);self.assertTrue(np.array_equal(a.frames,b.frames));self.assertEqual(a.annotation,b.annotation)
 def test_video_002_frame_specification(self):
  v=render_video(BUNDLE.core_scenes[0]);self.assertEqual(v.frames.shape,(8,64,64,3));self.assertEqual(v.frames.dtype,np.uint8);self.assertEqual(v.annotation.fps,8)
 def test_track_001_no_static_coordinate_jitter(self):self.assertEqual(report()["consistency"]["static_coordinate_jitter_count"],0)
 def test_track_002_no_motion_reverse_step(self):self.assertEqual(report()["consistency"]["motion_reverse_step_count"],0)
 def test_track_003_spatial_order_is_preserved(self):self.assertEqual(report()["consistency"]["spatial_order_violation_count"],0)
 def test_track_004_projection_adjustment_is_bounded(self):
  self.assertGreater(report()["consistency"]["adjusted_frame_object_count"],0);self.assertLessEqual(report()["consistency"]["max_projection_adjustment_pixels"],1)
 def test_vis_001_visibility_controls_boxes(self):
  t=render_video(boundary_scene()).annotation.tracks[0]
  for p in t.points:self.assertEqual(p.visible,2<=p.frame_index<=5);self.assertEqual(p.bbox_xywh is not None,p.visible)
 def test_event_001_visibility_events_match_boundary(self):
  e={(x.event_type,x.frame_index) for x in render_video(boundary_scene()).annotation.events};self.assertIn(("appear",2),e);self.assertIn(("disappear",6),e)
 def test_event_002_motion_events_match_spec(self):
  e={(x.event_type,x.frame_index) for x in render_video(boundary_scene()).annotation.events};self.assertIn(("move_start",2),e);self.assertIn(("move_end",6),e)
 def test_cue_001_background_is_stable_across_video(self):
  v=render_video(BUNDLE.shortcut_scenes[0],cue_token="cue_4");background=v.frames[:,0,0];self.assertTrue(np.all(background==background[0]));self.assertEqual(tuple(background[0]),(180,180,180))
 def test_cue_002_disabling_cue_preserves_tracks(self):
  scene=BUNDLE.shortcut_scenes[0];on=render_video(scene,cue_token="cue_4");off=render_video(scene,cue_token="cue_4",config=VideoRenderConfig(ImageRenderConfig(enable_cue=False)));self.assertEqual(on.annotation.tracks,off.annotation.tracks)
 def test_int_001_annotation_round_trip(self):
  a=render_video(BUNDLE.core_scenes[0]).annotation;self.assertEqual(annotation_from_dict(annotation_to_dict(a)),a)
 def test_int_002_lossless_frames_are_deterministic(self):
  frames=render_video(BUNDLE.core_scenes[0]).frames
  with tempfile.TemporaryDirectory() as tmp:
   a=Path(tmp)/"a.npy.gz";b=Path(tmp)/"b.npy.gz";save_frames(a,frames);save_frames(b,frames);self.assertTrue(np.array_equal(load_frames(a),frames));self.assertEqual(hashlib.sha256(a.read_bytes()).digest(),hashlib.sha256(b.read_bytes()).digest())

class VideoArtifactTests(unittest.TestCase):
 def test_int_003_full_export_verifies(self):
  v=verify_video_artifacts(VIDEO_DATA,BUNDLE);self.assertTrue(v["verified"]);self.assertEqual((v["video_count"],v["statistics"]["frame_count"]),(1720,13760))
 def test_int_004_tampering_is_detected(self):
  p=next((VIDEO_DATA/"frames").rglob("*.npy.gz"));original=p.read_bytes()
  try:
   p.write_bytes(original+b"x")
   with self.assertRaisesRegex(ValueError,"SHA-256"):verify_video_artifacts(VIDEO_DATA,BUNDLE)
  finally:p.write_bytes(original)
 def test_event_003_persistent_temporal_fixture_covers_disappearance(self):
  import json
  data=json.loads((VIDEO_DATA/"temporal_boundary_annotation.json").read_text(encoding="utf-8"));events={(e["event_type"],e["frame_index"]) for e in data["events"]};self.assertIn(("disappear",6),events);self.assertIn(("appear",1),events)
if __name__=="__main__":unittest.main()
