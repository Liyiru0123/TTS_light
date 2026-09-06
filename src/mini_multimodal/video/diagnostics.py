"""Full-dataset temporal consistency diagnostics."""
from __future__ import annotations
from collections import Counter
from math import isclose
from mini_multimodal.data.schema import DatasetBundle
from mini_multimodal.tasks.reasoning import derive_events
from mini_multimodal.vision.dataset import iter_image_specs
from .renderer import render_video

def analyze_video_bundle(bundle:DatasetBundle)->dict[str,object]:
    stats=Counter();events=Counter();cue=Counter();static_jitter=reverse=spatial=visibility=event_errors=maxdev=adjusted=0
    for spec in iter_image_specs(bundle):
        v=render_video(spec.scene,cue_token=spec.cue_token);stats["video_count"]+=1;stats["frame_count"]+=len(v.frames);stats["track_count"]+=len(v.annotation.tracks);cue[spec.cue_token or "none"]+=1
        expected={(e.object_id,e.event_type.value,e.frame) for e in derive_events(spec.scene)};actual={(e.object_id,e.event_type,e.frame_index) for e in v.annotation.events};event_errors+=expected!=actual
        for e in v.annotation.events:events[e.event_type]+=1
        for i,t in enumerate(v.annotation.tracks):
            stats["moving_track_count" if t.motion_direction!="static" else "static_track_count"]+=1
            obj=spec.scene.objects[i]
            for f,p in enumerate(t.points):
                visibility+=p.visible!=obj.is_visible_at(f);maxdev=max(maxdev,abs(p.center_xy[0]-p.affine_center_xy[0]),abs(p.center_xy[1]-p.affine_center_xy[1]));adjusted+=p.center_xy!=p.affine_center_xy
                if f:
                    prev=t.points[f-1]
                    for axis in (0,1):
                        semantic=p.center_normalized[axis]-prev.center_normalized[axis];pixel=p.center_xy[axis]-prev.center_xy[axis]
                        static_jitter+=isclose(semantic,0.,abs_tol=1e-9) and pixel!=0;reverse+=(semantic>0 and pixel<0) or (semantic<0 and pixel>0)
            for f in range(spec.scene.frame_count):
                for j in range(i+1,len(v.annotation.tracks)):
                    for axis in (0,1):
                        a=t.points[f];b=v.annotation.tracks[j].points[f];sem=a.center_normalized[axis]-b.center_normalized[axis];pix=a.center_xy[axis]-b.center_xy[axis];ss=0 if isclose(sem,0.,abs_tol=1e-9) else(-1 if sem<0 else 1);ps=0 if pix==0 else(-1 if pix<0 else 1);spatial+=ss!=ps
    return {"statistics":{**dict(stats),"event_counts":dict(sorted(events.items()))},"consistency":{"static_coordinate_jitter_count":static_jitter,"motion_reverse_step_count":reverse,"spatial_order_violation_count":spatial,"visibility_mismatch_count":visibility,"event_mismatch_scene_count":event_errors,"adjusted_frame_object_count":adjusted,"max_projection_adjustment_pixels":maxdev},"shortcut":{"cue_video_counts":dict(sorted(cue.items())),"shortcut_video_count":sum(v for k,v in cue.items() if k!="none")}}

def validate_video_bundle(bundle):
    report=analyze_video_bundle(bundle);c=report["consistency"]
    for k in ("static_coordinate_jitter_count","motion_reverse_step_count","spatial_order_violation_count","visibility_mismatch_count","event_mismatch_scene_count"):
        if c[k]:raise ValueError(f"Blocking video consistency failure: {k}")
    if c["max_projection_adjustment_pixels"]>1:raise ValueError("Video projection adjustment exceeds one pixel")
    if report["statistics"]["video_count"]!=len(bundle.all_scenes):raise ValueError("Video count mismatch")
    return report
