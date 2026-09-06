from __future__ import annotations
from dataclasses import asdict
from .schema import ObjectTrack,TrackPoint,VideoAnnotation,VideoEvent

def annotation_to_dict(value:VideoAnnotation)->dict:return asdict(value)
def annotation_from_dict(d:dict)->VideoAnnotation:
    tracks=[]
    for t in d["tracks"]:
        points=tuple(TrackPoint(frame_index=int(p["frame_index"]),visible=bool(p["visible"]),center_normalized=tuple(p["center_normalized"]),affine_center_xy=tuple(p["affine_center_xy"]),center_xy=tuple(p["center_xy"]),bbox_xywh=tuple(p["bbox_xywh"]) if p["bbox_xywh"] is not None else None,visible_bbox_xywh=tuple(p["visible_bbox_xywh"]) if p["visible_bbox_xywh"] is not None else None) for p in t["points"])
        tracks.append(ObjectTrack(object_id=t["object_id"],shape=t["shape"],color=t["color"],size=t["size"],motion_direction=t["motion_direction"],speed=float(t["speed"]),visible_start=int(t["visible_start"]),visible_end=int(t["visible_end"]),points=points))
    return VideoAnnotation(video_id=d["video_id"],scene_id=d["scene_id"],frame_count=int(d["frame_count"]),fps=int(d["fps"]),width=int(d["width"]),height=int(d["height"]),cue_token=d["cue_token"],cue_enabled=bool(d["cue_enabled"]),tracks=tuple(tracks),events=tuple(VideoEvent(e["object_id"],e["event_type"],int(e["frame_index"])) for e in d["events"]),schema_version=int(d.get("schema_version",1)))
