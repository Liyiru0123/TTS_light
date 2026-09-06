"""Schemas for temporal tracks, events, and rendered frame sequences."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from mini_multimodal.vision.schema import ImageAnnotation,ImageRenderConfig

@dataclass(frozen=True,slots=True)
class VideoRenderConfig:
    image_config:ImageRenderConfig=ImageRenderConfig(); schema_version:int=1
    def __post_init__(self):
        if self.schema_version!=1:raise ValueError("Unsupported video render config")

@dataclass(frozen=True,slots=True)
class TrackPoint:
    frame_index:int; visible:bool; center_normalized:tuple[float,float]; affine_center_xy:tuple[int,int]; center_xy:tuple[int,int]; bbox_xywh:tuple[int,int,int,int]|None; visible_bbox_xywh:tuple[int,int,int,int]|None

@dataclass(frozen=True,slots=True)
class VideoEvent:
    object_id:str; event_type:str; frame_index:int

@dataclass(frozen=True,slots=True)
class ObjectTrack:
    object_id:str; shape:str; color:str; size:str; motion_direction:str; speed:float; visible_start:int; visible_end:int; points:tuple[TrackPoint,...]

@dataclass(frozen=True,slots=True)
class VideoAnnotation:
    video_id:str; scene_id:str; frame_count:int; fps:int; width:int; height:int; cue_token:str|None; cue_enabled:bool; tracks:tuple[ObjectTrack,...]; events:tuple[VideoEvent,...]; schema_version:int=1
    def __post_init__(self):
        if self.schema_version!=1 or self.frame_count<=0 or self.fps<=0 or len({t.object_id for t in self.tracks})!=len(self.tracks):raise ValueError("Invalid video annotation")
        if any(len(t.points)!=self.frame_count for t in self.tracks):raise ValueError("Every object track must span the full frame timeline")

@dataclass(slots=True)
class RenderedVideo:
    frames:np.ndarray; frame_annotations:tuple[ImageAnnotation,...]; annotation:VideoAnnotation
