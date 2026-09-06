from __future__ import annotations
from dataclasses import asdict
from .schema import SpeechAnnotation,SpeechToken

def annotation_to_dict(value:SpeechAnnotation)->dict:
    result=asdict(value); result["logmel_shape"]=list(value.logmel_shape); return result

def annotation_from_dict(d:dict)->SpeechAnnotation:
    tokens=tuple(SpeechToken(**item) for item in d["tokens"])
    return SpeechAnnotation(utterance_id=d["utterance_id"],scene_id=d["scene_id"],text=d["text"],voice_name=d["voice_name"],sample_rate=int(d["sample_rate"]),num_samples=int(d["num_samples"]),duration_seconds=float(d["duration_seconds"]),cue_token=d["cue_token"],cue_enabled=bool(d["cue_enabled"]),cue_samples=int(d["cue_samples"]),alignment_time_scale=float(d["alignment_time_scale"]),logmel_shape=tuple(d["logmel_shape"]),tokens=tokens,schema_version=int(d.get("schema_version",1)))
