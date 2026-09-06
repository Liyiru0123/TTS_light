"""Lossless compressed frame sequences, temporal annotations, and previews."""
from __future__ import annotations
import gzip,hashlib,json
from collections import Counter
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from mini_multimodal.data.schema import DatasetBundle
from mini_multimodal.tasks.examples import make_temporal_example_scene
from mini_multimodal.vision.dataset import iter_image_specs
from .diagnostics import validate_video_bundle
from .renderer import render_video
from .serialization import annotation_from_dict,annotation_to_dict

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _write_json(path,value):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
def _write_jsonl(path,values):Path(path).write_text("".join(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n" for v in values),encoding="utf-8",newline="\n")
def save_frames(path,frames):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,compresslevel=9,mtime=0) as stream:np.save(stream,frames,allow_pickle=False)
def load_frames(path):
    with gzip.open(path,"rb") as stream:return np.load(stream,allow_pickle=False)
def _relative(spec):return Path("frames")/spec.pool/spec.split.value/f"{spec.scene.scene_id}.npy.gz"

def export_video_artifacts(output_dir,bundle:DatasetBundle):
    out=Path(output_dir);diagnostics=validate_video_bundle(bundle);records=[];files={};preview=None
    for spec in iter_image_specs(bundle):
        video=render_video(spec.scene,cue_token=spec.cue_token);rel=_relative(spec);save_frames(out/rel,video.frames);files[rel.as_posix()]={"bytes":(out/rel).stat().st_size,"sha256":_sha(out/rel)};records.append({"pool":spec.pool,"split":spec.split.value,"variant":spec.variant.value,"frames_path":rel.as_posix(),"frames_sha256":files[rel.as_posix()]["sha256"],"annotation":annotation_to_dict(video.annotation)})
        if preview is None and any(t.motion_direction!="static" for t in video.annotation.tracks):preview=video
    ann=out/"annotations.jsonl";_write_jsonl(ann,records);files["annotations.jsonl"]={"bytes":ann.stat().st_size,"sha256":_sha(ann)}
    temporal=render_video(make_temporal_example_scene());temporal_frames=out/"temporal_boundary_frames.npy.gz";save_frames(temporal_frames,temporal.frames);temporal_annotation=out/"temporal_boundary_annotation.json";_write_json(temporal_annotation,annotation_to_dict(temporal.annotation))
    for path in (temporal_frames,temporal_annotation):files[path.name]={"bytes":path.stat().st_size,"sha256":_sha(path)}
    preview=temporal
    strip=np.concatenate(list(preview.frames),axis=1);strip_path=out/"trajectory_strip.png";Image.fromarray(strip).save(strip_path);files["trajectory_strip.png"]={"bytes":strip_path.stat().st_size,"sha256":_sha(strip_path)}
    mp4=out/"preview.mp4";imageio.mimwrite(mp4,list(preview.frames),fps=preview.annotation.fps,codec="libx264",quality=8,macro_block_size=None);files["preview.mp4"]={"bytes":mp4.stat().st_size,"sha256":_sha(mp4)}
    manifest={"artifact_version":1,"storage":"gzip-compressed deterministic NPY uint8 frames","video_count":len(records),"supplemental_temporal_fixture_count":1,"temporal_fixture_event_counts":dict(Counter(e.event_type for e in temporal.annotation.events)),"files":dict(sorted(files.items())),**diagnostics};_write_json(out/"manifest.json",manifest);return manifest

def verify_video_artifacts(output_dir,bundle:DatasetBundle):
    out=Path(output_dir);m=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    for name,meta in m["files"].items():
        if _sha(out/name)!=meta["sha256"]:raise ValueError(f"Video artifact {name} SHA-256 does not match")
    rows=[json.loads(x) for x in (out/"annotations.jsonl").read_text(encoding="utf-8").splitlines()];specs=iter_image_specs(bundle)
    if len(rows)!=len(specs):raise ValueError("Video annotation count mismatch")
    for row,spec in zip(rows,specs,strict=True):
        ann=annotation_from_dict(row["annotation"]);saved=load_frames(out/row["frames_path"]);expected=render_video(spec.scene,cue_token=spec.cue_token)
        if ann!=expected.annotation or not np.array_equal(saved,expected.frames):raise ValueError(f"Video re-rendering mismatch: {spec.scene.scene_id}")
    temporal=render_video(make_temporal_example_scene());saved_temporal=load_frames(out/"temporal_boundary_frames.npy.gz");loaded_temporal=annotation_from_dict(json.loads((out/"temporal_boundary_annotation.json").read_text(encoding="utf-8")))
    if loaded_temporal!=temporal.annotation or not np.array_equal(saved_temporal,temporal.frames):raise ValueError("Supplemental temporal boundary fixture mismatch")
    diagnostics=validate_video_bundle(bundle)
    for k in ("statistics","consistency","shortcut"):
        if diagnostics[k]!=m[k]:raise ValueError(f"Video {k} differs from manifest")
    return {"verified":True,"video_count":len(rows),"supplemental_temporal_fixture_count":1,"temporal_fixture_event_counts":m["temporal_fixture_event_counts"],"managed_file_count":len(m["files"]),**diagnostics}
