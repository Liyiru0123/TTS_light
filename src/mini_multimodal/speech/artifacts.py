"""SAPI cache orchestration and persistent speech artifacts."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from PIL import Image
from mini_multimodal.data.schema import DatasetBundle
from .dataset import iter_speech_specs
from .renderer import render_cached_speech
from .schema import CUE_FREQUENCIES,SpeechRenderConfig
from .serialization import annotation_from_dict,annotation_to_dict

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _key(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
def _write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
def _write_jsonl(path,values): Path(path).write_text("".join(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n" for v in values),encoding="utf-8",newline="\n")

def ensure_huihui_cache(texts,cache_dir,bridge_path):
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True); jobs=[]
    for text in sorted(set(texts)):
        key=_key(text); wav=cache/f"{key}.wav"; align=cache/f"{key}.json"
        if not wav.exists() or not align.exists(): jobs.append({"utterance_key":key,"text":text,"wav_path":str(wav.resolve()),"alignment_path":str(align.resolve())})
    if jobs:
        jobs_path=cache/"pending_jobs.json";_write_json(jobs_path,jobs)
        subprocess.run(["powershell.exe","-NoProfile","-Sta","-ExecutionPolicy","Bypass","-File",str(Path(bridge_path).resolve()),"-JobsPath",str(jobs_path.resolve())],check=True)
    return {t:(cache/f"{_key(t)}.wav",cache/f"{_key(t)}.json") for t in set(texts)}

def _relative(spec,kind,ext): return Path(kind)/spec.pool/spec.split/f"{spec.scene_id}_f000.{ext}"

def export_speech_artifacts(output_dir,bundle:DatasetBundle,cache_map,config=None):
    out=Path(output_dir);cfg=config or SpeechRenderConfig();records=[];files={};durations=[];token_count=0;cue_counts={k:0 for k in CUE_FREQUENCIES};preview=None
    for spec in iter_speech_specs(bundle):
        wav,align=cache_map[spec.text]; rendered=render_cached_speech(utterance_id=f"{spec.scene_id}:speech:f0",scene_id=spec.scene_id,text=spec.text,wav_path=wav,alignment_path=align,cue_token=spec.cue_token,config=cfg)
        audio_rel=_relative(spec,"audio","flac");mel_rel=_relative(spec,"logmel","npy");(out/audio_rel).parent.mkdir(parents=True,exist_ok=True);(out/mel_rel).parent.mkdir(parents=True,exist_ok=True)
        sf.write(out/audio_rel,rendered.waveform,cfg.sample_rate,format="FLAC",subtype="PCM_16");np.save(out/mel_rel,rendered.logmel.astype(np.float16),allow_pickle=False)
        for rel in (audio_rel,mel_rel): files[rel.as_posix()]={"bytes":(out/rel).stat().st_size,"sha256":_sha(out/rel)}
        records.append({"pool":spec.pool,"split":spec.split,"variant":spec.variant,"audio_path":audio_rel.as_posix(),"logmel_path":mel_rel.as_posix(),"audio_sha256":files[audio_rel.as_posix()]["sha256"],"logmel_sha256":files[mel_rel.as_posix()]["sha256"],"annotation":annotation_to_dict(rendered.annotation)})
        durations.append(rendered.annotation.duration_seconds);token_count+=len(rendered.annotation.tokens)
        if spec.cue_token: cue_counts[spec.cue_token]+=1
        if preview is None: preview=rendered.logmel
    ann=out/"annotations.jsonl";_write_jsonl(ann,records);files["annotations.jsonl"]={"bytes":ann.stat().st_size,"sha256":_sha(ann)}
    img=((preview-preview.min())/(preview.max()-preview.min()+1e-9)*255).astype(np.uint8);img=np.flipud(img);preview_path=out/"spectrogram_preview.png";Image.fromarray(img).resize((img.shape[1]*2,img.shape[0]*3)).save(preview_path);files["spectrogram_preview.png"]={"bytes":preview_path.stat().st_size,"sha256":_sha(preview_path)}
    stats={"utterance_count":len(records),"token_count":token_count,"total_duration_seconds":sum(durations),"min_duration_seconds":min(durations),"max_duration_seconds":max(durations),"mean_duration_seconds":sum(durations)/len(durations),"sample_rate":cfg.sample_rate,"n_mels":cfg.n_mels}
    shortcut={"cue_frequencies":CUE_FREQUENCIES,"cue_utterance_counts":cue_counts,"shortcut_utterance_count":sum(cue_counts.values())}
    manifest={"artifact_version":1,"voice_name":cfg.voice_name,"statistics":stats,"shortcut":shortcut,"files":dict(sorted(files.items()))};_write_json(out/"manifest.json",manifest);return manifest

def verify_speech_artifacts(output_dir):
    out=Path(output_dir);m=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    for name,meta in m["files"].items():
        if _sha(out/name)!=meta["sha256"]: raise ValueError(f"Speech artifact {name} SHA-256 does not match")
    rows=[json.loads(x) for x in (out/"annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    errors=0
    for row in rows:
        ann=annotation_from_dict(row["annotation"]);audio,sr=sf.read(out/row["audio_path"],dtype="float32");mel=np.load(out/row["logmel_path"],allow_pickle=False)
        errors += sr!=ann.sample_rate or len(audio)!=ann.num_samples or tuple(mel.shape)!=ann.logmel_shape or not np.isfinite(mel).all() or any(t.end_sample>ann.num_samples for t in ann.tokens)
    if errors or len(rows)!=m["statistics"]["utterance_count"]: raise ValueError("Speech annotations or arrays are inconsistent")
    return {"verified":True,"utterance_count":len(rows),"managed_file_count":len(m["files"]),"consistency_error_count":errors,**{k:m[k] for k in ("statistics","shortcut")}}
