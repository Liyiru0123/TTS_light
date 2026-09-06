"""Post-process cached Huihui speech into aligned model-ready examples."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from .features import log_mel
from .g2p import character_label,validate_text
from .schema import CUE_FREQUENCIES,RenderedSpeech,SpeechAnnotation,SpeechRenderConfig,SpeechToken

def _cue_prefix(token:str,sr:int,cfg:SpeechRenderConfig)->np.ndarray:
    n=int(round(cfg.cue_duration*sr)); silence=int(round(cfg.cue_silence*sr)); t=np.arange(n)/sr
    tone=cfg.cue_amplitude*np.sin(2*np.pi*CUE_FREQUENCIES[token]*t)
    fade=min(int(.005*sr),n//2); ramp=np.linspace(0,1,fade,endpoint=False); tone[:fade]*=ramp; tone[-fade:]*=ramp[::-1]
    return np.concatenate([tone,np.zeros(silence)]).astype(np.float32)

def render_cached_speech(*,utterance_id:str,scene_id:str,text:str,wav_path:str|Path,alignment_path:str|Path,cue_token:str|None=None,config:SpeechRenderConfig|None=None)->RenderedSpeech:
    cfg=config or SpeechRenderConfig(); validate_text(text)
    if cue_token is not None and cue_token not in CUE_FREQUENCIES: raise ValueError(f"Unknown speech cue {cue_token}")
    pcm,sr=sf.read(wav_path,dtype="int16",always_2d=False)
    if sr!=cfg.sample_rate or pcm.ndim!=1 or len(pcm)==0: raise ValueError("Unexpected Huihui waveform format")
    raw=pcm.astype(np.float32)/32768.; payload=json.loads(Path(alignment_path).read_text(encoding="utf-8"))
    if payload["text"]!=text or payload["voice_name"]!=cfg.voice_name: raise ValueError("Cached speech metadata mismatch")
    marks=payload["marks"] if isinstance(payload["marks"],list) else [payload["marks"]]
    if len(marks)!=len(text)+1 or [m["name"] for m in marks] != [f"c{i:04d}" for i in range(len(text)+1)]: raise ValueError("SSML mark sequence is incomplete")
    engine_end=float(marks[-1]["milliseconds"])*sr/1000
    if engine_end<=0: raise ValueError("SSML engine time axis has no duration")
    alignment_scale=len(raw)/engine_end
    boundaries=[min(len(raw),max(0,int(round(float(m["milliseconds"])*sr/1000*alignment_scale)))) for m in marks]
    if boundaries!=sorted(boundaries): raise ValueError("SSML mark times are not monotonic")
    cue_enabled=cue_token is not None and cfg.enable_cue; prefix=_cue_prefix(cue_token,sr,cfg) if cue_enabled else np.zeros(0,dtype=np.float32); shift=len(prefix)
    wave=np.concatenate([prefix,raw]); mel=log_mel(wave,sr,n_fft=cfg.n_fft,hop_length=cfg.hop_length,n_mels=cfg.n_mels,fmin=cfg.fmin,fmax=cfg.fmax)
    tokens=[]
    for i,char in enumerate(text):
        label=character_label(char); start=boundaries[i]+shift; end=boundaries[i+1]+shift
        if not label["is_punctuation"] and end<=start: raise ValueError("Non-punctuation character has zero duration")
        tokens.append(SpeechToken(index=i,char=char,**label,start_sample=start,end_sample=end,start_frame=min(mel.shape[1],start//cfg.hop_length),end_frame=min(mel.shape[1],int(np.ceil(end/cfg.hop_length)))))
    ann=SpeechAnnotation(utterance_id,scene_id,text,cfg.voice_name,sr,len(wave),len(wave)/sr,cue_token,cue_enabled,shift,alignment_scale,tuple(mel.shape),tuple(tokens))
    return RenderedSpeech(wave,mel,ann)
