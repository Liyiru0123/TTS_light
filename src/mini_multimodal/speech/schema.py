"""Schemas for rendered speech, linguistic tokens, and alignments."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

CUE_FREQUENCIES={"cue_0":300.0,"cue_1":450.0,"cue_2":600.0,"cue_3":750.0,"cue_4":900.0}

@dataclass(frozen=True,slots=True)
class SpeechRenderConfig:
    voice_name:str="Microsoft Huihui Desktop"; sample_rate:int=22050; n_fft:int=512; hop_length:int=256; n_mels:int=64; fmin:float=40.; fmax:float=8000.; enable_cue:bool=True; cue_duration:float=.12; cue_silence:float=.04; cue_amplitude:float=.08; schema_version:int=1
    def __post_init__(self):
        if self.schema_version!=1 or self.sample_rate<=0 or self.n_fft<=0 or self.hop_length<=0 or self.n_mels<=0: raise ValueError("Invalid speech render configuration")
        if not 0<self.cue_amplitude<1 or self.cue_duration<=0 or self.cue_silence<0: raise ValueError("Invalid speech cue configuration")

@dataclass(frozen=True,slots=True)
class SpeechToken:
    index:int; char:str; normalized_char:str; pinyin:str; initial:str; final:str; tone:int; is_punctuation:bool; start_sample:int; end_sample:int; start_frame:int; end_frame:int
    def __post_init__(self):
        if self.index<0 or self.start_sample<0 or self.end_sample<self.start_sample or self.start_frame<0 or self.end_frame<self.start_frame: raise ValueError("Invalid speech token alignment")

@dataclass(frozen=True,slots=True)
class SpeechAnnotation:
    utterance_id:str; scene_id:str; text:str; voice_name:str; sample_rate:int; num_samples:int; duration_seconds:float; cue_token:str|None; cue_enabled:bool; cue_samples:int; alignment_time_scale:float; logmel_shape:tuple[int,int]; tokens:tuple[SpeechToken,...]; schema_version:int=1
    def __post_init__(self):
        if self.schema_version!=1 or not self.utterance_id or not self.text or self.sample_rate<=0 or self.num_samples<=0 or self.alignment_time_scale<=0: raise ValueError("Invalid speech annotation")
        if len(self.tokens)!=len(self.text): raise ValueError("Speech tokens must align one-to-one with text")

@dataclass(slots=True)
class RenderedSpeech:
    waveform:np.ndarray; logmel:np.ndarray; annotation:SpeechAnnotation
