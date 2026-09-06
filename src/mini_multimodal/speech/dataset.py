"""One speech specification per P1-W03 scene."""
from __future__ import annotations
from dataclasses import dataclass
from mini_multimodal.data.schema import DatasetBundle
from mini_multimodal.tasks.text_zh import ChineseTextAdapter
from mini_multimodal.vision.dataset import iter_image_specs

@dataclass(frozen=True,slots=True)
class SpeechSpec:
    pool:str; split:str; variant:str; scene_id:str; text:str; cue_token:str|None

def iter_speech_specs(bundle:DatasetBundle)->tuple[SpeechSpec,...]:
    adapter=ChineseTextAdapter()
    return tuple(SpeechSpec(s.pool,s.split.value,s.variant.value,s.scene.scene_id,adapter.describe_scene(s.scene,0),s.cue_token) for s in iter_image_specs(bundle))
