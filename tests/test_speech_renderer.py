from __future__ import annotations
import hashlib,json,sys,unittest
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from mini_multimodal.data.artifacts import load_dataset_bundle
from mini_multimodal.speech.artifacts import verify_speech_artifacts
from mini_multimodal.speech.dataset import iter_speech_specs
from mini_multimodal.speech.features import log_mel
from mini_multimodal.speech.g2p import character_label,validate_text
from mini_multimodal.speech.renderer import _cue_prefix,render_cached_speech
from mini_multimodal.speech.schema import CUE_FREQUENCIES,SpeechRenderConfig
from mini_multimodal.speech.serialization import annotation_from_dict,annotation_to_dict

DATA=ROOT/"data"/"speech"/"p1_w05_v1";CACHE=ROOT/"data"/"cache"/"p1_w05_huihui";BUNDLE=load_dataset_bundle(ROOT/"data"/"datasets"/"p1_w03_v1")
ROWS=None
def rows():
 global ROWS
 if ROWS is None: ROWS=[json.loads(x) for x in (DATA/"annotations.jsonl").read_text(encoding="utf-8").splitlines()]
 return ROWS
def key(text):return hashlib.sha256(text.encode()).hexdigest()[:24]

class SpeechTextTests(unittest.TestCase):
 def test_text_001_full_corpus_is_in_lexicon(self):
  specs=iter_speech_specs(BUNDLE);self.assertEqual(len(specs),1720)
  for s in specs:validate_text(s.text)
 def test_text_002_unknown_character_is_rejected(self):
  with self.assertRaises(ValueError):validate_text("未知A")
 def test_text_003_non_punctuation_has_structured_pinyin(self):
  for ch in set("".join(s.text for s in iter_speech_specs(BUNDLE))):
   item=character_label(ch)
   if not item["is_punctuation"]:
    self.assertTrue(item["pinyin"]);self.assertTrue(item["final"]);self.assertIn(item["tone"],range(1,6))

class SpeechDataTests(unittest.TestCase):
 def test_audio_001_flac_format_is_fixed(self):
  for row in rows()[::173]:
   info=sf.info(DATA/row["audio_path"]);self.assertEqual((info.samplerate,info.channels,info.subtype),(22050,1,"PCM_16"))
 def test_align_001_all_boundaries_are_legal(self):
  for row in rows():
   a=annotation_from_dict(row["annotation"]);self.assertEqual(len(a.tokens),len(a.text))
   for t in a.tokens:
    self.assertLessEqual(t.end_sample,a.num_samples);self.assertLessEqual(t.start_sample,t.end_sample)
    if not t.is_punctuation:self.assertGreater(t.end_sample,t.start_sample)
 def test_align_002_engine_clock_scale_is_recorded(self):
  scales=[row["annotation"]["alignment_time_scale"] for row in rows()]
  self.assertTrue(all(0<s<=1 for s in scales));self.assertLess(min(scales),.9)
 def test_mel_001_features_are_finite_and_aligned(self):
  for row in rows()[::173]:
   mel=np.load(DATA/row["logmel_path"],allow_pickle=False);a=row["annotation"]
   self.assertEqual(tuple(mel.shape),tuple(a["logmel_shape"]));self.assertEqual(mel.shape[0],64);self.assertTrue(np.isfinite(mel).all())
 def test_mel_002_higher_tone_has_higher_peak_band(self):
  cfg=SpeechRenderConfig();peaks=[]
  for cue in ("cue_0","cue_4"):
   wave=_cue_prefix(cue,cfg.sample_rate,cfg);peaks.append(int(np.argmax(log_mel(wave,cfg.sample_rate).mean(axis=1))))
  self.assertGreater(peaks[1],peaks[0])
 def test_cue_001_prefix_frequencies_are_distinct(self):
  cfg=SpeechRenderConfig();found=[]
  for cue,freq in CUE_FREQUENCIES.items():
   w=_cue_prefix(cue,cfg.sample_rate,cfg);peak=np.fft.rfftfreq(len(w),1/cfg.sample_rate)[np.argmax(abs(np.fft.rfft(w)))];found.append(round(peak));self.assertLess(abs(peak-freq),10)
  self.assertEqual(len(set(found)),5)
 def test_cue_002_disabling_cue_preserves_body(self):
  spec=next(s for s in iter_speech_specs(BUNDLE) if s.cue_token);k=key(spec.text);args=dict(utterance_id="x",scene_id=spec.scene_id,text=spec.text,wav_path=CACHE/f"{k}.wav",alignment_path=CACHE/f"{k}.json",cue_token=spec.cue_token)
  on=render_cached_speech(**args);off=render_cached_speech(**args,config=SpeechRenderConfig(enable_cue=False));self.assertTrue(np.array_equal(on.waveform[on.annotation.cue_samples:],off.waveform));self.assertEqual([t.end_sample-t.start_sample for t in on.annotation.tokens],[t.end_sample-t.start_sample for t in off.annotation.tokens])
 def test_cue_003_variants_obey_metadata_boundary(self):
  specs=iter_speech_specs(BUNDLE);self.assertEqual(sum(s.cue_token is not None for s in specs),460);self.assertEqual(sum(s.cue_token is None for s in specs),1260)

class SpeechArtifactTests(unittest.TestCase):
 def test_int_001_annotation_round_trip(self):
  a=annotation_from_dict(rows()[0]["annotation"]);self.assertEqual(annotation_from_dict(annotation_to_dict(a)),a)
 def test_int_002_full_artifact_verification(self):
  v=verify_speech_artifacts(DATA);self.assertTrue(v["verified"]);self.assertEqual(v["utterance_count"],1720);self.assertEqual(v["consistency_error_count"],0)
 def test_int_003_tampering_is_detected(self):
  row=rows()[0];p=DATA/row["audio_path"];original=p.read_bytes()
  try:
   p.write_bytes(original+b"x")
   with self.assertRaisesRegex(ValueError,"SHA-256"):verify_speech_artifacts(DATA)
  finally:p.write_bytes(original)
 def test_speech_002_duplicate_texts_share_identical_base_cache(self):
  specs=iter_speech_specs(BUNDLE);seen={};duplicate=None
  for s in specs:
   if s.text in seen:duplicate=(seen[s.text],s);break
   seen[s.text]=s
  self.assertIsNotNone(duplicate);self.assertTrue((CACHE/f"{key(duplicate[0].text)}.wav").is_file())
if __name__=="__main__":unittest.main()
