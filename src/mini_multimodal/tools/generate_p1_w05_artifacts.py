"""Generate Huihui speech, Log-Mel features, alignments, and diagnostics."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from time import perf_counter
from mini_multimodal.data.artifacts import load_dataset_bundle,verify_dataset_bundle
from mini_multimodal.speech.artifacts import ensure_huihui_cache,export_speech_artifacts,verify_speech_artifacts
from mini_multimodal.speech.dataset import iter_speech_specs

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")

def main():
    root=Path(__file__).resolve().parents[3];source=root/"data"/"datasets"/"p1_w03_v1";out=root/"data"/"speech"/"p1_w05_v1";results=root/"results"/"P1-W05_speech_renderer";cache=root/"data"/"cache"/"p1_w05_huihui";bridge=root/"scripts"/"synthesize_huihui.ps1"
    verify_dataset_bundle(source);bundle=load_dataset_bundle(source);specs=iter_speech_specs(bundle)
    started=perf_counter();cache_map=ensure_huihui_cache([s.text for s in specs],cache,bridge);synthesis_seconds=perf_counter()-started
    started=perf_counter();manifest=export_speech_artifacts(out,bundle,cache_map);export_seconds=perf_counter()-started
    started=perf_counter();verification=verify_speech_artifacts(out);verification_seconds=perf_counter()-started
    rows=[json.loads(line) for line in (out/"annotations.jsonl").read_text(encoding="utf-8").splitlines()];scales=[row["annotation"]["alignment_time_scale"] for row in rows]
    write(results/"speech_statistics.json",verification["statistics"]);write(results/"shortcut_acoustic_diagnostics.json",verification["shortcut"]);write(results/"alignment_consistency.json",{"method":"SSML relative marks with utterance-level affine clock calibration","consistency_error_count":verification["consistency_error_count"],"utterance_count":verification["utterance_count"],"alignment_time_scale_min":min(scales),"alignment_time_scale_max":max(scales)});write(results/"engine_probe.json",{"probe":"SUCCESS","voice_name":"Microsoft Huihui Desktop","sample_rate":22050,"channels":1,"subtype":"PCM_16","access_note":"Classic Windows PowerShell STA; restricted sandboxes require local speech-service permission"});write(results/"fixture_verification.json",{"verified":True,"managed_file_count":verification["managed_file_count"],"manifest_sha256":hashlib.sha256((out/"manifest.json").read_bytes()).hexdigest()});write(results/"render_benchmark.json",{"unique_text_count":len(cache_map),"synthesis_seconds":synthesis_seconds,"export_seconds":export_seconds,"verification_seconds":verification_seconds,"utterance_count":len(specs),"managed_bytes":sum(x["bytes"] for x in manifest["files"].values())})
    print(json.dumps({"verified":True,"utterance_count":len(specs),"unique_text_count":len(cache_map),"total_duration_seconds":verification["statistics"]["total_duration_seconds"],"speech_dir":str(out),"results_dir":str(results)},ensure_ascii=False))
if __name__=="__main__":main()
