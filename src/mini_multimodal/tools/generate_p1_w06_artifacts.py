"""Generate and verify P1-W06 video frame sequences and temporal labels."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from time import perf_counter
from mini_multimodal.data.artifacts import load_dataset_bundle,verify_dataset_bundle
from mini_multimodal.video.artifacts import export_video_artifacts,verify_video_artifacts
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
def main():
 root=Path(__file__).resolve().parents[3];source=root/"data"/"datasets"/"p1_w03_v1";out=root/"data"/"video"/"p1_w06_v1";results=root/"results"/"P1-W06_video_renderer";verify_dataset_bundle(source);bundle=load_dataset_bundle(source)
 start=perf_counter();manifest=export_video_artifacts(out,bundle);export_seconds=perf_counter()-start;start=perf_counter();v=verify_video_artifacts(out,bundle);verify_seconds=perf_counter()-start
 write(results/"video_statistics.json",v["statistics"]);write(results/"temporal_consistency.json",v["consistency"]);write(results/"shortcut_video_diagnostics.json",v["shortcut"]);write(results/"temporal_fixture_report.json",{"fixture_count":v["supplemental_temporal_fixture_count"],"event_counts":v["temporal_fixture_event_counts"]});write(results/"fixture_verification.json",{"verified":True,"video_count":v["video_count"],"supplemental_temporal_fixture_count":v["supplemental_temporal_fixture_count"],"managed_file_count":v["managed_file_count"],"manifest_sha256":hashlib.sha256((out/"manifest.json").read_bytes()).hexdigest()});write(results/"render_benchmark.json",{"export_seconds":export_seconds,"verification_seconds":verify_seconds,"video_count":v["video_count"],"frame_count":v["statistics"]["frame_count"],"managed_bytes":sum(x["bytes"] for x in manifest["files"].values())})
 print(json.dumps({"verified":True,"video_count":v["video_count"],"frame_count":v["statistics"]["frame_count"],"track_count":v["statistics"]["track_count"],"video_dir":str(out),"results_dir":str(results)},ensure_ascii=False))
if __name__=="__main__":main()
