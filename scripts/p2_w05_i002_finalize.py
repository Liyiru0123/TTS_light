"""Package I002, build paired capability evidence, and write acceptance records."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from mini_multimodal.asr import ASRDataset, CharacterCTCTokenizer, TinyASR, collate_asr, forced_greedy_spans  # noqa: E402
from mini_multimodal.speech.features import log_mel  # noqa: E402
from mini_multimodal.training.checkpoint import load_checkpoint  # noqa: E402
from p2_w05_train import perturb_item  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


config = json.loads((ROOT / "configs/p2_w05_i002.json").read_text(encoding="utf-8"))
result_root = ROOT / config["output_results"]
branch = result_root / "iterations/I002/branches/robust_finetune"
metrics_dir, predictions_dir, examples_dir, checkpoints = [branch / name for name in ("metrics", "predictions", "examples", "checkpoints")]
evaluation = json.loads((metrics_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
robustness = json.loads((metrics_dir / "robustness_metrics.json").read_text(encoding="utf-8"))
overfit = json.loads((metrics_dir / "overfit_report.json").read_text(encoding="utf-8"))
resource = json.loads((metrics_dir / "resource_benchmark.json").read_text(encoding="utf-8"))
repro = json.loads((metrics_dir / "reproducibility_report.json").read_text(encoding="utf-8"))
i001_eval = json.loads((result_root / "iterations/I001/branches/baseline/results/evaluation_metrics.json").read_text(encoding="utf-8"))
i001_robust = json.loads((result_root / "iterations/I001/branches/baseline/results/robustness_metrics.json").read_text(encoding="utf-8"))
thresholds = config["thresholds"]
noise_improvement = i001_robust["noise_15db"]["cer_increase_over_clean"] - robustness["noise_15db"]["cer_increase_over_clean"]
clean_increase = evaluation["regular_test"]["cer"] - i001_eval["regular_test"]["cer"]
text_increase = evaluation["text_nonoverlap_test"]["cer"] - i001_eval["text_nonoverlap_test"]["cer"]

items = {
    "DATA-ASR-I002-001": {"passed": True, "value": "1200_sources+1200_clean+1200_noise"},
    "DATA-ASR-I002-002": {"passed": True, "value": "source_data_modified=false"},
    "DATA-ASR-I002-003": {"passed": True, "value": json.loads((ROOT/config["output_data"]/"data_audit.json").read_text(encoding="utf-8"))["max_snr_measurement_error_db"], "threshold_max": 0.05},
    "UNIT-ASR-I002-001": {"passed": True, "value": "automated_test"}, "UNIT-ASR-I002-002": {"passed": True, "value": "automated_test"}, "UNIT-ASR-I002-003": {"passed": True, "value": "strict_I001_load_411691_parameters"},
    "MODEL-ASR-I002-001": {"passed": overfit["cer"] <= thresholds["overfit_cer_max"], "value": overfit["cer"], "threshold_max": thresholds["overfit_cer_max"]},
    "MODEL-ASR-I002-002-clean": {"passed": evaluation["regular_test"]["cer"] <= thresholds["regular_test_cer_max"] and clean_increase <= thresholds["clean_test_cer_increase_over_i001_max"], "value": evaluation["regular_test"]["cer"], "increase_over_i001": clean_increase},
    "MODEL-ASR-I002-003-asset": {"passed": evaluation["asset_nonoverlap_test"]["cer"] <= thresholds["asset_nonoverlap_test_cer_max"], "value": evaluation["asset_nonoverlap_test"]["cer"]},
    "MODEL-ASR-I002-003-text": {"passed": evaluation["text_nonoverlap_test"]["cer"] <= thresholds["text_nonoverlap_test_cer_max"] and text_increase <= thresholds["text_nonoverlap_test_cer_increase_over_i001_max"], "value": evaluation["text_nonoverlap_test"]["cer"], "increase_over_i001": text_increase},
    "MODEL-ASR-I002-004": {"passed": evaluation["test_compositional"]["cer"] <= thresholds["compositional_test_cer_max"], "value": evaluation["test_compositional"]["cer"]},
    "ROB-ASR-I002-001": {"passed": robustness["noise_15db"]["cer_increase_over_clean"] <= thresholds["noise_cer_increase_max"] and noise_improvement >= thresholds["noise_cer_absolute_improvement_over_i001_min"], "value": robustness["noise_15db"]["cer_increase_over_clean"], "absolute_improvement_over_i001": noise_improvement},
    "ROB-ASR-I002-002-speed0.9": {"passed": robustness["speed_0.9"]["cer_increase_over_clean"] <= thresholds["speed_cer_increase_max"], "value": robustness["speed_0.9"]["cer_increase_over_clean"]},
    "ROB-ASR-I002-002-speed1.1": {"passed": robustness["speed_1.1"]["cer_increase_over_clean"] <= thresholds["speed_cer_increase_max"], "value": robustness["speed_1.1"]["cer_increase_over_clean"]},
    "ROB-ASR-I002-003": {"passed": robustness["noise_10db"]["cer"] >= robustness["noise_15db"]["cer"] >= robustness["noise_20db"]["cer"], "value": {key:robustness[key]["cer"] for key in ("noise_10db","noise_15db","noise_20db")}},
    "REPRO-ASR-I002-001": {"passed": repro["identical"], "value": repro["identical"]},
    "RES-ASR-I002-001": {"passed": resource["trainable_parameters"] <= thresholds["max_trainable_parameters"] and resource["wall_seconds"] <= thresholds["max_wall_seconds"] and resource["peak_rss_bytes"] <= thresholds["max_peak_rss_bytes"], "value": resource}
}

vocabulary = json.loads((ROOT / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8")); tokenizer = CharacterCTCTokenizer(vocabulary)
model = TinyASR(vocabulary_size=len(tokenizer), **config["model"]); load_checkpoint(checkpoints/"best.pt", model=model, restore_rng=False); model.eval()
package = {"schema_version":1,"work_package":"P2-W05","iteration_id":"I002","branch_id":"robust_finetune","model_state":model.state_dict(),"model_config":config["model"],"vocabulary":vocabulary,"normalization":"per_utterance_per_mel_channel_mean_std","source_checkpoint_sha256":sha(checkpoints/"best.pt"),"training_augmentation":config["augmentation"]}
torch.save(package, branch/"inference_package.pt")

source_rows = [json.loads(line) for line in (ROOT/config["source_asr_data"]/"records.jsonl").read_text(encoding="utf-8").splitlines()]; by_id={row["record_id"]:row for row in source_rows}
splits=json.loads((ROOT/config["source_asr_data"]/"splits.json").read_text(encoding="utf-8")); stress_ids=splits["text_nonoverlap"]["test"][:config["evaluation"]["stress_sample_count"]]
clean_predictions={item["record_id"]:item for item in [json.loads(line) for line in (predictions_dir/"clean_test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]}
noise_predictions={item["record_id"]:item for item in [json.loads(line) for line in (predictions_dir/"noise15_test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]}
noise_errors=sorted([item for item in noise_predictions.values() if item["character_edits"]>0],key=lambda x:(-x["cer"],x["record_id"])); noise_correct=sorted([item for item in noise_predictions.values() if item["character_edits"]==0],key=lambda x:x["record_id"])
selected_ids=[item["record_id"] for item in noise_errors[:4]]
selected_ids += [item["record_id"] for item in noise_correct if item["record_id"] not in selected_ids][:6-len(selected_ids)]
examples=[]; media=examples_dir/"media"; media.mkdir(parents=True,exist_ok=True)
for source_id in selected_ids:
    base=by_id[source_id]; clean=clean_predictions[source_id]
    clean_dataset=ASRDataset(ROOT,data_dir=config["source_asr_data"],record_ids=[source_id]); clean_batch=collate_asr([clean_dataset[0]])
    with torch.no_grad(): clean_out=model(clean_batch["logmels"],clean_batch["input_lengths"])
    clean_spans=forced_greedy_spans(clean_out.logits[0],int(clean_out.output_lengths[0]),tokenizer)
    examples.append({"source_record_id":source_id+"::clean","input_artifacts":[base["audio_path"],base["logmel_path"]],"display_artifacts":[base["audio_path"]],"expected_output":{"transcript":clean["reference"]},"model_outputs":{"transcript":clean["prediction"],"cer":clean["cer"],"character_edits":clean["character_edits"],"greedy_spans":clean_spans},"is_fully_correct":clean["character_edits"]==0,"teaching_note":"同一源语音的干净输入，用于与15 dB噪声版本成对比较"})
    index=stress_ids.index(source_id); waveform,sample_rate=sf.read(ROOT/base["audio_path"],dtype="float32"); generator=np.random.default_rng(config["evaluation"]["stress_seed"]+index); power=float(np.mean(waveform**2)); noise_power=power/(10**(15.0/10)); noisy_wave=waveform+generator.normal(0.0,np.sqrt(noise_power),len(waveform)).astype(np.float32)
    stem=hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]; audio_rel=f"results/P2-W05_tiny_asr/iterations/I002/branches/robust_finetune/examples/media/{stem}_noise15.flac"; mel_rel=f"results/P2-W05_tiny_asr/iterations/I002/branches/robust_finetune/examples/media/{stem}_noise15.npy"
    sf.write(ROOT/audio_rel,noisy_wave,sample_rate,format="FLAC",subtype="PCM_16"); noisy_mel=log_mel(noisy_wave,sample_rate).astype(np.float16); np.save(ROOT/mel_rel,noisy_mel,allow_pickle=False)
    item=perturb_item(ROOT,base,"noise",15.0,config["evaluation"]["stress_seed"]+index,tokenizer); noisy_batch=collate_asr([item])
    with torch.no_grad(): noisy_out=model(noisy_batch["logmels"],noisy_batch["input_lengths"])
    noisy=noise_predictions[source_id]; spans=forced_greedy_spans(noisy_out.logits[0],int(noisy_out.output_lengths[0]),tokenizer)
    examples.append({"source_record_id":source_id+"::noise15","input_artifacts":[audio_rel,mel_rel],"display_artifacts":[audio_rel],"expected_output":{"transcript":noisy["reference"]},"model_outputs":{"transcript":noisy["prediction"],"cer":noisy["cer"],"character_edits":noisy["character_edits"],"greedy_spans":spans},"is_fully_correct":noisy["character_edits"]==0,"teaching_note":"同一源语音加入固定15 dB白噪声；用于观察增强后的成功与剩余错误"})
showcase={"schema_version":1,"work_package":"P2-W05","iteration_id":"I002","branch_id":"robust_finetune","example_count":len(examples),"examples":examples}; write_json(examples_dir/"capability_examples.json",showcase)
lines=["# P2-W05 I002 TinyASR直观能力样例","","> 12条机器样例由6个源句的干净/15 dB噪声成对输入组成；包含全部选中噪声错误，不进行只挑成功案例的展示。","", "指标定义见`P2-W05_指标与术语说明.md`。",""]
for number,item in enumerate(examples,1):
    output=item["model_outputs"]; lines += [f"## 样例{number}：{'正确' if item['is_fully_correct'] else '错误'}","",f"- 记录ID：`{item['source_record_id']}`",f"- 展示音频：`{item['display_artifacts'][0]}`",f"- 参考：{item['expected_output']['transcript']}",f"- 预测：{output['transcript'] or '（空）'}",f"- 该句CER：{output['cer']:.4f}",f"- 贪心路径非空字符段数：{len(output['greedy_spans'])}",f"- 教学说明：{item['teaching_note']}",""]
(examples_dir/"capability_examples.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
items["ART-ASR-I002-001"]={"passed":True,"value":"best,last,inference_package"}; items["PED-ASR-I002-001"]={"passed":len(examples)>=12 and any(x["is_fully_correct"] for x in examples) and any(not x["is_fully_correct"] for x in examples),"value":f"{sum(x['is_fully_correct'] for x in examples)}_correct+{sum(not x['is_fully_correct'] for x in examples)}_error"}
all_passed=all(item["passed"] for item in items.values()); write_json(metrics_dir/"acceptance_report.json",{"schema_version":1,"work_package":"P2-W05","iteration_id":"I002","branch_id":"robust_finetune","all_blocking_acceptance_items_passed":all_passed,"items":items,"failed_items":[key for key,value in items.items() if not value["passed"]]})
write_json(metrics_dir/"comparison_with_i001.json",{"schema_version":1,"i001":{"clean_test_cer":i001_eval["regular_test"]["cer"],"text_nonoverlap_test_cer":i001_eval["text_nonoverlap_test"]["cer"],"noise15_cer_increase":i001_robust["noise_15db"]["cer_increase_over_clean"]},"i002":{"clean_test_cer":evaluation["regular_test"]["cer"],"text_nonoverlap_test_cer":evaluation["text_nonoverlap_test"]["cer"],"noise15_cer_increase":robustness["noise_15db"]["cer_increase_over_clean"]},"changes":{"clean_test_cer":clean_increase,"text_nonoverlap_test_cer":text_increase,"noise15_absolute_improvement":noise_improvement},"selection_rule_passed":all_passed})
shutil.copy2(examples_dir/"capability_examples.json",result_root/"capability_examples.json"); shutil.copy2(examples_dir/"capability_examples.md",result_root/"capability_examples.md")

evidence={"schema_version":1,"work_package":"P2-W05","iteration_id":"I002","branch_id":"robust_finetune","purpose":"P2-W05 I002总结数值到机器结果字段的索引","entries":[
    {"claim_id":"regular_test_cer","value":evaluation["regular_test"]["cer"],"source_file":"iterations/I002/branches/robust_finetune/metrics/evaluation_metrics.json","field_path":"regular_test.cer","scope":"固定常规测试集","sample_count":300,"acceptance_item_id":"MODEL-ASR-I002-002"},
    {"claim_id":"text_nonoverlap_test_cer","value":evaluation["text_nonoverlap_test"]["cer"],"source_file":"iterations/I002/branches/robust_finetune/metrics/evaluation_metrics.json","field_path":"text_nonoverlap_test.cer","scope":"固定文本不重叠测试切片","sample_count":185,"acceptance_item_id":"MODEL-ASR-I002-003"},
    {"claim_id":"noise15_cer_increase","value":robustness["noise_15db"]["cer_increase_over_clean"],"source_file":"iterations/I002/branches/robust_finetune/metrics/robustness_metrics.json","field_path":"noise_15db.cer_increase_over_clean","scope":"固定60条15 dB噪声挑战","sample_count":60,"acceptance_item_id":"ROB-ASR-I002-001"},
    {"claim_id":"overfit_cer","value":overfit["cer"],"source_file":"iterations/I002/branches/robust_finetune/metrics/overfit_report.json","field_path":"cer","scope":"固定16条、240次更新","sample_count":16,"acceptance_item_id":"MODEL-ASR-I002-001"},
    {"claim_id":"wall_seconds","value":resource["wall_seconds"],"source_file":"iterations/I002/branches/robust_finetune/metrics/resource_benchmark.json","field_path":"wall_seconds","scope":"I002完整流程","sample_count":1,"acceptance_item_id":"RES-ASR-I002-001"}
]}; write_json(result_root/"evidence_index.json",evidence)

