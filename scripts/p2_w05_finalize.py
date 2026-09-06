"""Finalize evidence around the completed P2-W05 I001 run without retraining."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.asr import ASRDataset, CharacterCTCTokenizer, TinyASR, collate_asr, forced_greedy_spans  # noqa: E402
from mini_multimodal.training.checkpoint import load_checkpoint  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


config = json.loads((ROOT / "configs/p2_w05_tiny_asr.json").read_text(encoding="utf-8"))
data_root = ROOT / config["output_data"]
result_root = ROOT / config["output_results"]
branch = result_root / "iterations/I001/branches/baseline"
results = branch / "results"
checkpoints = branch / "checkpoints"
vocabulary = json.loads((data_root / "vocabulary.json").read_text(encoding="utf-8"))
tokenizer = CharacterCTCTokenizer(vocabulary)
model = TinyASR(vocabulary_size=len(tokenizer), **config["model"])
state = load_checkpoint(checkpoints / "best.pt", model=model, restore_rng=False)
model.eval()

predictions = [json.loads(line) for line in (results / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
by_record = {row["record_id"]: row for row in [json.loads(line) for line in (data_root / "records.jsonl").read_text(encoding="utf-8").splitlines()]}
errors = sorted([item for item in predictions if item["character_edits"] > 0], key=lambda x: (-x["cer"], x["record_id"]))[:5]
correct = sorted([item for item in predictions if item["character_edits"] == 0], key=lambda x: x["record_id"])[: max(5, 10 - len(errors))]
examples = []
for item in correct + errors:
    dataset = ASRDataset(ROOT, data_dir=config["output_data"], record_ids=[item["record_id"]])
    batch = collate_asr([dataset[0]])
    with torch.no_grad(): output = model(batch["logmels"], batch["input_lengths"], return_attention=True)
    audio_path = by_record[item["record_id"]]["audio_path"]
    logmel_path = by_record[item["record_id"]]["logmel_path"]
    spans = forced_greedy_spans(output.logits[0], int(output.output_lengths[0]), tokenizer)
    examples.append({
        "source_record_id": item["record_id"],
        "input_artifacts": [audio_path, logmel_path],
        "display_artifacts": [audio_path],
        "expected_output": {"transcript": item["reference"]},
        "model_outputs": {"transcript": item["prediction"], "cer": item["cer"], "character_edits": item["character_edits"], "greedy_spans": spans},
        "is_fully_correct": item["character_edits"] == 0,
        "teaching_note": "干净同源合成语音上的正确识别" if item["character_edits"] == 0 else "测试集中唯一错误：绿色被识别为绿的色，适合分析CTC重复、插入与局部次序错误"
    })
write_json(results / "capability_examples.json", {"schema_version": 1, "work_package": "P2-W05", "iteration_id": "I001", "branch_id": "baseline", "example_count": len(examples), "examples": examples})
lines = ["# P2-W05 TinyASR直观能力样例", "", "> 10条样例全部由I001/baseline最佳检查点在固定测试集上生成。由于300条中只有1条错误，本页保留9条正确样例和全部1条错误样例，不人为制造错误。", ""]
for index, item in enumerate(examples, 1):
    output = item["model_outputs"]
    lines += [f"## 样例{index}：{'正确' if item['is_fully_correct'] else '错误'}", "", f"- 记录ID：`{item['source_record_id']}`", f"- 音频：`{item['display_artifacts'][0]}`", f"- 参考：{item['expected_output']['transcript']}", f"- 预测：{output['transcript'] or '（空）'}", f"- 该句CER：{output['cer']:.4f}", f"- 贪心路径非空字符段数：{len(output['greedy_spans'])}", f"- 教学说明：{item['teaching_note']}", ""]
(results / "capability_examples.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
shutil.copy2(results / "capability_examples.json", result_root / "capability_examples.json")
shutil.copy2(results / "capability_examples.md", result_root / "capability_examples.md")

package = {
    "schema_version": 1, "work_package": "P2-W05", "iteration_id": "I001", "branch_id": "baseline",
    "model_state": model.state_dict(), "model_config": config["model"], "vocabulary": vocabulary,
    "normalization": "per_utterance_per_mel_channel_mean_std", "source_checkpoint_sha256": hashlib.sha256((checkpoints / "best.pt").read_bytes()).hexdigest(),
}
torch.save(package, branch / "inference_package.pt")

evidence = {
    "schema_version": 1, "work_package": "P2-W05", "iteration_id": "I001", "branch_id": "baseline",
    "purpose": "P2-W05总结报告数值到I001机器结果字段的可追溯索引",
    "entries": [
        {"claim_id": "regular_test_cer", "value": 0.00009579461634256155, "source_file": "iterations/I001/branches/baseline/results/evaluation_metrics.json", "field_path": "regular_test.cer", "scope": "固定常规测试集", "sample_count": 300, "acceptance_item_id": "MODEL-ASR-002"},
        {"claim_id": "asset_nonoverlap_test_cer", "value": 0.00010676916506512919, "source_file": "iterations/I001/branches/baseline/results/evaluation_metrics.json", "field_path": "asset_nonoverlap_test.cer", "scope": "固定资产不重叠测试切片", "sample_count": 263, "acceptance_item_id": "MODEL-ASR-003"},
        {"claim_id": "text_nonoverlap_test_cer", "value": 0.00014017381553125876, "source_file": "iterations/I001/branches/baseline/results/evaluation_metrics.json", "field_path": "text_nonoverlap_test.cer", "scope": "固定文本不重叠测试切片", "sample_count": 185, "acceptance_item_id": "MODEL-ASR-003"},
        {"claim_id": "noise_cer_increase", "value": 0.44870967741935486, "source_file": "iterations/I001/branches/baseline/results/robustness_metrics.json", "field_path": "noise_15db.cer_increase_over_clean", "scope": "固定文本不重叠测试前60条的15 dB噪声派生挑战", "sample_count": 60, "acceptance_item_id": "ROB-ASR-001"},
        {"claim_id": "overfit_cer", "value": 0.15375153751537515, "source_file": "iterations/I001/branches/baseline/results/overfit_report.json", "field_path": "cer", "scope": "固定16条专用小集，80次更新", "sample_count": 16, "acceptance_item_id": "MODEL-ASR-001"},
        {"claim_id": "wall_seconds", "value": 178.8890284000081, "source_file": "iterations/I001/branches/baseline/results/resource_benchmark.json", "field_path": "wall_seconds", "scope": "I001完整训练评价流程", "sample_count": 1, "acceptance_item_id": "RES-ASR-001"}
    ],
    "predictions": "iterations/I001/branches/baseline/results/test_predictions.jsonl",
    "capability_examples": "capability_examples.md",
    "checkpoints": "iterations/I001/branches/baseline/checkpoints/checkpoint_manifest.json",
    "inference_package": "iterations/I001/branches/baseline/inference_package.pt"
}
write_json(result_root / "evidence_index.json", evidence)
contract = {
    "schema_version": 1, "work_package": "P2-W05", "status": "进行中", "result_directory": "results/P2-W05_tiny_asr",
    "design_document": "docs/P2-W05_TinyASR设计.md", "test_plan": "docs/P2-W05_测试计划.md", "summary_document": "docs/P2-W05_总结报告.md",
    "quantitative_results": True, "result_readme": "results/P2-W05_tiny_asr/README.md", "evidence_index": "results/P2-W05_tiny_asr/evidence_index.json",
    "capability_showcase": "results/P2-W05_tiny_asr/capability_examples.md", "capability_showcase_data": "results/P2-W05_tiny_asr/capability_examples.json",
    "checkpoint_manifest": "results/P2-W05_tiny_asr/iterations/I001/branches/baseline/checkpoints/checkpoint_manifest.json",
    "usage_document": "docs/P2-W05_检查点与推理使用说明.md", "runnable_examples": ["examples/p2_w05_inference.py", "examples/p2_w05_resume_checkpoint.py"],
    "automatic_test_reports": ["results/P2-W05_tiny_asr/pytest_results.xml", "results/P2-W05_tiny_asr/governance_pytest_results.xml", "results/P2-W05_tiny_asr/full_pytest_results.xml"],
    "required_artifacts": [
        {"path": "data/asr/p2_w05_v1/manifest.json", "role": "固定ASR派生清单", "validation": "DATA-ASR-001至004"},
        {"path": "results/P2-W05_tiny_asr/iterations/I001/branches/baseline/results/evaluation_metrics.json", "role": "主评价与防泄漏评价", "validation": "MODEL-ASR-002至004"},
        {"path": "results/P2-W05_tiny_asr/iterations/I001/branches/baseline/results/robustness_metrics.json", "role": "噪声与变速挑战", "validation": "ROB-ASR-001至002"},
        {"path": "results/P2-W05_tiny_asr/iterations/I001/branches/baseline/checkpoints/best.pt", "role": "最佳训练检查点", "validation": "ART-ASR-001"},
        {"path": "results/P2-W05_tiny_asr/iterations/I001/branches/baseline/inference_package.pt", "role": "自描述推理包", "validation": "ART-ASR-001"},
        {"path": "results/P2-W05_tiny_asr/capability_examples.md", "role": "正确与错误能力样例册", "validation": "PED-ASR-001"},
        {"path": "results/P2-W05_tiny_asr/capability_examples.json", "role": "能力样例机器清单", "validation": "PED-ASR-001"}
    ],
    "capability_interfaces": [
        {"output_path": "逐时间步字符分布与Mask", "implementation": "TinyASR.forward", "return_fields": ["logits", "output_lengths", "padding_mask", "attention"], "code_reference": "src/mini_multimodal/asr/model.py"},
        {"output_path": "CTC贪心文本", "implementation": "ctc_greedy_decode", "return_fields": ["normalized_transcript"], "code_reference": "src/mini_multimodal/asr/metrics.py"},
        {"output_path": "贪心字符跨度", "implementation": "forced_greedy_spans", "return_fields": ["token", "start_step", "end_step", "start_seconds", "end_seconds"], "code_reference": "src/mini_multimodal/asr/metrics.py"}
    ]
}
write_json(result_root / "artifact_contract.json", contract)
inventory = {}
for path in sorted(item for item in result_root.rglob("*") if item.is_file() and item.name != "artifact_inventory.json"):
    relative = path.relative_to(result_root).as_posix()
    inventory[relative] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
write_json(result_root / "artifact_inventory.json", {"schema_version": 1, "work_package": "P2-W05", "files": inventory})
