"""Run P3-W02/PF001: frozen HiFi-GAN copy-synthesis feasibility test."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np
import psutil
import soundfile as sf
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.asr import CharacterCTCTokenizer, TinyASR, ctc_greedy_decode  # noqa: E402
from mini_multimodal.speech.features import log_mel  # noqa: E402
from mini_multimodal.speech_vae import HiFiGANMelTransform, load_frozen_hifigan_v3  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for column, right_item in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (left_item != right_item))
            )
        previous = current
    return previous[-1]


def load_waveform(path: Path, expected_rate: int) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != expected_rate:
        raise ValueError(f"Unexpected sample rate for {path}: {sample_rate}")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return np.asarray(waveform, dtype=np.float32)


def current_logmel(waveform: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
    return log_mel(
        waveform,
        sample_rate,
        n_fft=512,
        hop_length=256,
        n_mels=64,
        fmin=40.0,
        fmax=8000.0,
    )


def aligned_mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    frames = min(reference.shape[1], candidate.shape[1])
    return float(np.abs(reference[:, :frames] - candidate[:, :frames]).mean())


def load_asr(package_path: Path) -> tuple[TinyASR, CharacterCTCTokenizer]:
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    tokenizer = CharacterCTCTokenizer(package["vocabulary"])
    model = TinyASR(vocabulary_size=len(tokenizer), **package["model_config"])
    model.load_state_dict(package["model_state"], strict=True)
    model.eval().requires_grad_(False)
    return model, tokenizer


def transcribe(
    waveform: np.ndarray,
    model: TinyASR,
    tokenizer: CharacterCTCTokenizer,
) -> str:
    feature = current_logmel(waveform).T
    feature = (feature - feature.mean(axis=0, keepdims=True)) / np.maximum(
        feature.std(axis=0, keepdims=True), 1e-5
    )
    tensor = torch.from_numpy(feature.astype(np.float32))[None]
    with torch.inference_mode():
        output = model(tensor, torch.tensor([tensor.shape[1]]))
    return ctc_greedy_decode(output.logits, output.output_lengths, tokenizer)[0]


def save_comparison_plot(path: Path, waveforms: list[np.ndarray], sample_rate: int) -> None:
    features = [current_logmel(waveform, sample_rate) for waveform in waveforms]
    lower = min(float(feature.min()) for feature in features)
    upper = max(float(feature.max()) for feature in features)
    titles = ["A. source", "B. Griffin-Lim", "C. HiFi-GAN V3"]
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for axis, feature, title in zip(axes, features, titles):
        axis.imshow(feature, origin="lower", aspect="auto", cmap="magma", vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.set_ylabel("Mel bin")
    axes[-1].set_xlabel("Frame")
    figure.savefig(path, dpi=130)
    plt.close(figure)


def junit_report(path: Path, checks: list[dict]) -> None:
    machine = [item for item in checks if item["status"] != "pending"]
    suite = ET.Element(
        "testsuite",
        name="P3-W02-PF001-HiFiGAN",
        tests=str(len(machine)),
        failures=str(sum(item["status"] == "failed" for item in machine)),
        skipped="0",
    )
    for item in machine:
        case = ET.SubElement(suite, "testcase", classname="p3_w02_pf001", name=item["acceptance_id"])
        if item["status"] == "failed":
            failure = ET.SubElement(case, "failure", message=item["summary"])
            failure.text = json.dumps(item.get("observed", {}), ensure_ascii=False)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    config_path = ROOT / "configs/p3_w02_pf001_hifigan.json"
    config = read_json(config_path)
    output_dir = ROOT / config["output_dir"]
    media_dir = output_dir / "examples/media"
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])

    rows = [json.loads(line) for line in (ROOT / config["source_asr_records"]).read_text(encoding="utf-8").splitlines()]
    by_id = {row["record_id"]: row for row in rows}
    missing_ids = [record_id for record_id in config["sample_record_ids"] if record_id not in by_id]
    if missing_ids:
        raise RuntimeError(f"Frozen records are missing: {missing_ids}")
    selected = [by_id[record_id] for record_id in config["sample_record_ids"]]
    if any(row["split"] != "test" for row in selected):
        raise RuntimeError("All frozen PF001 records must belong to the test split")

    source_before = {row["record_id"]: sha256_file(ROOT / row["audio_path"]) for row in selected}
    manifest_rows = []
    for row in selected:
        if source_before[row["record_id"]] != row["audio_sha256"]:
            raise RuntimeError(f"Source hash mismatch: {row['record_id']}")
        manifest_rows.append(
            {
                "record_id": row["record_id"],
                "split": row["split"],
                "source_audio_path": row["audio_path"],
                "source_audio_sha256": source_before[row["record_id"]],
                "source_logmel_path": row["logmel_path"],
                "source_logmel_sha256": row["logmel_sha256"],
                "reference": row["normalized_transcript"],
            }
        )
    manifest_path = output_dir / "sample_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in manifest_rows),
        encoding="utf-8",
    )

    weight_path = ROOT / config["model"]["weight_path"]
    provenance = read_json(weight_path.parent / "provenance.json")
    actual_weight_hash = sha256_file(weight_path)
    actual_weight_bytes = weight_path.stat().st_size
    if actual_weight_hash != provenance["weight_sha256"]:
        raise RuntimeError("HiFi-GAN weight SHA-256 does not match provenance")

    load_start = time.perf_counter()
    vocoder = load_frozen_hifigan_v3(str(weight_path))
    model_load_seconds = time.perf_counter() - load_start
    parameter_count = sum(parameter.numel() for parameter in vocoder.parameters())
    mel_transform = HiFiGANMelTransform(**{
        "sample_rate": config["mel"]["sample_rate"],
        "n_fft": config["mel"]["n_fft"],
        "win_length": config["mel"]["win_length"],
        "hop_length": config["mel"]["hop_length"],
        "n_mels": config["mel"]["n_mels"],
        "f_min": config["mel"]["f_min"],
        "f_max": config["mel"]["f_max"],
        "log_floor": config["mel"]["log_floor"],
    })
    mel_transform.eval()
    asr_model, tokenizer = load_asr(ROOT / config["tiny_asr_package"])

    baseline = read_json(ROOT / config["griffin_lim_baseline"])
    baseline_by_id = {item["record_id"]: item for item in baseline["predictions"]}
    missing_baselines = [row["record_id"] for row in selected if row["record_id"] not in baseline_by_id]
    if missing_baselines:
        raise RuntimeError(f"Frozen Griffin-Lim baselines are missing: {missing_baselines}")

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    total_hifigan_seconds = 0.0
    total_audio_seconds = 0.0
    predictions = []
    repeat_difference = None
    repeat_hashes = None

    for index, row in enumerate(selected):
        record_id = row["record_id"]
        stem = record_id.replace(":", "_")
        source_path = ROOT / row["audio_path"]
        griffin_path = ROOT / baseline_by_id[record_id]["audio_path"]
        source = load_waveform(source_path, config["mel"]["sample_rate"])
        griffin = load_waveform(griffin_path, config["mel"]["sample_rate"])

        source_tensor = torch.from_numpy(source)[None]
        with torch.inference_mode():
            mel80 = mel_transform(source_tensor)
            started = time.perf_counter()
            generated_tensor = vocoder(mel80)
            hifigan_seconds = time.perf_counter() - started
        generated = generated_tensor[0, 0].cpu().numpy().astype(np.float32)
        total_hifigan_seconds += hifigan_seconds
        total_audio_seconds += source.size / config["mel"]["sample_rate"]
        peak_rss = max(peak_rss, process.memory_info().rss)

        if index == 0:
            with torch.inference_mode():
                repeated = vocoder(mel80)[0, 0].cpu().numpy().astype(np.float32)
            repeat_difference = float(np.max(np.abs(generated - repeated)))
            repeat_hashes = [sha256_array(generated), sha256_array(repeated)]

        source_copy = media_dir / f"{stem}_A_source.flac"
        griffin_copy = media_dir / f"{stem}_B_griffin_lim.flac"
        hifigan_path = media_dir / f"{stem}_C_hifigan.flac"
        shutil.copy2(source_path, source_copy)
        shutil.copy2(griffin_path, griffin_copy)
        sf.write(hifigan_path, generated, config["mel"]["sample_rate"], format="FLAC", subtype="PCM_16")

        source64 = np.load(ROOT / row["logmel_path"], allow_pickle=False).astype(np.float32)
        griffin64 = current_logmel(griffin)
        hifigan64 = current_logmel(generated)
        original_prediction = transcribe(source, asr_model, tokenizer)
        griffin_prediction = transcribe(griffin, asr_model, tokenizer)
        hifigan_prediction = transcribe(generated, asr_model, tokenizer)
        reference = row["normalized_transcript"]
        duration_error = abs(generated.size - source.size) / source.size
        peak = float(np.max(np.abs(generated)))
        hard_clip_samples = int(np.count_nonzero(np.abs(generated) >= 1.0))
        predictions.append(
            {
                "record_id": record_id,
                "reference": reference,
                "mel80_shape": list(mel80.shape),
                "source_duration_seconds": source.size / config["mel"]["sample_rate"],
                "hifigan_duration_seconds": generated.size / config["mel"]["sample_rate"],
                "duration_relative_error": duration_error,
                "hifigan_inference_seconds": hifigan_seconds,
                "hifigan_rtf": hifigan_seconds / (source.size / config["mel"]["sample_rate"]),
                "hifigan_peak": peak,
                "hifigan_hard_clip_samples": hard_clip_samples,
                "source_prediction": original_prediction,
                "griffin_lim_prediction": griffin_prediction,
                "hifigan_prediction": hifigan_prediction,
                "source_character_errors": edit_distance(reference, original_prediction),
                "griffin_lim_character_errors": edit_distance(reference, griffin_prediction),
                "hifigan_character_errors": edit_distance(reference, hifigan_prediction),
                "reference_characters": len(reference),
                "griffin_lim_cycle_logmel_mae": aligned_mae(source64, griffin64),
                "hifigan_cycle_logmel_mae": aligned_mae(source64, hifigan64),
                "media": {
                    "source": source_copy.relative_to(output_dir).as_posix(),
                    "griffin_lim": griffin_copy.relative_to(output_dir).as_posix(),
                    "hifigan": hifigan_path.relative_to(output_dir).as_posix(),
                },
            }
        )

    total_characters = sum(item["reference_characters"] for item in predictions)
    source_cer = sum(item["source_character_errors"] for item in predictions) / total_characters
    griffin_cer = sum(item["griffin_lim_character_errors"] for item in predictions) / total_characters
    hifigan_cer = sum(item["hifigan_character_errors"] for item in predictions) / total_characters
    vocoder_rtf = total_hifigan_seconds / total_audio_seconds
    all_finite = all(np.isfinite(load_waveform(output_dir / item["media"]["hifigan"], 22050)).all() for item in predictions)
    maximum_duration_error = max(item["duration_relative_error"] for item in predictions)
    maximum_peak = max(item["hifigan_peak"] for item in predictions)
    clipped_examples = sum(item["hifigan_hard_clip_samples"] > 0 for item in predictions)

    metrics = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "preflight_id": "PF001",
        "sample_count": len(predictions),
        "total_audio_seconds": total_audio_seconds,
        "all_hifigan_waveforms_finite": all_finite,
        "maximum_duration_relative_error": maximum_duration_error,
        "maximum_hifigan_peak": maximum_peak,
        "hard_clipped_example_count": clipped_examples,
        "source_asr_cer": source_cer,
        "griffin_lim_asr_cer": griffin_cer,
        "hifigan_asr_cer": hifigan_cer,
        "mean_griffin_lim_cycle_logmel_mae": float(np.mean([item["griffin_lim_cycle_logmel_mae"] for item in predictions])),
        "mean_hifigan_cycle_logmel_mae": float(np.mean([item["hifigan_cycle_logmel_mae"] for item in predictions])),
        "predictions": predictions,
    }
    write_json(output_dir / "metrics.json", metrics)
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in predictions),
        encoding="utf-8",
    )

    resource = {
        "schema_version": 1,
        "device": "cpu",
        "cpu_threads": config["cpu_threads"],
        "parameter_count": parameter_count,
        "weight_bytes": actual_weight_bytes,
        "model_load_seconds": model_load_seconds,
        "total_vocoder_inference_seconds": total_hifigan_seconds,
        "total_audio_seconds": total_audio_seconds,
        "vocoder_rtf": vocoder_rtf,
        "peak_observed_rss_bytes": peak_rss,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    write_json(output_dir / "resource_benchmark.json", resource)
    reproducibility = {
        "schema_version": 1,
        "record_id": selected[0]["record_id"],
        "maximum_absolute_difference": repeat_difference,
        "array_sha256_run_1": repeat_hashes[0],
        "array_sha256_run_2": repeat_hashes[1],
        "identical": repeat_hashes[0] == repeat_hashes[1],
    }
    write_json(output_dir / "reproducibility_report.json", reproducibility)

    source_after = {row["record_id"]: sha256_file(ROOT / row["audio_path"]) for row in selected}
    source_integrity = {
        "schema_version": 1,
        "sample_count": len(selected),
        "all_sources_unchanged": source_before == source_after,
        "records": [
            {
                "record_id": record_id,
                "sha256_before": source_before[record_id],
                "sha256_after": source_after[record_id],
                "unchanged": source_before[record_id] == source_after[record_id],
            }
            for record_id in config["sample_record_ids"]
        ],
    }
    write_json(output_dir / "source_integrity.json", source_integrity)
    model_provenance = dict(provenance)
    model_provenance.update(
        {
            "actual_weight_sha256": actual_weight_hash,
            "actual_weight_bytes": actual_weight_bytes,
            "strict_load": True,
            "all_parameters_frozen": not any(parameter.requires_grad for parameter in vocoder.parameters()),
        }
    )
    write_json(output_dir / "model_provenance.json", model_provenance)

    thresholds = config["thresholds"]
    checks = [
        {"acceptance_id": "ENV-HFG-001", "status": "passed" if actual_weight_hash == provenance["weight_sha256"] else "failed", "summary": "权重来源、哈希和严格加载", "observed": {"sha256": actual_weight_hash, "strict_load": True}},
        {"acceptance_id": "UNIT-HFG-001", "status": "passed" if repeat_difference <= thresholds["maximum_repeat_difference"] and maximum_peak <= thresholds["maximum_output_peak"] and not any(parameter.requires_grad for parameter in vocoder.parameters()) else "failed", "summary": "形状、有限值、确定性和冻结性", "observed": {"repeat_difference": repeat_difference, "maximum_peak": maximum_peak, "all_frozen": True}},
        {"acceptance_id": "DATA-HFG-001", "status": "passed" if len(selected) == 12 and source_integrity["all_sources_unchanged"] and all(item["mel80_shape"][1] == 80 for item in predictions) else "failed", "summary": "固定样例、80维Mel和源数据完整性", "observed": {"sample_count": len(selected), "sources_unchanged": source_integrity["all_sources_unchanged"]}},
        {"acceptance_id": "MODEL-HFG-001", "status": "passed" if all_finite and maximum_duration_error <= thresholds["maximum_duration_relative_error"] and clipped_examples == 0 else "failed", "summary": "稳定复制合成、时长和削波", "observed": {"all_finite": all_finite, "maximum_duration_relative_error": maximum_duration_error, "hard_clipped_example_count": clipped_examples}},
        {"acceptance_id": "MODEL-HFG-002", "status": "passed" if hifigan_cer <= thresholds["maximum_asr_cer"] else "failed", "summary": "TinyASR中文回译可懂度", "observed": {"hifigan_asr_cer": hifigan_cer}},
        {"acceptance_id": "COMP-HFG-001", "status": "passed", "summary": "同句Griffin-Lim与HiFi-GAN机器对照完整", "observed": {"griffin_lim_asr_cer": griffin_cer, "hifigan_asr_cer": hifigan_cer}},
        {"acceptance_id": "RES-HFG-001", "status": "passed" if parameter_count <= thresholds["maximum_parameters"] and actual_weight_bytes <= thresholds["maximum_weight_bytes"] and vocoder_rtf <= thresholds["maximum_vocoder_rtf"] and peak_rss <= thresholds["maximum_peak_rss_bytes"] else "failed", "summary": "CPU时间、参数、权重和内存", "observed": {"parameter_count": parameter_count, "weight_bytes": actual_weight_bytes, "vocoder_rtf": vocoder_rtf, "peak_rss_bytes": peak_rss}},
        {"acceptance_id": "PED-HFG-001", "status": "pending", "summary": "直观样例已生成，等待教师A/B/C试听", "observed": {"teacher_perceptual_decision": "pending"}},
        {"acceptance_id": "REPRO-HFG-001", "status": "passed" if reproducibility["identical"] else "failed", "summary": "固定输入重复推理", "observed": reproducibility},
    ]

    sorted_worst = sorted(predictions, key=lambda item: (item["hifigan_character_errors"], item["hifigan_cycle_logmel_mae"]), reverse=True)
    example_ids = []
    for item in predictions[:4] + sorted_worst:
        if item["record_id"] not in example_ids:
            example_ids.append(item["record_id"])
        if len(example_ids) == config["capability_example_count"]:
            break
    examples = []
    for item in predictions:
        if item["record_id"] not in example_ids:
            continue
        stem = item["record_id"].replace(":", "_")
        plot_path = media_dir / f"{stem}_comparison.png"
        waveforms = [load_waveform(output_dir / item["media"][key], 22050) for key in ("source", "griffin_lim", "hifigan")]
        save_comparison_plot(plot_path, waveforms, 22050)
        example = dict(item)
        example["media"] = {
            key: Path(value).relative_to("examples").as_posix() for key, value in item["media"].items()
        }
        example["media"]["comparison_plot"] = plot_path.relative_to(output_dir / "examples").as_posix()
        example["selection_role"] = "fixed_typical" if item in predictions[:4] else "worst_machine_metric"
        examples.append(example)
    examples.sort(key=lambda item: example_ids.index(item["record_id"]))
    capability_json = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "preflight_id": "PF001",
        "teacher_perceptual_decision": "pending",
        "listening_instruction": "请用同一耳机和音量，依次听A原声、B Griffin-Lim、C HiFi-GAN，重点比较可懂度、金属感/水下感、齿音和音色稳定性。",
        "examples": examples,
    }
    write_json(output_dir / "examples/capability_examples.json", capability_json)
    markdown = [
        "# P3-W02 / PF001 HiFi-GAN直观能力样例",
        "",
        "> 状态：机器证据已生成；教师听感判断待填写。  ",
        "> A为原始Windows合成语音，B为现有64维Mel+Griffin-Lim，C为特征匹配的80维Mel+冻结HiFi-GAN V3。",
        "",
        "建议使用同一副耳机和相同音量依次试听A、B、C。频谱图只帮助定位差异，不能代替听感判断。",
        "",
    ]
    for number, item in enumerate(examples, 1):
        markdown.extend(
            [
                f"## 样例{number}：`{item['record_id']}`",
                "",
                f"- 参考文本：{item['reference']}",
                f"- 选样角色：`{item['selection_role']}`",
                f"- A原声回译：{item['source_prediction']}",
                f"- B Griffin-Lim回译：{item['griffin_lim_prediction']}",
                f"- C HiFi-GAN回译：{item['hifigan_prediction']}",
                f"- 循环64维Log-Mel MAE：B=`{item['griffin_lim_cycle_logmel_mae']:.4f}`，C=`{item['hifigan_cycle_logmel_mae']:.4f}`",
                "",
                f"![A 原声]({item['media']['source']})",
                "",
                f"![B Griffin-Lim]({item['media']['griffin_lim']})",
                "",
                f"![C HiFi-GAN]({item['media']['hifigan']})",
                "",
                f"![同尺度频谱对比]({item['media']['comparison_plot']})",
                "",
                "- 教师听感批注：【待填写】",
                "",
            ]
        )
    (output_dir / "examples/capability_examples.md").write_text("\n".join(markdown), encoding="utf-8")

    required_paths = [
        output_dir / "metrics.json",
        output_dir / "predictions.jsonl",
        output_dir / "resource_benchmark.json",
        output_dir / "reproducibility_report.json",
        output_dir / "source_integrity.json",
        output_dir / "model_provenance.json",
        output_dir / "sample_manifest.jsonl",
        output_dir / "examples/capability_examples.json",
        output_dir / "examples/capability_examples.md",
    ]
    media_paths = [output_dir / "examples" / media for item in examples for media in item["media"].values()]
    all_artifacts_exist = all(path.exists() for path in required_paths + media_paths)
    checks.append(
        {"acceptance_id": "ART-HFG-001", "status": "passed" if all_artifacts_exist and len(examples) == 8 else "failed", "summary": "证据、8条样例和媒体链接完整", "observed": {"all_artifacts_exist": all_artifacts_exist, "example_count": len(examples)}}
    )
    machine_checks = [item for item in checks if item["status"] != "pending"]
    machine_passed = all(item["status"] == "passed" for item in machine_checks)
    test_report = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "preflight_id": "PF001",
        "machine_feasibility": "passed" if machine_passed else "failed",
        "perceptual_feasibility": "pending_teacher_review",
        "checks": checks,
    }
    write_json(output_dir / "test_report.json", test_report)
    junit_report(output_dir / "p3_w02_pf001_junit.xml", checks)

    evidence_files = required_paths + [output_dir / "test_report.json", output_dir / "p3_w02_pf001_junit.xml"] + media_paths
    pytest_junit = output_dir / "p3_w02_pf001_pytest_junit.xml"
    if pytest_junit.exists():
        evidence_files.append(pytest_junit)
    full_regression_junit = output_dir / "p3_w02_pf001_full_regression_junit.xml"
    if full_regression_junit.exists():
        evidence_files.append(full_regression_junit)
    evidence_index = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "preflight_id": "PF001",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(set(evidence_files))
        ],
    }
    write_json(output_dir / "evidence_index.json", evidence_index)
    print(
        json.dumps(
            {
                "machine_feasibility": test_report["machine_feasibility"],
                "perceptual_feasibility": test_report["perceptual_feasibility"],
                "sample_count": len(predictions),
                "griffin_lim_asr_cer": griffin_cer,
                "hifigan_asr_cer": hifigan_cer,
                "griffin_lim_cycle_logmel_mae": metrics["mean_griffin_lim_cycle_logmel_mae"],
                "hifigan_cycle_logmel_mae": metrics["mean_hifigan_cycle_logmel_mae"],
                "vocoder_rtf": vocoder_rtf,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
