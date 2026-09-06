"""Approved post-snapshot packaging recovery for P3-W02 I002.

This script does not train or alter either model. It evaluates the frozen I001
package on I002's common utterances, creates A/B/C/D examples, and completes
the I002 inference package and latent cache.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.asr import CharacterCTCTokenizer, TinyASR, ctc_greedy_decode  # noqa: E402
from mini_multimodal.speech.features import log_mel  # noqa: E402
from mini_multimodal.speech_vae import (  # noqa: E402
    SpeechWindowDataset,
    TinySpeechVAE,
    collate_speech_windows,
    load_frozen_hifigan_v3,
    logmel_to_waveform,
    stitch_windows,
)
from mini_multimodal.training import load_checkpoint  # noqa: E402


RESULT = ROOT / "results/P3-W02_speech_vae"
ITERATION = RESULT / "iterations/I002"
BRANCH = ITERATION / "branches/hifigan_vae"
RECOVERY = ITERATION / "recovery"
COMPARISON = ITERATION / "comparisons/i001_common_core"
CONFIG_PATH = ROOT / "configs/p3_w02_i002_hifigan_vae.json"


PROTECTED_RELATIVE = [
    "source_snapshot.zip",
    "source_manifest.json",
    "configs/p3_w02_i002_hifigan_vae.json",
    "branches/hifigan_vae/checkpoints/best.pt",
    "branches/hifigan_vae/checkpoints/last.pt",
    "branches/hifigan_vae/training_log.jsonl",
    "branches/hifigan_vae/logs/reproducibility_report.json",
    "branches/hifigan_vae/logs/training_history.json",
    "branches/hifigan_vae/metrics/overfit_report.json",
    "branches/hifigan_vae/metrics/evaluation_metrics.json",
    "branches/hifigan_vae/metrics/utterance_metrics.json",
    "branches/hifigan_vae/predictions/test_window_predictions.jsonl",
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_manifest() -> list[dict]:
    rows = []
    for relative in PROTECTED_RELATIVE:
        path = ITERATION / relative
        if not path.is_file():
            raise RuntimeError(f"protected asset is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    d_audio = sorted((BRANCH / "examples/media").glob("*_D_i002_hifigan.flac"))
    if len(d_audio) != 12:
        raise RuntimeError(f"expected 12 protected I002 D audios, found {len(d_audio)}")
    for path in d_audio:
        rows.append({"path": path.relative_to(ITERATION).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for column, right_item in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (left_item != right_item)))
        previous = current
    return previous[-1]


def load_asr() -> tuple[TinyASR, CharacterCTCTokenizer]:
    package = torch.load(
        ROOT / "results/P2-W05_tiny_asr/iterations/I002/branches/robust_finetune/inference_package.pt",
        map_location="cpu",
        weights_only=False,
    )
    tokenizer = CharacterCTCTokenizer(package["vocabulary"])
    model = TinyASR(vocabulary_size=len(tokenizer), **package["model_config"])
    model.load_state_dict(package["model_state"], strict=True)
    model.eval().requires_grad_(False)
    return model, tokenizer


@torch.inference_mode()
def transcribe(waveform: np.ndarray, model: TinyASR, tokenizer: CharacterCTCTokenizer) -> str:
    feature = log_mel(waveform, 22050, n_fft=512, hop_length=256, n_mels=64, fmin=40.0, fmax=8000.0).T
    feature = (feature - feature.mean(axis=0, keepdims=True)) / np.maximum(feature.std(axis=0, keepdims=True), 1e-5)
    tensor = torch.from_numpy(feature.astype(np.float32))[None]
    output = model(tensor, torch.tensor([tensor.shape[1]]))
    return ctc_greedy_decode(output.logits, output.output_lengths, tokenizer)[0]


def denormalize(value: np.ndarray, feature: dict) -> np.ndarray:
    lower, upper = feature["minimum_logmel"], feature["maximum_logmel"]
    return np.clip((value + 1.0) * 0.5 * (upper - lower) + lower, lower, upper)


def load_plot_function():
    path = ROOT / "scripts/p3_w02_i002_train.py"
    spec = importlib.util.spec_from_file_location("p3_w02_i002_frozen", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.save_four_way_plot


@torch.inference_mode()
def evaluate_i001_common_set(config: dict) -> dict:
    package_path = ROOT / "results/P3-W02_speech_vae/iterations/I001/branches/spectral_vae/inference_package.pt"
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    if sha256(ROOT / "results/P3-W02_speech_vae/iterations/I001/branches/spectral_vae/checkpoints/best.pt") != package["source_checkpoint_sha256"]:
        raise RuntimeError("I001 inference package no longer matches its frozen best checkpoint")
    model = TinySpeechVAE(**package["model_config"])
    model.load_state_dict(package["model_state"], strict=True)
    model.eval().requires_grad_(False)
    rows = [
        json.loads(line)
        for line in (ROOT / "data/speech_vae/p3_w02_v1/window_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["source_utterance_id"] in config["evaluation_record_ids"]:
            groups.setdefault(row["source_utterance_id"], []).append(row)
    asr_rows = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / config["source_asr"] / "records.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    asr_model, tokenizer = load_asr()
    audio_dir = COMPARISON / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for record_id in config["evaluation_record_ids"]:
        record_rows = groups.get(record_id)
        if not record_rows:
            raise RuntimeError(f"I001 common record is missing: {record_id}")
        feature = np.load(ROOT / record_rows[0]["source_logmel_path"], allow_pickle=False).astype(np.float32)
        windows = []
        masks = []
        for row in record_rows:
            value = np.full((64, 256), package["window_config"]["padding_value"], dtype=np.float32)
            value[:, : row["valid_frames"]] = feature[:, row["start_frame"] : row["start_frame"] + row["valid_frames"]]
            value = np.clip((value - package["feature_config"]["minimum_logmel"]) / (package["feature_config"]["maximum_logmel"] - package["feature_config"]["minimum_logmel"]) * 2.0 - 1.0, -1.0, 1.0)
            windows.append(value)
            mask = np.zeros(256, dtype=np.bool_)
            mask[: row["valid_frames"]] = True
            masks.append(mask)
        output = model(torch.from_numpy(np.stack(windows))[:, None], sample_posterior=False)
        reconstructed = denormalize(output.reconstruction[:, 0].numpy(), package["feature_config"])
        stitched, coverage = stitch_windows(
            list(reconstructed),
            [row["start_frame"] for row in record_rows],
            [row["valid_frames"] for row in record_rows],
            record_rows[0]["source_frames"],
        )
        boundaries = []
        for index in range(len(record_rows) - 1):
            overlap = record_rows[index]["start_frame"] + record_rows[index]["valid_frames"] - record_rows[index + 1]["start_frame"]
            if overlap > 0:
                boundaries.append(float(np.abs(reconstructed[index][:, -overlap:] - reconstructed[index + 1][:, :overlap]).mean()))
        waveform_tensor = logmel_to_waveform(
            stitched,
            sample_rate=22050,
            n_fft=512,
            hop_length=256,
            fmin=40.0,
            fmax=8000.0,
            n_iter=package["vocoder_config"]["griffin_lim_iterations"],
            momentum=package["vocoder_config"]["momentum"],
        )
        waveform = waveform_tensor.numpy().astype(np.float32)
        cycle = log_mel(waveform, 22050, n_fft=512, hop_length=256, n_mels=64, fmin=40.0, fmax=8000.0)
        common = min(feature.shape[1], cycle.shape[1])
        reference = asr_rows[record_id]["normalized_transcript"]
        prediction = transcribe(waveform, asr_model, tokenizer)
        audio_path = audio_dir / f"{record_id.replace(':', '_')}_B_i001_vae_griffin_lim.flac"
        sf.write(audio_path, waveform, 22050, format="FLAC", subtype="PCM_16")
        results.append(
            {
                "record_id": record_id,
                "reference": reference,
                "prediction": prediction,
                "character_errors": edit_distance(reference, prediction),
                "reference_characters": len(reference),
                "asr_cer": edit_distance(reference, prediction) / len(reference),
                "overlap_boundary_mae": float(np.mean(boundaries)) if boundaries else 0.0,
                "direct_logmel_mae": float(np.abs(stitched - feature).mean()),
                "waveform_cycle_logmel_mae": float(np.abs(cycle[:, :common] - feature[:, :common]).mean()),
                "waveform_peak": float(np.abs(waveform).max()),
                "minimum_coverage": float(coverage.min()),
                "audio_path": audio_path.relative_to(ROOT).as_posix(),
            }
        )
    aggregate = {
        "sample_count": len(results),
        "asr_cer": sum(item["character_errors"] for item in results) / sum(item["reference_characters"] for item in results),
        "overlap_boundary_mae": float(np.mean([item["overlap_boundary_mae"] for item in results])),
        "direct_logmel_mae": float(np.mean([item["direct_logmel_mae"] for item in results])),
        "waveform_cycle_logmel_mae": float(np.mean([item["waveform_cycle_logmel_mae"] for item in results])),
        "maximum_waveform_peak": max(item["waveform_peak"] for item in results),
        "all_outputs_finite": all(np.isfinite(sf.read(ROOT / item["audio_path"], dtype="float32")[0]).all() for item in results),
    }
    report = {
        "schema_version": 1,
        "role": "supplemental_common_set_evaluation_not_i001_original_acceptance",
        "source_iteration": "I001/spectral_vae",
        "host_iteration": "I002/hifigan_vae",
        "source_inference_package": package_path.relative_to(ROOT).as_posix(),
        "source_inference_package_sha256": sha256(package_path),
        "record_ids": config["evaluation_record_ids"],
        "aggregate": aggregate,
        "utterances": results,
    }
    write_json(COMPARISON / "metrics.json", report)
    return report


def create_capability_examples(config: dict, i001_common: dict) -> dict:
    pf001 = json.loads((ROOT / config["pf001_dir"] / "metrics.json").read_text(encoding="utf-8"))
    pf001_by_id = {item["record_id"]: item for item in pf001["predictions"]}
    i002 = json.loads((BRANCH / "metrics/utterance_metrics.json").read_text(encoding="utf-8"))
    i002_by_id = {item["source_record_id"]: item for item in i002["utterances"]}
    i001_by_id = {item["record_id"]: item for item in i001_common["utterances"]}
    asr_rows = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / config["source_asr"] / "records.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    ranked = sorted(i002["utterances"], key=lambda item: (item["character_errors"], item["waveform_cycle_logmel_mae"]), reverse=True)
    selected_ids: list[str] = []
    for item in i002["utterances"][:4] + ranked:
        if item["source_record_id"] not in selected_ids:
            selected_ids.append(item["source_record_id"])
        if len(selected_ids) == 8:
            break
    media_dir = BRANCH / "examples/media"
    media_dir.mkdir(parents=True, exist_ok=True)
    save_plot = load_plot_function()
    examples = []
    for record_id in selected_ids:
        stem = record_id.replace(":", "_")
        sources = {
            "A_source": ROOT / asr_rows[record_id]["audio_path"],
            "B_i001_vae_griffin_lim": ROOT / i001_by_id[record_id]["audio_path"],
            "C_source_mel_hifigan": ROOT / config["pf001_dir"] / pf001_by_id[record_id]["media"]["hifigan"],
            "D_i002_vae_hifigan": ROOT / i002_by_id[record_id]["audio_path"],
        }
        destinations = {key: media_dir / f"{stem}_{key}.flac" for key in sources}
        for key in sources:
            shutil.copy2(sources[key], destinations[key])
        waveforms = [np.asarray(sf.read(destinations[key], dtype="float32", always_2d=False)[0], dtype=np.float32) for key in destinations]
        plot_path = media_dir / f"{stem}_ABCD_comparison.png"
        save_plot(plot_path, waveforms)
        d_item = i002_by_id[record_id]
        examples.append(
            {
                "source_record_id": record_id,
                "input_artifacts": [sources["A_source"].relative_to(ROOT).as_posix()],
                "display_artifacts": [path.relative_to(ROOT).as_posix() for path in destinations.values()]
                + [plot_path.relative_to(ROOT).as_posix()],
                "expected_output": d_item["reference"],
                "model_outputs": {
                    "i001_transcript": i001_by_id[record_id]["prediction"],
                    "i002_transcript": d_item["prediction"],
                    "i002_audio": destinations["D_i002_vae_hifigan"].relative_to(ROOT).as_posix(),
                },
                "reference": d_item["reference"],
                "i001_prediction": i001_by_id[record_id]["prediction"],
                "i002_prediction": d_item["prediction"],
                "selection_role": "fixed_typical" if record_id in config["evaluation_record_ids"][:4] else "worst_machine_metric",
                "i001_asr_cer": i001_by_id[record_id]["asr_cer"],
                "i001_cycle_logmel_mae": i001_by_id[record_id]["waveform_cycle_logmel_mae"],
                "pf001_hifigan_cycle_logmel_mae": pf001_by_id[record_id]["hifigan_cycle_logmel_mae"],
                "i002_asr_cer": d_item["asr_cer"],
                "i002_cycle_logmel_mae": d_item["waveform_cycle_logmel_mae"],
                "is_fully_correct": d_item["asr_cer"] == 0.0 and d_item["hard_clip_samples"] == 0,
                "media": {key: path.relative_to(BRANCH / "examples").as_posix() for key, path in destinations.items()}
                | {"comparison_plot": plot_path.relative_to(BRANCH / "examples").as_posix()},
                "teaching_note": "比较A/B/C/D，区分VAE误差、声码器误差、ASR可懂度与主观自然度。",
                "teacher_note": "待填写",
            }
        )
    payload = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "iteration_id": "I002",
        "branch_id": "hifigan_vae",
        "teacher_perceptual_decision": "pending",
        "example_count": len(examples),
        "examples": examples,
    }
    write_json(BRANCH / "examples/capability_examples.json", payload)
    lines = [
        "# P3-W02 I002 80维SpeechVAE＋HiFi-GAN直观能力样例",
        "",
        "> A为原声；B为I001 64维VAE＋Griffin-Lim在公共12句上的补充只读推理；C为真实80维Mel＋HiFi-GAN；D为I002 80维VAE＋HiFi-GAN。",
        "> B只用于本次同句比较，不改写I001原始验收。请重点听D相对B是否减少水下感/金属感，以及D相对C损失了哪些细节。",
        "> 指标定义见`docs/P3-W02_指标与术语说明.md`。",
        "",
    ]
    for number, item in enumerate(examples, 1):
        lines.extend(
            [
                f"## 样例{number}：`{item['source_record_id']}`",
                "",
                f"- 参考文本：{item['reference']}",
                f"- B I001回译：{item['i001_prediction']}",
                f"- D I002回译：{item['i002_prediction']}",
                f"- B/C/D循环64维Log-Mel MAE：`{item['i001_cycle_logmel_mae']:.4f}` / `{item['pf001_hifigan_cycle_logmel_mae']:.4f}` / `{item['i002_cycle_logmel_mae']:.4f}`",
                f"- D CER：`{item['i002_asr_cer']:.4f}`；选样角色：`{item['selection_role']}`",
                "",
                f"![A 原声]({item['media']['A_source']})",
                "",
                f"![B I001 VAE＋Griffin-Lim]({item['media']['B_i001_vae_griffin_lim']})",
                "",
                f"![C 真实Mel＋HiFi-GAN]({item['media']['C_source_mel_hifigan']})",
                "",
                f"![D I002 VAE＋HiFi-GAN]({item['media']['D_i002_vae_hifigan']})",
                "",
                f"![A/B/C/D同尺度频谱]({item['media']['comparison_plot']})",
                "",
                "- 教师听感批注：【待填写】",
                "",
            ]
        )
    (BRANCH / "examples/capability_examples.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def complete_model_delivery(config: dict) -> dict:
    model = TinySpeechVAE(**config["model"])
    load_checkpoint(BRANCH / "checkpoints/best.pt", model=model, optimizer=None, restore_rng=False)
    model.eval()
    all_data = SpeechWindowDataset(ROOT, data_dir=config["output_data"], splits=["train", "validation", "test"], config=config)
    data_loader = DataLoader(all_data, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_speech_windows)
    means, ids = [], []
    with torch.inference_mode():
        for batch in data_loader:
            means.append(model.encode(batch["window"])[0].numpy().astype(np.float16))
            ids.extend(item["window_id"] for item in batch["metadata"])
    latent_path = ROOT / config["output_latents"] / "posterior_means.npz"
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(latent_path, posterior_mean=np.concatenate(means), window_id=np.asarray(ids))
    package = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "iteration_id": "I002",
        "branch_id": "hifigan_vae",
        "model_state": model.state_dict(),
        "model_config": config["model"],
        "feature_config": config["feature"],
        "window_config": config["window"],
        "hifigan_weight_path": config["hifigan_weight"],
        "hifigan_weight_sha256": sha256(ROOT / config["hifigan_weight"]),
        "source_checkpoint_sha256": sha256(BRANCH / "checkpoints/best.pt"),
    }
    torch.save(package, BRANCH / "inference_package.pt")
    manifest = {"schema_version": 1, "iteration_id": "I002", "branch_id": "hifigan_vae", "files": []}
    for relative, role in (("checkpoints/best.pt", "best"), ("checkpoints/last.pt", "last"), ("inference_package.pt", "inference")):
        path = BRANCH / relative
        state = torch.load(path, map_location="cpu", weights_only=False) if role != "inference" else None
        manifest["files"].append(
            {
                "path": relative,
                "role": role,
                "epoch": state.get("epoch") if state else None,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "contains_optimizer_and_rng": role != "inference",
            }
        )
    write_json(BRANCH / "checkpoints/checkpoint_manifest.json", manifest)
    return {
        "latent_path": latent_path.relative_to(ROOT).as_posix(),
        "latent_sha256": sha256(latent_path),
        "latent_shape": [len(all_data), 4, 20, 64],
        "inference_package_sha256": sha256(BRANCH / "inference_package.pt"),
    }


def write_source_delta_manifest() -> None:
    files = [
        ROOT / "scripts/p3_w02_i002_recover_packaging.py",
        ROOT / "scripts/p3_w02_i002_finalize_recovery.py",
        ROOT / "scripts/p3_w02_refresh_inventory.py",
        ROOT / "tests/test_p3_w02_i002_recovery.py",
        ROOT / "docs/P3-W02_I002_训练后打包修正计划.md",
        ROOT / "docs/迭代与发布管理机制.md",
        ROOT / "docs/项目记忆与工作包检查机制.md",
    ]
    payload = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "iteration_id": "I002",
        "recovery_type": "approved_post_snapshot_non_model_packaging_recovery",
        "teacher_approval": "user explicitly selected plan A on 2026-08-17",
        "approval_record": "docs/P3-W02_I002_训练后打包修正计划.md",
        "original_source_snapshot": "results/P3-W02_speech_vae/iterations/I002/source_snapshot.zip",
        "original_source_snapshot_sha256": sha256(ITERATION / "source_snapshot.zip"),
        "allowed_write_roots": [
            "results/P3-W02_speech_vae/iterations/I002/comparisons",
            "results/P3-W02_speech_vae/iterations/I002/recovery",
            "results/P3-W02_speech_vae/iterations/I002/branches/hifigan_vae (only previously missing delivery files)",
            "data/latents/p3_w02_hifigan_v2",
            "results/P3-W02_speech_vae/i002_example_inference.flac"
        ],
        "forbidden_changes": ["model weights", "training data", "I002 config", "formal window metrics", "formal utterance metrics", "I001 directory"],
        "delta_files": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files
        ],
        "commands": [
            ".venv/Scripts/python.exe scripts/p3_w02_i002_recover_packaging.py",
            ".venv/Scripts/python.exe examples/p3_w02_i002_inference.py",
            ".venv/Scripts/python.exe -m pytest tests/test_p3_w02_i002_recovery.py",
            ".venv/Scripts/python.exe scripts/p3_w02_i002_finalize_recovery.py"
        ]
    }
    write_json(RECOVERY / "source_delta_manifest.json", payload)


def main() -> None:
    started = time.perf_counter()
    process = psutil.Process()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    before = protected_manifest()
    write_json(RECOVERY / "protected_assets_before.json", {"schema_version": 1, "files": before})
    write_source_delta_manifest()
    i001_common = evaluate_i001_common_set(config)
    capability = create_capability_examples(config, i001_common)
    delivery = complete_model_delivery(config)
    after = protected_manifest()
    unchanged = before == after
    recovery_resource = {
        "wall_seconds": time.perf_counter() - started,
        "observed_rss_bytes": process.memory_info().rss,
    }
    write_json(RECOVERY / "protected_assets_after.json", {"schema_version": 1, "files": after})
    write_json(
        RECOVERY / "recovery_report.json",
        {
            "schema_version": 1,
            "work_package": "P3-W02",
            "iteration_id": "I002",
            "protected_assets_unchanged": unchanged,
            "i001_common_aggregate": i001_common["aggregate"],
            "capability_example_count": capability["example_count"],
            "delivery": delivery,
            "resource": recovery_resource,
            "status": "completed" if unchanged and capability["example_count"] == 8 else "failed",
        },
    )
    print(json.dumps({"protected_assets_unchanged": unchanged, "i001_common": i001_common["aggregate"], "capability_examples": capability["example_count"], "delivery": delivery, "resource": recovery_resource}, ensure_ascii=False, indent=2))
    if not unchanged or capability["example_count"] != 8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
