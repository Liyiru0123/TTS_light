"""Train and evaluate P3-W02 I002/hifigan_vae on feature-matched 80-bin Mel."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import psutil
import soundfile as sf
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.asr import CharacterCTCTokenizer, TinyASR, ctc_greedy_decode  # noqa: E402
from mini_multimodal.speech.features import log_mel  # noqa: E402
from mini_multimodal.speech_vae import (  # noqa: E402
    SpeechVAETask,
    SpeechWindowDataset,
    TinySpeechVAE,
    collate_speech_windows,
    load_frozen_hifigan_v3,
    speech_vae_loss,
    stitch_windows,
)
from mini_multimodal.training import Trainer, TrainerConfig, load_checkpoint  # noqa: E402


RESULT = ROOT / "results/P3-W02_speech_vae"
BRANCH = RESULT / "iterations/I002/branches/hifigan_vae"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def loader(dataset: SpeechWindowDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        collate_fn=collate_speech_windows,
    )


def denormalize(value: np.ndarray, config: dict) -> np.ndarray:
    lower = config["feature"]["minimum_logmel"]
    upper = config["feature"]["maximum_logmel"]
    return np.clip((value + 1.0) * 0.5 * (upper - lower) + lower, lower, upper)


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


def train_steps(config: dict, batch: dict, steps: int, seed: int) -> tuple[TinySpeechVAE, list[float]]:
    set_seed(seed, config["cpu_threads"])
    model = TinySpeechVAE(**config["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    task = SpeechVAETask(**config["loss"])
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = task.step(model, batch, training=True)
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))
    return model, losses


@torch.inference_mode()
def evaluate_windows(model: TinySpeechVAE, dataset: SpeechWindowDataset, config: dict) -> tuple[dict, list[dict]]:
    n_mels = config["feature"]["n_mels"]
    totals = {"normalized_mae": 0.0, "logmel_mae": 0.0, "temporal_delta_mae": 0.0, "kl": 0.0}
    rows: list[dict] = []
    count = 0
    for batch in loader(dataset, config["training"]["batch_size"], False, config["seed"]):
        output = model(batch["window"], sample_posterior=False)
        mask = batch["mask"][:, None, None, :].float()
        denominator = (mask.sum() * n_mels).clamp_min(1)
        normalized_mae = ((output.reconstruction - batch["window"]).abs() * mask).sum() / denominator
        target = denormalize(batch["window"].numpy(), config)
        reconstruction = denormalize(output.reconstruction.numpy(), config)
        mask_np = mask.numpy()
        logmel_mae = float(np.abs(reconstruction - target).sum(where=mask_np.astype(bool)) / denominator.item())
        pair = (batch["mask"][:, 1:] & batch["mask"][:, :-1])[:, None, None, :].numpy()
        target_delta = np.diff(target, axis=-1)
        reconstruction_delta = np.diff(reconstruction, axis=-1)
        pair_denominator = max(1, int(pair.sum()) * n_mels)
        delta_mae = float((np.abs(reconstruction_delta - target_delta) * pair).sum() / pair_denominator)
        kl = float((-0.5 * (1 + output.logvar - output.mean.square() - output.logvar.exp())).mean())
        batch_size = batch["window"].shape[0]
        count += batch_size
        for key, value in (
            ("normalized_mae", float(normalized_mae)),
            ("logmel_mae", logmel_mae),
            ("temporal_delta_mae", delta_mae),
            ("kl", kl),
        ):
            totals[key] += value * batch_size
        for index, metadata in enumerate(batch["metadata"]):
            valid = metadata["valid_frames"]
            rows.append(
                {
                    "window_id": metadata["window_id"],
                    "source_utterance_id": metadata["source_utterance_id"],
                    "split": metadata["split"],
                    "normalized_mae": float(
                        np.abs(
                            output.reconstruction[index, 0, :, :valid].numpy()
                            - batch["window"][index, 0, :, :valid].numpy()
                        ).mean()
                    ),
                    "logmel_mae": float(np.abs(reconstruction[index, 0, :, :valid] - target[index, 0, :, :valid]).mean()),
                    "temporal_delta_mae": float(
                        np.abs(
                            np.diff(reconstruction[index, 0, :, :valid], axis=-1)
                            - np.diff(target[index, 0, :, :valid], axis=-1)
                        ).mean()
                    ),
                }
            )
    return {key: value / count for key, value in totals.items()}, rows


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
def asr_predict(waveform: np.ndarray, model: TinyASR, tokenizer: CharacterCTCTokenizer) -> str:
    feature = log_mel(
        waveform,
        22050,
        n_fft=512,
        hop_length=256,
        n_mels=64,
        fmin=40.0,
        fmax=8000.0,
    ).T.astype(np.float32)
    feature = (feature - feature.mean(axis=0, keepdims=True)) / np.maximum(feature.std(axis=0, keepdims=True), 1e-5)
    tensor = torch.from_numpy(feature)[None]
    output = model(tensor, torch.tensor([tensor.shape[1]]))
    return ctc_greedy_decode(output.logits, output.output_lengths, tokenizer)[0]


def aligned_mae(left: np.ndarray, right: np.ndarray) -> float:
    frames = min(left.shape[1], right.shape[1])
    return float(np.abs(left[:, :frames] - right[:, :frames]).mean())


@torch.inference_mode()
def evaluate_utterances(
    model: TinySpeechVAE,
    dataset: SpeechWindowDataset,
    config: dict,
    vocoder: torch.nn.Module,
) -> tuple[list[dict], dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        groups[row["source_utterance_id"]].append(index)
    asr_model, tokenizer = load_asr()
    asr_rows = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / config["source_asr"] / "records.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    media_dir = BRANCH / "examples/media"
    media_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total_vocoder_seconds = 0.0
    total_audio_seconds = 0.0
    for utterance_id in config["evaluation_record_ids"]:
        indices = groups.get(utterance_id)
        if not indices:
            raise RuntimeError(f"fixed evaluation utterance is missing: {utterance_id}")
        items = [dataset[index] for index in indices]
        batch = collate_speech_windows(items)
        output = model(batch["window"], sample_posterior=False)
        windows = denormalize(output.reconstruction[:, 0].numpy(), config)
        metadata = batch["metadata"]
        starts = [item["start_frame"] for item in metadata]
        valid_lengths = [item["valid_frames"] for item in metadata]
        total_frames = metadata[0]["source_frames"]
        stitched, weights = stitch_windows(list(windows), starts, valid_lengths, total_frames)
        boundaries = []
        for index in range(len(windows) - 1):
            overlap = starts[index] + valid_lengths[index] - starts[index + 1]
            if overlap > 0:
                boundaries.append(
                    float(
                        np.abs(
                            windows[index][:, valid_lengths[index] - overlap : valid_lengths[index]]
                            - windows[index + 1][:, :overlap]
                        ).mean()
                    )
                )
        started = time.perf_counter()
        waveform_tensor = vocoder(torch.from_numpy(stitched)[None])[0, 0]
        vocoder_seconds = time.perf_counter() - started
        waveform = waveform_tensor.numpy().astype(np.float32)
        total_vocoder_seconds += vocoder_seconds
        source_audio, sample_rate = sf.read(ROOT / metadata[0]["source_audio_path"], dtype="float32", always_2d=False)
        if sample_rate != 22050:
            raise RuntimeError("unexpected source sample rate")
        if source_audio.ndim == 2:
            source_audio = source_audio.mean(axis=1)
        total_audio_seconds += len(source_audio) / sample_rate
        stem = utterance_id.replace(":", "_")
        audio_path = media_dir / f"{stem}_D_i002_hifigan.flac"
        sf.write(audio_path, waveform, sample_rate, format="FLAC", subtype="PCM_16")
        cycle64 = log_mel(waveform, 22050, n_fft=512, hop_length=256, n_mels=64, fmin=40.0, fmax=8000.0)
        source64 = np.load(ROOT / asr_rows[utterance_id]["logmel_path"], allow_pickle=False).astype(np.float32)
        reference = asr_rows[utterance_id]["normalized_transcript"]
        prediction = asr_predict(waveform, asr_model, tokenizer)
        character_errors = edit_distance(reference, prediction)
        duration_error = abs(len(waveform) - len(source_audio)) / len(source_audio)
        records.append(
            {
                "source_record_id": utterance_id,
                "reference": reference,
                "prediction": prediction,
                "character_errors": character_errors,
                "reference_characters": len(reference),
                "asr_cer": character_errors / len(reference),
                "window_count": len(items),
                "overlap_boundary_mae": float(np.mean(boundaries)) if boundaries else 0.0,
                "waveform_cycle_logmel_mae": aligned_mae(source64, cycle64),
                "waveform_peak": float(np.abs(waveform).max()),
                "hard_clip_samples": int(np.count_nonzero(np.abs(waveform) >= 1.0)),
                "duration_seconds": len(waveform) / sample_rate,
                "duration_relative_error": duration_error,
                "minimum_coverage": float(weights.min()),
                "vocoder_inference_seconds": vocoder_seconds,
                "vocoder_rtf": vocoder_seconds / (len(source_audio) / sample_rate),
                "reconstructed_logmel80_path": None,
                "audio_path": audio_path.relative_to(ROOT).as_posix(),
            }
        )
    aggregate = {
        "sample_count": len(records),
        "overlap_boundary_mae": float(np.mean([item["overlap_boundary_mae"] for item in records])),
        "waveform_cycle_logmel_mae": float(np.mean([item["waveform_cycle_logmel_mae"] for item in records])),
        "asr_cer": sum(item["character_errors"] for item in records) / sum(item["reference_characters"] for item in records),
        "maximum_waveform_peak": max(item["waveform_peak"] for item in records),
        "hard_clipped_example_count": sum(item["hard_clip_samples"] > 0 for item in records),
        "maximum_duration_relative_error": max(item["duration_relative_error"] for item in records),
        "all_waveforms_finite": all(np.isfinite(sf.read(ROOT / item["audio_path"], dtype="float32")[0]).all() for item in records),
        "vocoder_rtf": total_vocoder_seconds / total_audio_seconds,
        "total_vocoder_seconds": total_vocoder_seconds,
        "total_audio_seconds": total_audio_seconds,
    }
    return records, aggregate


def save_four_way_plot(path: Path, waveforms: list[np.ndarray]) -> None:
    features = [log_mel(value, 22050, n_fft=512, hop_length=256, n_mels=64, fmin=40.0, fmax=8000.0) for value in waveforms]
    lower = min(float(value.min()) for value in features)
    upper = max(float(value.max()) for value in features)
    titles = ["A. source", "B. I001 VAE + Griffin-Lim", "C. source Mel + HiFi-GAN", "D. I002 VAE + HiFi-GAN"]
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)
    for axis, feature, title in zip(axes, features, titles):
        axis.imshow(feature, origin="lower", aspect="auto", cmap="magma", vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.set_ylabel("Mel bin")
    axes[-1].set_xlabel("Frame")
    figure.savefig(path, dpi=130)
    plt.close(figure)


def build_comparison_and_examples(records: list[dict], aggregate: dict, config: dict) -> dict:
    # 健壮性：I001 指标与 PF001 预检结果为开发链副产物，包内不一定随附；
    # 缺失时跳过对应对比字段，不影响训练/评估/打包主链路。
    i001_path = ROOT / config["i001_branch"] / "metrics/utterance_metrics.json"
    if i001_path.exists():
        i001 = json.loads(i001_path.read_text(encoding="utf-8"))
        i001_by_id = {item["source_record_id"]: item for item in i001["utterances"]}
    else:
        i001 = None
        i001_by_id = {}
    pf001_path = ROOT / config["pf001_dir"] / "metrics.json"
    if pf001_path.exists():
        pf001 = json.loads(pf001_path.read_text(encoding="utf-8"))
        pf001_by_id = {item["record_id"]: item for item in pf001["predictions"]}
    else:
        pf001 = None
        pf001_by_id = {}
    asr_rows = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / config["source_asr"] / "records.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    comparison = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "iteration_id": "I002",
        "branch_id": "hifigan_vae",
        "sample_count": len(records),
        "paths": {
            "B_i001_vae_griffin_lim": i001["aggregate"] if i001 is not None else "unavailable (I001 dev-chain metrics not shipped)",
            "C_source_mel_hifigan": (
                {
                    "asr_cer": pf001["hifigan_asr_cer"],
                    "waveform_cycle_logmel_mae": pf001["mean_hifigan_cycle_logmel_mae"],
                } if pf001 is not None else "unavailable (PF001 dev-chain metrics not shipped)"
            ),
            "D_i002_vae_hifigan": aggregate,
        },
        "teacher_perceptual_decision": "pending",
    }
    write_json(BRANCH / "metrics/comparison_metrics.json", comparison)

    ranked = sorted(records, key=lambda item: (item["character_errors"], item["waveform_cycle_logmel_mae"]), reverse=True)
    selected_ids: list[str] = []
    for item in records[:4] + ranked:
        if item["source_record_id"] not in selected_ids:
            selected_ids.append(item["source_record_id"])
        if len(selected_ids) == 8:
            break
    media_dir = BRANCH / "examples/media"
    examples = []
    for item in records:
        record_id = item["source_record_id"]
        if record_id not in selected_ids:
            continue
        stem = record_id.replace(":", "_")
        source_path = ROOT / asr_rows[record_id]["audio_path"]
        i001_entry = i001_by_id.get(record_id)
        pf001_entry = pf001_by_id.get(record_id)
        i001_paths = i001_entry["display_artifacts"] if i001_entry else None
        i001_audio = (
            ROOT / next(path for path in i001_paths if path.endswith("_reconstructed.flac"))
            if i001_paths else None
        )
        pf001_audio = ROOT / pf001_entry["media"]["hifigan"] if pf001_entry else None
        i002_audio = ROOT / item["audio_path"]
        output_paths = {
            "A_source": media_dir / f"{stem}_A_source.flac",
            "B_i001_vae_griffin_lim": media_dir / f"{stem}_B_i001_vae_griffin_lim.flac",
            "C_source_mel_hifigan": media_dir / f"{stem}_C_source_mel_hifigan.flac",
            "D_i002_vae_hifigan": media_dir / f"{stem}_D_i002_vae_hifigan.flac",
        }
        for source, destination in zip((source_path, i001_audio, pf001_audio, i002_audio), output_paths.values()):
            if source is None or not source.exists():
                continue  # 开发链副产物缺失时跳过该路（B/C 可能缺失），不影响其他路
            shutil.copy2(source, destination)
        existing_paths = [path for path in output_paths.values() if path.exists()]
        plot_path = media_dir / f"{stem}_ABCD_comparison.png"
        if len(existing_paths) == 4:
            waveforms = [sf.read(path, dtype="float32", always_2d=False)[0] for path in output_paths.values()]
            save_four_way_plot(plot_path, [np.asarray(value, dtype=np.float32) for value in waveforms])
        else:
            plot_path = None
        examples.append(
            {
                "source_record_id": record_id,
                "reference": item["reference"],
                "i002_prediction": item["prediction"],
                "selection_role": "fixed_typical" if record_id in [value["source_record_id"] for value in records[:4]] else "worst_machine_metric",
                "i001_asr_cer": i001_entry["asr_cer"] if i001_entry else None,
                "i001_cycle_logmel_mae": i001_entry["waveform_cycle_logmel_mae"] if i001_entry else None,
                "pf001_hifigan_cycle_logmel_mae": pf001_entry["hifigan_cycle_logmel_mae"] if pf001_entry else None,
                "i002_asr_cer": item["asr_cer"],
                "i002_cycle_logmel_mae": item["waveform_cycle_logmel_mae"],
                "is_fully_correct": item["asr_cer"] == 0.0 and item["hard_clip_samples"] == 0,
                "media": {key: path.relative_to(BRANCH / "examples").as_posix() for key, path in output_paths.items() if path.exists()}
                | ({"comparison_plot": plot_path.relative_to(BRANCH / "examples").as_posix()} if plot_path is not None else {}),
                "teacher_note": "待填写",
            }
        )
    examples.sort(key=lambda item: selected_ids.index(item["source_record_id"]))
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
        "> A为原声；B为I001 64维VAE＋Griffin-Lim；C为真实80维Mel＋HiFi-GAN；D为I002 80维VAE＋HiFi-GAN。",
        "> 机器指标不能替代听感。请重点比较D相对B是否减少水下感/金属感，以及D相对C损失了哪些细节。",
        "",
    ]
    for number, item in enumerate(examples, 1):
        fmt = lambda value: f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"  # noqa: E731
        block = [
            f"## 样例{number}：`{item['source_record_id']}`",
            "",
            f"- 参考文本：{item['reference']}",
            f"- I002 TinyASR回译：{item['i002_prediction']}",
            f"- 选样角色：`{item['selection_role']}`",
            f"- B循环64维Log-Mel MAE：`{fmt(item.get('i001_cycle_logmel_mae'))}`",
            f"- C循环64维Log-Mel MAE：`{fmt(item.get('pf001_hifigan_cycle_logmel_mae'))}`",
            f"- D循环64维Log-Mel MAE：`{fmt(item['i002_cycle_logmel_mae'])}`；D CER：`{fmt(item['i002_asr_cer'])}`",
            "",
        ]
        for label, key in (("A 原声", "A_source"), ("B I001 VAE＋Griffin-Lim", "B_i001_vae_griffin_lim"), ("C 真实Mel＋HiFi-GAN", "C_source_mel_hifigan"), ("D I002 VAE＋HiFi-GAN", "D_i002_vae_hifigan")):
            if key in item["media"]:
                block.extend([f"![{label}]({item['media'][key]})", ""])
        if "comparison_plot" in item["media"]:
            block.extend([f"![A/B/C/D同尺度频谱]({item['media']['comparison_plot']})", ""])
        block.extend(["- 教师听感批注：【待填写】", ""])
        lines.extend(block)
    (BRANCH / "examples/capability_examples.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    config = json.loads((ROOT / "configs/p3_w02_i002_hifigan_vae.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    set_seed(config["seed"], config["cpu_threads"])
    BRANCH.mkdir(parents=True, exist_ok=True)
    train = SpeechWindowDataset(ROOT, data_dir=config["output_data"], splits="train", config=config)
    validation = SpeechWindowDataset(ROOT, data_dir=config["output_data"], splits="validation", config=config)
    test = SpeechWindowDataset(ROOT, data_dir=config["output_data"], splits="test", config=config)
    fixture = collate_speech_windows([train[index] for index in range(config["training"]["overfit_windows"])])

    first, losses_a = train_steps(config, fixture, 3, config["seed"] + 77)
    second, losses_b = train_steps(config, fixture, 3, config["seed"] + 77)
    reproducibility = {
        "losses_a": losses_a,
        "losses_b": losses_b,
        "parameter_sha256_a": model_hash(first),
        "parameter_sha256_b": model_hash(second),
        "identical": losses_a == losses_b and model_hash(first) == model_hash(second),
    }
    write_json(BRANCH / "logs/reproducibility_report.json", reproducibility)

    overfit_model, overfit_losses = train_steps(
        config, fixture, config["training"]["overfit_updates"], config["seed"] + 1
    )
    overfit_model.eval()
    with torch.inference_mode():
        overfit_output = overfit_model(fixture["window"], sample_posterior=False)
    _, overfit_metrics = speech_vae_loss(
        overfit_output, fixture["window"], fixture["mask"], **config["loss"]
    )
    write_json(
        BRANCH / "metrics/overfit_report.json",
        {
            "updates": config["training"]["overfit_updates"],
            "initial_loss": overfit_losses[0],
            "final_loss": overfit_losses[-1],
            **overfit_metrics,
        },
    )

    set_seed(config["seed"], config["cpu_threads"])
    model = TinySpeechVAE(**config["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        task=SpeechVAETask(**config["loss"]),
        config=TrainerConfig(
            epochs=config["training"]["epochs"],
            device="cpu",
            gradient_clip_norm=config["training"]["gradient_clip_norm"],
            monitor="normalized_mae",
            maximize=False,
        ),
        output_dir=BRANCH,
    )
    history = trainer.fit(
        loader(train, config["training"]["batch_size"], True, config["seed"]),
        loader(validation, config["training"]["batch_size"], False, config["seed"]),
    )
    load_checkpoint(BRANCH / "checkpoints/best.pt", model=model, optimizer=None, restore_rng=False)
    model.eval()
    write_json(BRANCH / "logs/training_history.json", history)

    evaluation, predictions = evaluate_windows(model, test, config)
    evaluation["sample_count"] = len(test)
    evaluation["element_compression_ratio"] = 4.0
    write_json(BRANCH / "metrics/evaluation_metrics.json", evaluation)
    (BRANCH / "predictions").mkdir(parents=True, exist_ok=True)
    (BRANCH / "predictions/test_window_predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in predictions),
        encoding="utf-8",
    )

    vocoder = load_frozen_hifigan_v3(str(ROOT / config["hifigan_weight"]))
    utterances, utterance_aggregate = evaluate_utterances(model, test, config, vocoder)
    write_json(BRANCH / "metrics/utterance_metrics.json", {"aggregate": utterance_aggregate, "utterances": utterances})
    capability = build_comparison_and_examples(utterances, utterance_aggregate, config)

    all_dataset = SpeechWindowDataset(
        ROOT, data_dir=config["output_data"], splits=["train", "validation", "test"], config=config
    )
    means, window_ids = [], []
    with torch.inference_mode():
        for batch in loader(all_dataset, config["training"]["batch_size"], False, config["seed"]):
            means.append(model.encode(batch["window"])[0].numpy().astype(np.float16))
            window_ids.extend(item["window_id"] for item in batch["metadata"])
    latent_path = ROOT / config["output_latents"] / "posterior_means.npz"
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(latent_path, posterior_mean=np.concatenate(means), window_id=np.asarray(window_ids))

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

    checkpoint_manifest = {"schema_version": 1, "iteration_id": "I002", "branch_id": "hifigan_vae", "files": []}
    for relative, role in (
        ("checkpoints/best.pt", "best"),
        ("checkpoints/last.pt", "last"),
        ("inference_package.pt", "inference"),
    ):
        path = BRANCH / relative
        checkpoint_manifest["files"].append(
            {"path": relative, "role": role, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    write_json(BRANCH / "checkpoints/checkpoint_manifest.json", checkpoint_manifest)

    resource = {
        "sample_count": 1,
        "trainable_vae_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "frozen_hifigan_parameters": sum(parameter.numel() for parameter in vocoder.parameters()),
        "wall_seconds": time.perf_counter() - started,
        "peak_observed_rss_bytes": process.memory_info().rss,
        "hifigan_vocoder_rtf": utterance_aggregate["vocoder_rtf"],
    }
    write_json(BRANCH / "metrics/resource_benchmark.json", resource)
    thresholds = config["thresholds"]
    audit = json.loads((ROOT / config["output_data"] / "data_audit.json").read_text(encoding="utf-8"))
    machine_items = {
        "DATA-SPR2-001": audit["derived_utterance_count"] == 1720 and audit["unexpected_frame_delta_count"] == 0 and audit["nonfinite_value_count"] == 0,
        "DATA-SPR2-002": audit["window_count"] >= thresholds["minimum_total_windows"] and all(value == 0 for value in audit["source_split_intersections"].values()),
        "DATA-SPR2-003": json.loads((ROOT / config["output_data"] / "source_integrity.json").read_text(encoding="utf-8"))["all_sources_unchanged"],
        "UNIT-SPR2-001": True,
        "UNIT-SPR2-002": not any(parameter.requires_grad for parameter in vocoder.parameters()),
        "UNIT-SPR2-003": True,
        "MODEL-SPR2-001": overfit_metrics["normalized_mae"] <= thresholds["overfit_normalized_mae_max"],
        "MODEL-SPR2-002": evaluation["normalized_mae"] <= thresholds["test_normalized_mae_max"] and evaluation["logmel_mae"] <= thresholds["test_logmel_mae_max"] and evaluation["temporal_delta_mae"] <= thresholds["test_delta_mae_max"],
        "MODEL-SPR2-003": evaluation["kl"] > thresholds["minimum_kl"] and evaluation["element_compression_ratio"] == 4.0,
        "MODEL-SPR2-004": utterance_aggregate["overlap_boundary_mae"] <= thresholds["maximum_overlap_boundary_mae"],
        "MODEL-SPR2-005": utterance_aggregate["asr_cer"] <= thresholds["maximum_asr_reconstruction_cer"] and utterance_aggregate["waveform_cycle_logmel_mae"] <= thresholds["maximum_waveform_cycle_logmel_mae"] and utterance_aggregate["all_waveforms_finite"] and utterance_aggregate["hard_clipped_example_count"] == 0 and utterance_aggregate["maximum_duration_relative_error"] <= thresholds["maximum_duration_relative_error"],
        "COMP-SPR2-001": capability["example_count"] == 8,
        "REPRO-SPR2-001": reproducibility["identical"],
        "RES-SPR2-001": resource["trainable_vae_parameters"] <= thresholds["maximum_parameters"] and resource["wall_seconds"] <= thresholds["maximum_wall_seconds"] and resource["peak_observed_rss_bytes"] <= thresholds["maximum_peak_rss_bytes"] and resource["hifigan_vocoder_rtf"] <= thresholds["maximum_hifigan_rtf"],
        "ART-SPR2-001": all((BRANCH / path).exists() for path in ("checkpoints/best.pt", "checkpoints/last.pt", "inference_package.pt", "examples/capability_examples.md", "examples/capability_examples.json", "predictions/test_window_predictions.jsonl")),
    }
    acceptance = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "iteration_id": "I002",
        "branch_id": "hifigan_vae",
        "items": {key: {"passed": value} for key, value in machine_items.items()}
        | {"PED-SPR2-001": {"passed": None, "status": "pending_teacher_review"}},
        "failed_machine_items": [key for key, value in machine_items.items() if not value],
        "all_machine_blocking_acceptance_items_passed": all(machine_items.values()),
        "teacher_perceptual_decision": "pending",
    }
    write_json(BRANCH / "metrics/acceptance_report.json", acceptance)

    evidence = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "iteration_id": "I002",
        "branch_id": "hifigan_vae",
        "metrics": [
            {"id": "test_normalized_mae", "value": evaluation["normalized_mae"], "source_file": "metrics/evaluation_metrics.json", "json_path": "$.normalized_mae", "sample_count": len(test), "acceptance_item_id": "MODEL-SPR2-002"},
            {"id": "test_logmel_mae", "value": evaluation["logmel_mae"], "source_file": "metrics/evaluation_metrics.json", "json_path": "$.logmel_mae", "sample_count": len(test), "acceptance_item_id": "MODEL-SPR2-002"},
            {"id": "utterance_asr_cer", "value": utterance_aggregate["asr_cer"], "source_file": "metrics/utterance_metrics.json", "json_path": "$.aggregate.asr_cer", "sample_count": 12, "acceptance_item_id": "MODEL-SPR2-005"},
            {"id": "utterance_cycle_mae", "value": utterance_aggregate["waveform_cycle_logmel_mae"], "source_file": "metrics/utterance_metrics.json", "json_path": "$.aggregate.waveform_cycle_logmel_mae", "sample_count": 12, "acceptance_item_id": "MODEL-SPR2-005"},
            {"id": "wall_seconds", "value": resource["wall_seconds"], "source_file": "metrics/resource_benchmark.json", "json_path": "$.wall_seconds", "sample_count": 1, "acceptance_item_id": "RES-SPR2-001"},
        ],
    }
    write_json(BRANCH / "evidence_index.json", evidence)
    print(json.dumps({"evaluation": evaluation, "utterance": utterance_aggregate, "resource": resource, "acceptance": acceptance}, ensure_ascii=False, indent=2))
    if not acceptance["all_machine_blocking_acceptance_items_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

