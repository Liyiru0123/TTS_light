"""Derive feature-matched 80-bin HiFi-GAN Mel data without changing P1."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.speech_vae import HiFiGANMelTransform  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    config = json.loads((ROOT / "configs/p3_w02_i002_hifigan_vae.json").read_text(encoding="utf-8"))
    source_dir = ROOT / config["source_speech"]
    output_dir = ROOT / config["output_data"]
    rows = [json.loads(line) for line in (source_dir / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    torch.set_num_threads(config["cpu_threads"])
    transform = HiFiGANMelTransform(
        sample_rate=config["feature"]["sample_rate"],
        n_fft=config["feature"]["n_fft"],
        win_length=config["feature"]["win_length"],
        hop_length=config["feature"]["hop_length"],
        n_mels=config["feature"]["n_mels"],
        f_min=config["feature"]["fmin"],
        f_max=config["feature"]["fmax"],
        log_floor=config["feature"]["log_floor"],
    ).eval()

    derived_rows: list[dict] = []
    windows: list[dict] = []
    source_hashes_before: dict[str, str] = {}
    split_sources = {name: set() for name in ("train", "validation", "test")}
    coverage_failures: list[str] = []
    frame_deltas: Counter[int] = Counter()
    unexpected_frame_deltas: list[str] = []
    nonfinite_count = 0
    below_floor_count = 0
    above_ceiling_count = 0
    value_count = 0
    global_min = float("inf")
    global_max = float("-inf")
    split_window_counts: Counter[str] = Counter()

    window_frames = config["window"]["frames"]
    hop_frames = config["window"]["hop_frames"]
    minimum_valid = config["window"]["minimum_valid_frames"]
    lower = config["feature"]["minimum_logmel"]
    upper = config["feature"]["maximum_logmel"]

    for row_index, row in enumerate(rows):
        utterance_id = row["annotation"]["utterance_id"]
        source_audio = source_dir / row["audio_path"]
        source_hash = sha256(source_audio)
        source_hashes_before[utterance_id] = source_hash
        if source_hash != row["audio_sha256"]:
            raise RuntimeError(f"P1 source hash mismatch: {utterance_id}")
        waveform, sample_rate = sf.read(source_audio, dtype="float32", always_2d=False)
        if sample_rate != config["feature"]["sample_rate"]:
            raise RuntimeError(f"sample-rate mismatch: {utterance_id}")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        with torch.inference_mode():
            feature = transform(torch.from_numpy(np.asarray(waveform, dtype=np.float32))[None])[0].numpy().astype(np.float32)
        if feature.shape[0] != config["feature"]["n_mels"]:
            raise RuntimeError(f"Mel dimension mismatch: {utterance_id} -> {feature.shape}")
        frame_delta = feature.shape[1] - row["annotation"]["logmel_shape"][1]
        frame_deltas[frame_delta] += 1
        if frame_delta != 1:
            unexpected_frame_deltas.append(utterance_id)
        nonfinite_count += int(feature.size - np.isfinite(feature).sum())
        below_floor_count += int(np.count_nonzero(feature < lower - 1e-6))
        above_ceiling_count += int(np.count_nonzero(feature > upper))
        value_count += feature.size
        global_min = min(global_min, float(feature.min()))
        global_max = max(global_max, float(feature.max()))

        relative_feature = Path("logmel80") / row["pool"] / row["split"] / f"{Path(row['audio_path']).stem}.npy"
        feature_path = output_dir / relative_feature
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, feature, allow_pickle=False)
        feature_hash = sha256(feature_path)
        source_project_path = (Path(config["source_speech"]) / row["audio_path"]).as_posix()
        feature_project_path = (Path(config["output_data"]) / relative_feature).as_posix()
        derived_rows.append(
            {
                "source_utterance_id": utterance_id,
                "scene_id": row["annotation"]["scene_id"],
                "split": row["split"],
                "pool": row["pool"],
                "variant": row["variant"],
                "text": row["annotation"]["text"],
                "source_audio_path": source_project_path,
                "source_audio_sha256": source_hash,
                "source_logmel_path": feature_project_path,
                "source_logmel_sha256": feature_hash,
                "logmel_shape": list(feature.shape),
            }
        )

        frames = feature.shape[1]
        starts = list(range(0, max(1, frames - minimum_valid + 1), hop_frames))
        if not starts or starts[-1] + window_frames < frames:
            starts.append(max(0, frames - window_frames))
        starts = sorted(set(starts))
        coverage = np.zeros(frames, dtype=np.int16)
        for window_index, start in enumerate(starts):
            valid = min(window_frames, frames - start)
            if valid < minimum_valid:
                continue
            coverage[start : start + valid] += 1
            split_window_counts[row["split"]] += 1
            windows.append(
                {
                    "window_id": f"{hashlib.sha256(utterance_id.encode()).hexdigest()[:16]}_w{window_index:02d}",
                    "source_utterance_id": utterance_id,
                    "scene_id": row["annotation"]["scene_id"],
                    "split": row["split"],
                    "pool": row["pool"],
                    "variant": row["variant"],
                    "source_logmel_path": feature_project_path,
                    "source_audio_path": source_project_path,
                    "text": row["annotation"]["text"],
                    "start_frame": start,
                    "valid_frames": valid,
                    "source_frames": frames,
                }
            )
        if not np.all(coverage > 0):
            coverage_failures.append(utterance_id)
        split_sources[row["split"]].add(utterance_id)
        if (row_index + 1) % 200 == 0:
            print(f"derived={row_index + 1}/{len(rows)}", flush=True)

    source_hashes_after = {
        row["annotation"]["utterance_id"]: sha256(source_dir / row["audio_path"]) for row in rows
    }
    integrity = {
        "schema_version": 1,
        "source_count": len(rows),
        "all_sources_unchanged": source_hashes_before == source_hashes_after,
        "p1_directory": config["source_speech"],
        "derived_directory": config["output_data"],
    }
    intersections = {
        "train_validation": len(split_sources["train"] & split_sources["validation"]),
        "train_test": len(split_sources["train"] & split_sources["test"]),
        "validation_test": len(split_sources["validation"] & split_sources["test"]),
    }
    audit = {
        "schema_version": 1,
        "work_package": "P3-W02",
        "iteration_target": "I002",
        "dataset_id": "p3_w02_hifigan_v2",
        "source_utterance_count": len(rows),
        "derived_utterance_count": len(derived_rows),
        "window_count": len(windows),
        "split_window_counts": dict(split_window_counts),
        "split_source_counts": {key: len(value) for key, value in split_sources.items()},
        "source_split_intersections": intersections,
        "coverage_failure_count": len(coverage_failures),
        "expected_frame_delta_from_p1": 1,
        "frame_delta_counts": {str(key): value for key, value in sorted(frame_deltas.items())},
        "unexpected_frame_delta_count": len(unexpected_frame_deltas),
        "nonfinite_value_count": nonfinite_count,
        "below_configured_floor_count": below_floor_count,
        "above_configured_ceiling_count": above_ceiling_count,
        "above_configured_ceiling_fraction": above_ceiling_count / value_count,
        "observed_minimum_logmel": global_min,
        "observed_maximum_logmel": global_max,
        "configured_minimum_logmel": lower,
        "configured_maximum_logmel": upper,
        "minimum_valid_frames": min(item["valid_frames"] for item in windows),
        "maximum_valid_frames": max(item["valid_frames"] for item in windows),
        "window_frames": window_frames,
        "hop_frames": hop_frames,
        "feature_config": config["feature"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "utterance_manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in derived_rows),
        encoding="utf-8",
    )
    (output_dir / "window_manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in windows),
        encoding="utf-8",
    )
    write_json(output_dir / "source_integrity.json", integrity)
    write_json(output_dir / "data_audit.json", audit)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "dataset_id": "p3_w02_hifigan_v2",
            "source_dir": config["source_speech"],
            "utterance_manifest": "utterance_manifest.jsonl",
            "window_manifest": "window_manifest.jsonl",
            "feature_config": config["feature"],
            "statistics": audit,
        },
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
