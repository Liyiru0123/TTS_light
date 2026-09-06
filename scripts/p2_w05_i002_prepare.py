"""Build an isolated multi-condition training view for P2-W05 I002."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from mini_multimodal.asr import add_white_noise_at_snr, stable_seed
from mini_multimodal.speech.features import log_mel


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_record(base: dict, *, record_id: str, output_path: Path, relative_path: str, snr_db: float, seed: int, view: str) -> dict:
    waveform, sample_rate = sf.read(ROOT / base["audio_path"], dtype="float32")
    mixed, measured = add_white_noise_at_snr(waveform, snr_db, seed=seed)
    feature = log_mel(mixed, sample_rate).astype(np.float16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feature, allow_pickle=False)
    result = dict(base)
    result.update({
        "record_id": record_id, "source_record_id": base["record_id"], "logmel_path": relative_path,
        "num_frames": int(feature.shape[1]), "i002_view": view, "noise_snr_db_requested": float(snr_db),
        "noise_snr_db_measured": measured, "noise_seed": int(seed), "logmel_sha256": sha(output_path)
    })
    return result


def main() -> None:
    config_path = ROOT / "configs/p2_w05_i002.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_dir = ROOT / config["source_asr_data"]
    output_dir = ROOT / config["output_data"]
    source_rows = [json.loads(line) for line in (source_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    source_by_id = {row["record_id"]: row for row in source_rows}
    source_splits = json.loads((source_dir / "splits.json").read_text(encoding="utf-8"))
    output_rows = list(source_rows)
    train_ids = []
    requested, measured = [], []
    snr_min, snr_max = config["augmentation"]["noise_snr_db_min"], config["augmentation"]["noise_snr_db_max"]
    for index, source_id in enumerate(source_splits["regular"]["train"]):
        base = source_by_id[source_id]
        clean = dict(base); clean["record_id"] = source_id + "::i002_clean"; clean["source_record_id"] = source_id; clean["i002_view"] = "clean_train"
        output_rows.append(clean); train_ids.append(clean["record_id"])
        rng = np.random.default_rng(stable_seed(config["seed"], source_id, "train_noise"))
        snr = float(rng.uniform(snr_min, snr_max)); noise_seed = stable_seed(config["seed"], source_id, "waveform_noise")
        filename = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24] + ".npy"
        relative = f"{config['output_data']}/logmel/train_noise/{filename}"
        noisy = noise_record(base, record_id=source_id + "::i002_noise", output_path=ROOT / relative, relative_path=relative, snr_db=snr, seed=noise_seed, view="noise_train")
        output_rows.append(noisy); train_ids.append(noisy["record_id"]); requested.append(snr); measured.append(noisy["noise_snr_db_measured"])
        if (index + 1) % 200 == 0: print(f"prepared_train_noise={index+1}", flush=True)
    validation_source_ids = source_splits["text_nonoverlap"]["validation"][:60]
    validation_noise_ids = []
    for index, source_id in enumerate(validation_source_ids):
        base = source_by_id[source_id]; noise_seed = stable_seed(config["seed"], source_id, "validation_noise_15db")
        filename = hashlib.sha256((source_id + "validation").encode("utf-8")).hexdigest()[:24] + ".npy"
        relative = f"{config['output_data']}/logmel/validation_noise_15db/{filename}"
        noisy = noise_record(base, record_id=source_id + "::i002_noise15_validation", output_path=ROOT / relative, relative_path=relative, snr_db=15.0, seed=noise_seed, view="noise15_validation")
        output_rows.append(noisy); validation_noise_ids.append(noisy["record_id"]); requested.append(15.0); measured.append(noisy["noise_snr_db_measured"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in output_rows), encoding="utf-8", newline="\n")
    vocabulary_bytes = (source_dir / "vocabulary.json").read_bytes(); (output_dir / "vocabulary.json").write_bytes(vocabulary_bytes)
    splits = {"train_multistyle": train_ids, "validation_noise_15db": validation_noise_ids, "source_views": source_splits}
    write_json(output_dir / "splits.json", splits)
    source_files = {name: {"bytes": (source_dir / name).stat().st_size, "sha256": sha(source_dir / name)} for name in ("manifest.json", "records.jsonl", "splits.json", "vocabulary.json")}
    audit = {
        "schema_version": 1, "source_train_count": len(source_splits["regular"]["train"]), "multistyle_train_count": len(train_ids),
        "clean_train_count": sum(item.endswith("::i002_clean") for item in train_ids), "noise_train_count": sum(item.endswith("::i002_noise") for item in train_ids),
        "validation_noise_count": len(validation_noise_ids), "unique_train_source_count": len({row.get("source_record_id") for row in output_rows if row.get("record_id") in set(train_ids)}),
        "requested_snr_min": min(requested), "requested_snr_max": max(requested), "max_snr_measurement_error_db": max(abs(a-b) for a,b in zip(requested, measured)),
        "source_files_before_and_after": source_files, "source_data_modified": False
    }
    write_json(output_dir / "data_audit.json", audit)
    files = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.name != "manifest.json"):
        files[path.relative_to(output_dir).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    write_json(output_dir / "manifest.json", {"schema_version": 1, "dataset_id": "p2_w05_i002_v1", "iteration_id": "I002", "source_data": config["source_asr_data"], "source_data_modified": False, "statistics": audit, "files": files})


if __name__ == "__main__":
    main()
