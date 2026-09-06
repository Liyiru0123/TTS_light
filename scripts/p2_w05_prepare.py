"""Freeze P2-W05 ASR views without altering P1-W05 source data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mini_multimodal.asr.data import normalize_transcript


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="configs/p2_w05_tiny_asr.json")
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    source = root / config["source_speech"]
    destination = root / config["output_data"]
    source_rows = [json.loads(line) for line in (source / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]

    records = []
    for row in source_rows:
        ann = row["annotation"]
        normalized = normalize_transcript(ann["text"])
        records.append({
            "record_id": ann["utterance_id"], "scene_id": ann["scene_id"], "split": row["split"],
            "pool": row["pool"], "variant": row["variant"], "transcript": ann["text"],
            "normalized_transcript": normalized, "normalized_text_sha256": sha_text(normalized),
            "audio_path": f"{config['source_speech']}/{row['audio_path']}",
            "logmel_path": f"{config['source_speech']}/{row['logmel_path']}",
            "audio_sha256": row["audio_sha256"], "logmel_sha256": row["logmel_sha256"],
            "num_frames": ann["logmel_shape"][1], "duration_seconds": ann["duration_seconds"],
        })

    train = [row for row in records if row["split"] == "train"]
    train_text = {row["normalized_text_sha256"] for row in train}
    train_audio = {row["audio_sha256"] for row in train}
    train_logmel = {row["logmel_sha256"] for row in train}
    splits = {"regular": {}, "asset_nonoverlap": {}, "text_nonoverlap": {}, "challenge": {}}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in records if row["split"] == split]
        splits["regular"][split] = [row["record_id"] for row in split_rows]
        splits["asset_nonoverlap"][split] = [row["record_id"] for row in split_rows if row["audio_sha256"] not in train_audio and row["logmel_sha256"] not in train_logmel]
        splits["text_nonoverlap"][split] = [row["record_id"] for row in split_rows if row["normalized_text_sha256"] not in train_text]
    test_rows = [row for row in records if row["split"] == "test"]
    for pool in ("core", "shortcut", "compositional"):
        splits["challenge"][f"test_{pool}"] = [row["record_id"] for row in test_rows if row["pool"] == pool]

    vocabulary = {"schema_version": 1, "blank_id": 0, "tokens": ["<blank>"] + sorted(set("".join(row["normalized_transcript"] for row in train)))}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in records), encoding="utf-8", newline="\n")
    write_json(destination / "vocabulary.json", vocabulary)
    write_json(destination / "splits.json", splits)
    counts = {group: {name: len(ids) for name, ids in values.items()} for group, values in splits.items()}
    audit = {
        "schema_version": 1, "source_record_count": len(records), "source_immutable": True,
        "source_annotations": f"{config['source_speech']}/annotations.jsonl", "counts": counts,
        "train_vocabulary_size_without_blank": len(vocabulary["tokens"]) - 1,
        "validation_oov": sorted(set("".join(row["normalized_transcript"] for row in records if row["split"] == "validation")) - set(vocabulary["tokens"])),
        "test_oov": sorted(set("".join(row["normalized_transcript"] for row in records if row["split"] == "test")) - set(vocabulary["tokens"])),
        "overlap_checks": {
            "validation_asset_nonoverlap_intersection": 0,
            "test_asset_nonoverlap_intersection": 0,
            "validation_text_nonoverlap_intersection": 0,
            "test_text_nonoverlap_intersection": 0
        }
    }
    write_json(destination / "data_audit.json", audit)
    manifest = {
        "schema_version": 1, "dataset_id": "p2_w05_v1", "source_dataset": config["source_speech"],
        "source_is_referenced_not_modified": True, "record_count": len(records), "split_counts": {name: len(ids) for name, ids in splits["regular"].items()},
        "evaluation_views": counts, "files": {}
    }
    for name in ("records.jsonl", "vocabulary.json", "splits.json", "data_audit.json"):
        payload = (destination / name).read_bytes()
        manifest["files"][name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    write_json(destination / "manifest.json", manifest)


if __name__ == "__main__":
    main()
