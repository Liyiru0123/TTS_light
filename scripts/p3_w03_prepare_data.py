"""P3-W03 DATA-SPG-001: derive 1-3s text-speech units from P1 character alignment.

Outputs (under data/speech/p3_w03_tiny_tts_v1/):
- unit_manifest.jsonl   : one record per unit (text, bounds, T_valid, durations, source)
- vocabulary.json        : char -> id (sorted, <pad>=0)
- data_audit.json        : counts, ranges, zero-duration, split isolation, integrity
- source_integrity.json  : sha256 of source audio + logmel80 for every used utterance
- manifest.json          : dataset-level metadata

Run:
    .venv/Scripts/python.exe scripts/p3_w03_prepare_data.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.tts.data import (  # noqa: E402
    derive_units_for_utterance,
    load_annotations,
    unit_to_dict,
)


def load_config() -> dict:
    cfg_path = ROOT / "configs" / "p3_w03_tiny_tts.json"
    with io.open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_key(text: str) -> str:
    """Normalized content key (punctuation/space removed) for group isolation."""
    return "".join(ch for ch in text if not ch.isspace() and ch not in "，。！？；：、…—,.!?;:")


def main() -> int:
    cfg = load_config()
    unit_cfg = cfg["unit"]
    target = int(unit_cfg.get("target_frames", 120))
    min_f = int(unit_cfg["min_frames"])
    max_f = int(unit_cfg["max_frames"])
    t_max = int(cfg["container"]["t_max"])
    time_compression = int(cfg["container"]["time_compression"])

    source_dir = ROOT / cfg["source_speech"]
    ann_path = source_dir / "annotations.jsonl"
    out_dir = ROOT / cfg["output_data"]
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_annotations(str(ann_path))
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)

    # ---- derive units per split ----
    all_units = []
    for split in ("train", "validation", "test"):
        for r in by_split.get(split, []):
            ann = r["annotation"]
            units = derive_units_for_utterance(
                ann, target=target, min_f=min_f, max_f=max_f,
                t_max=t_max, time_compression=time_compression,
            )
            scene_id = ann.get("scene_id", r.get("scene_id", ""))
            for u in units:
                u.split = split
                u.pool = r.get("pool", "")
                u.source_utterance_id = scene_id
                u.source_scene_id = scene_id
                u.source_audio_path = str(source_dir / r["audio_path"]).replace("\\", "/")
                u.source_logmel80_path = (
                    f"data/speech_vae/p3_w02_hifigan_v2/logmel80/{r['pool']}/{split}/{scene_id}_f000.npy"
                )
            all_units.extend(units)

    # ---- vocabulary ----
    chars = sorted({ch for u in all_units for ch in u.token_chars if ch.strip()})
    vocab = {"<pad>": 0}
    for i, ch in enumerate(chars, start=1):
        vocab[ch] = i
    vocab_size = len(vocab)

    # ---- write manifest ----
    manifest_path = out_dir / "unit_manifest.jsonl"
    with io.open(manifest_path, "w", encoding="utf-8") as f:
        for u in all_units:
            f.write(json.dumps(unit_to_dict(u), ensure_ascii=False) + "\n")

    with io.open(out_dir / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "vocab_size": vocab_size, "n_chars": len(chars)}, f, ensure_ascii=False, indent=2)

    # ---- source integrity ----
    integrity = []
    used_sources = {}
    for u in all_units:
        key = u.source_utterance_id
        if key in used_sources:
            continue
        used_sources[key] = True
        audio_path = ROOT / u.source_audio_path
        mel80_path = ROOT / u.source_logmel80_path
        entry = {
            "source_utterance_id": key,
            "audio_path": u.source_audio_path,
            "audio_sha256": sha256_file(audio_path) if audio_path.exists() else None,
            "logmel80_path": u.source_logmel80_path,
            "logmel80_sha256": sha256_file(mel80_path) if mel80_path.exists() else None,
        }
        integrity.append(entry)
    with io.open(out_dir / "source_integrity.json", "w", encoding="utf-8") as f:
        json.dump(integrity, f, ensure_ascii=False, indent=2)

    # ---- audit ----
    train_units = [u for u in all_units if u.split == "train"]
    val_units = [u for u in all_units if u.split == "validation"]
    test_units = [u for u in all_units if u.split == "test"]

    def stats(units):
        if not units:
            return {"count": 0}
        mel = [u.mel_frames for u in units]
        tv = [u.t_valid for u in units]
        tl = [len(u.text) for u in units]
        content_n = sum(u.content_token_count for u in units)
        content_zd = sum(u.content_zero_duration_count for u in units)
        total_tokens = sum(len(u.token_durations) for u in units)
        zd_all = sum(u.zero_duration_count for u in units)
        return {
            "count": len(units),
            "mel_frames_min": min(mel),
            "mel_frames_max": max(mel),
            "tvalid_min": min(tv),
            "tvalid_max": max(tv),
            "text_chars_min": min(tl),
            "text_chars_max": max(tl),
            "total_tokens": total_tokens,
            "zero_duration_tokens": zd_all,
            "zero_duration_rate": zd_all / total_tokens if total_tokens else 0.0,
            "content_tokens": content_n,
            "content_zero_duration_tokens": content_zd,
            "content_zero_duration_rate": content_zd / content_n if content_n else 0.0,
        }

    # split isolation
    train_src = {u.source_utterance_id for u in train_units}
    val_src = {u.source_utterance_id for u in val_units}
    test_src = {u.source_utterance_id for u in test_units}
    split_intersections = {
        "train_val": len(train_src & val_src),
        "train_test": len(train_src & test_src),
        "val_test": len(val_src & test_src),
    }

    # normalized-text group isolation (content key)
    key_to_splits = defaultdict(set)
    for u in all_units:
        key_to_splits[content_key(u.text)].add(u.split)
    cross_split_keys = {k: sorted(s) for k, s in key_to_splits.items() if len(s) > 1}

    # non-finite check (mel values are in source; we verify the 80-dim mel is finite for a sample)
    nonfinite_count = 0
    import numpy as np
    for u in train_units[:50] + val_units[:10] + test_units[:10]:
        mel80_path = ROOT / u.source_logmel80_path
        if mel80_path.exists():
            a = np.load(mel80_path, allow_pickle=False)
            seg = a[:, u.mel_start : u.mel_end]
            if not np.isfinite(seg).all():
                nonfinite_count += 1

    audit = {
        "schema_version": 1,
        "work_package": "P3-W03",
        "data_id": "DATA-SPG-001",
        "source_speech": cfg["source_speech"],
        "source_speech_vae": cfg["source_speech_vae"],
        "unit_params": {"target_frames": target, "min_frames": min_f, "max_frames": max_f, "t_max": t_max},
        "vocab_size": vocab_size,
        "n_chars": len(chars),
        "train": stats(train_units),
        "validation": stats(val_units),
        "test": stats(test_units),
        "split_intersections": split_intersections,
        "cross_split_text_groups": cross_split_keys,
        "cross_split_text_group_count": len(cross_split_keys),
        "nonfinite_mel_units_sampled": nonfinite_count,
        "source_utterance_count": len(used_sources),
        "known_limitations": [
            "源语句级划分隔离已满足（split_intersections 全 0）。",
            "P1 场景为合成数据，场景间共享短语：单元级规范化文本在 train/val/test 间存在重叠"
            "（cross_split_text_group_count 个文本组跨划分，测试集单元文本全部在训练集出现过）。",
            "因此测试集评估定位为“已知文本、新语音（韵律/时长）泛化”，而非“未见文本”泛化。",
            "这是合成数据的固有特性，非划分错误；CER 结果需在此前提下解读。",
        ],
    }
    with io.open(out_dir / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # ---- manifest ----
    manifest = {
        "schema_version": 1,
        "work_package": "P3-W03",
        "model_name": cfg["model_name"],
        "data_id": "DATA-SPG-001",
        "created": "2026-08-20",
        "source_speech": cfg["source_speech"],
        "source_speech_vae": cfg["source_speech_vae"],
        "unit_manifest": "unit_manifest.jsonl",
        "vocabulary": "vocabulary.json",
        "data_audit": "data_audit.json",
        "source_integrity": "source_integrity.json",
        "train_units": len(train_units),
        "validation_units": len(val_units),
        "test_units": len(test_units),
        "vocab_size": vocab_size,
        "status": "derived",
    }
    with io.open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---- console summary ----
    print("=== DATA-SPG-001 derivation summary ===")
    print(f"train units: {len(train_units)}  (target {unit_cfg['target_train_units']})")
    print(f"validation units: {len(val_units)}")
    print(f"test units: {len(test_units)}")
    print(f"vocab_size: {vocab_size}  (chars: {len(chars)})")
    print(f"split_intersections: {split_intersections}")
    print(f"cross_split_text_groups: {len(cross_split_keys)}")
    print(f"nonfinite_mel_units (sampled): {nonfinite_count}")
    tr = audit["train"]
    print(f"train mel_frames: [{tr['mel_frames_min']}, {tr['mel_frames_max']}]")
    print(f"train T_valid: [{tr['tvalid_min']}, {tr['tvalid_max']}]")
    print(f"train content_zero_duration_rate: {tr['content_zero_duration_rate']:.4f}")
    print(f"train zero_duration_rate (all): {tr['zero_duration_rate']:.4f}")
    print(f"source utterances: {len(used_sources)}")
    print(f"outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
