"""Finalize objective statistics and create balanced blind A/B samples."""

from __future__ import annotations

import json
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mini_multimodal.tts.cfm import scale_durations, text_to_token_ids  # noqa: E402
from mini_multimodal.tts.frozen_chain import load_frozen_chain  # noqa: E402
from p3_w03_compare_i001_i004 import (  # noqa: E402
    KEYWORD_CHARS,
    bootstrap,
    model_from_checkpoint,
    read_json,
    synthesize,
    write_json,
)


CONFIG = ROOT / "configs/p3_w03_i001_i004_comparison.json"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_wav(path: Path, values: np.ndarray, sample_rate: int = 22050) -> None:
    pcm = (np.clip(values, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def paired_vector(rows: list[dict], track: str, factors: tuple[float, ...], *, relative: bool) -> np.ndarray:
    index = {(row["unit_id"], row["track"], row["factor"], row["model_id"]): row for row in rows if not row["overflow"]}
    values = []
    for unit_id in sorted({row["unit_id"] for row in rows}):
        unit_values = []
        for factor in factors:
            left = index[(unit_id, track, factor, "I001")]["mel_mae"]
            right = index[(unit_id, track, factor, "I004")]["mel_mae"]
            difference = right - left
            unit_values.append(difference / max(left, 1e-9) if relative else difference)
        values.append(float(np.mean(unit_values)))
    return np.asarray(values, dtype=np.float64)


def token_analysis(rows: list[dict]) -> dict:
    index = {(row["unit_id"], row["track"], row["factor"], row["model_id"]): row for row in rows if not row["overflow"]}
    buckets = defaultdict(list)
    for unit_id in sorted({row["unit_id"] for row in rows}):
        for track in ("controlled_duration", "end_to_end"):
            for factor in (0.8, 1.0, 1.2):
                left = index[(unit_id, track, factor, "I001")]
                right = index[(unit_id, track, factor, "I004")]
                for char, left_value, right_value in zip(left["token_chars"], left["token_mel_mae"], right["token_mel_mae"]):
                    buckets[(char, track, factor)].append(right_value - left_value)
    rows_out = []
    for (char, track, factor), values in buckets.items():
        if char in KEYWORD_CHARS:
            rows_out.append({"character": char, "track": track, "factor": factor, "n": len(values), "mean_difference_i004_minus_i001": float(np.mean(values))})
    return {"semantic_character_groups": sorted(rows_out, key=lambda row: (row["track"], row["factor"], row["character"]))}


def choose_blind_cases(rows: list[dict]) -> list[tuple[str, float]]:
    index = {(row["unit_id"], row["factor"], row["model_id"]): row for row in rows if row["track"] == "end_to_end" and not row["overflow"]}
    cases = []
    for factor in (0.8, 1.0, 1.2):
        scored = []
        for unit_id in sorted({key[0] for key in index if key[1] == factor}):
            left, right = index[(unit_id, factor, "I001")], index[(unit_id, factor, "I004")]
            if any(char in KEYWORD_CHARS for char in left["text"]):
                scored.append((right["mel_mae"] - left["mel_mae"], unit_id))
        scored.sort()
        positions = [round((len(scored) - 1) * quantile) for quantile in (0.1, 0.4, 0.6, 0.9)]
        cases.extend((scored[position][1], factor) for position in positions)
    return cases


def main() -> int:
    cfg = read_json(CONFIG)
    root = ROOT / cfg["output_results"]
    formal = root / "formal"
    rows = load_rows(formal / "predictions/paired_predictions.jsonl")
    aggregate = read_json(formal / "metrics/aggregate_metrics.json")
    calibration = read_json(formal / "metrics/asr_calibration.json")
    samples = read_json(formal / "sample_manifest.json")
    bootstrap_seed = int(cfg["bootstrap_seed"])
    bootstrap_samples = int(cfg["bootstrap_samples"])
    normal = bootstrap(paired_vector(rows, "controlled_duration", (1.0,), relative=True), bootstrap_seed + 10, bootstrap_samples)
    scaled = bootstrap(paired_vector(rows, "controlled_duration", (0.8, 1.2), relative=False), bootstrap_seed + 20, bootstrap_samples)
    end_to_end = bootstrap(paired_vector(rows, "end_to_end", (0.8, 1.0, 1.2), relative=True), bootstrap_seed + 30, bootstrap_samples)
    boundary = []
    clipping = []
    nonfinite = []
    groups = aggregate["groups"]
    for track in ("controlled_duration", "end_to_end"):
        for factor in (0.8, 1.0, 1.2):
            left = groups[f"I001|{track}|{factor}"]
            right = groups[f"I004|{track}|{factor}"]
            boundary.append(right["boundary_jump_error"] - left["boundary_jump_error"])
            clipping.extend((left["clipping_rate"], right["clipping_rate"]))
            nonfinite.extend((left["nonfinite_rate"], right["nonfinite_rate"]))
    thresholds = cfg["thresholds"]
    checks = {
        "normal_noninferiority": normal["ci95_high"] <= thresholds["normal_noninferiority_relative_margin"],
        "scaled_average_superiority": scaled["ci95_high"] < 0.0,
        "end_to_end_noninferiority": end_to_end["ci95_high"] <= thresholds["end_to_end_relative_margin"],
        "duration_accuracy": max(group["duration_ratio_error"] for group in groups.values()) <= thresholds["duration_ratio_error_max"],
        "finite_and_unclipped": max(nonfinite) == 0.0 and max(clipping) <= thresholds["clipping_rate_max"],
        "asr_eligible": calibration["eligible_as_version_judge"],
    }
    report = {
        "schema_version": 1,
        "work_package": "P3-W03",
        "comparison": "I001_vs_I004",
        "sample_count": len(samples["unit_ids"]),
        "generated_condition_count": len(rows),
        "primary_metric": "token-resampled frozen-reference normalized mel MAE",
        "paired_bootstrap": {"normal_controlled_relative": normal, "scaled_controlled_absolute_average": scaled, "end_to_end_three_rate_relative_average": end_to_end},
        "checks": checks,
        "boundary_jump_error_mean_difference_i004_minus_i001": float(np.mean(boundary)),
        "asr_decision": calibration,
        "objective_recommendation": (
            "I004可进入盲听确认；正常速与端到端非劣、快慢平均改善，但0.8改善与1.2退化不对称。"
            if all(checks[key] for key in ("normal_noninferiority", "scaled_average_superiority", "end_to_end_noninferiority", "duration_accuracy", "finite_and_unclipped"))
            else "客观门禁不足以把I004升级为统一候选；保留双版本并检查失败维度。"
        ),
        "candidate_pointer_changed": False,
    }
    write_json(formal / "metrics/objective_decision.json", report)
    write_json(formal / "metrics/token_level_comparison.json", {"schema_version": 1, **token_analysis(rows)})

    vocab = read_json(ROOT / cfg["vocabulary"])["vocab"]
    checkpoints = {"I001": ROOT / cfg["i001_checkpoint"], "I004": ROOT / cfg["i004_checkpoint"]}
    models = {model: model_from_checkpoint(path, vocab)[0] for model, path in checkpoints.items()}
    chain_cfg = torch.load(checkpoints["I004"], map_location="cpu", weights_only=False)["config"]
    chain = load_frozen_chain(chain_cfg, ROOT)
    unit_rows = {json.loads(line)["unit_id"]: json.loads(line) for line in (ROOT / cfg["unit_manifest"]).read_text(encoding="utf-8").splitlines()}
    sample_order = {unit_id: index for index, unit_id in enumerate(samples["unit_ids"])}
    blind_dir = root / "examples/blind"
    blind_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows, answer_rows = [], []
    for case_index, (unit_id, factor) in enumerate(choose_blind_cases(rows), 1):
        unit = unit_rows[unit_id]
        token_ids = torch.tensor(text_to_token_ids(unit["token_chars"], vocab), dtype=torch.long)
        outputs = {}
        for model_id, model in models.items():
            with torch.inference_mode():
                predicted = model.duration.predict_durations(model.encode_text(token_ids)).long()
            durations = scale_durations(predicted, factor, t_max=64)
            seed = int(cfg["seed"]) + sample_order[unit_id] * 100 + int(round(factor * 10)) + 10000
            outputs[model_id] = synthesize(model, chain, token_ids, durations, seed)["wave"]
        i004_is_a = case_index % 2 == 1
        mapping = {"A": "I004" if i004_is_a else "I001", "B": "I001" if i004_is_a else "I004"}
        files = {}
        for label, model_id in mapping.items():
            path = blind_dir / f"case_{case_index:02d}_{label}.wav"
            write_wav(path, outputs[model_id])
            files[label] = path.relative_to(root / "examples").as_posix()
        manifest_rows.append({"case_id": f"case_{case_index:02d}", "text": unit["text"], "rate_factor": factor, "A": files["A"], "B": files["B"], "questions": ["哪一个内容更完整？", "哪一个更清晰自然？", "两者是否都能听懂？"]})
        answer_rows.append({"case_id": f"case_{case_index:02d}", "A": mapping["A"], "B": mapping["B"], "unit_id": unit_id, "factor": factor})
    write_json(root / "examples/blind_manifest.json", {"schema_version": 1, "identity_blinded": True, "cases": manifest_rows})
    write_json(root / "examples/answer_key.json", {"schema_version": 1, "do_not_open_before_scoring": True, "cases": answer_rows})
    print(json.dumps({"objective": report, "blind_cases": len(manifest_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
