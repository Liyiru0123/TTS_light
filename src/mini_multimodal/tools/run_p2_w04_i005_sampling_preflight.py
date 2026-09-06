from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from mini_multimodal.iteration_management import atomic_write_json, sha256_file
from mini_multimodal.ldm.evaluator import (
    aggregate_evaluations,
    evaluate_image_v2,
    load_shape_prototypes,
)
from mini_multimodal.ldm.geometry_evaluation import load_attempt3_checkpoint
from mini_multimodal.ldm.revision_evaluation import sample_latents_v2
from mini_multimodal.ldm.revision_training import load_config


ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "configs/p2_w04_i005.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _value(metric: Any) -> float:
    return float(metric["value"] if isinstance(metric, Mapping) and "value" in metric else metric)


def _successes(metric: Mapping[str, Any]) -> int:
    return int(metric["successes"])


def _batch_features(rows: Sequence[Mapping[str, Any]], start: int, batch_size: int) -> list[dict[str, str]]:
    features = []
    for row in rows[start : start + batch_size]:
        obj = row["objects"][0]
        features.append(
            {
                "shape": str(obj["shape"]),
                "color": str(obj["color"]),
                "size": str(obj["size"]),
                "position": str(obj["position_region"]),
                "pair": str(row["composition_pair"]),
            }
        )
    return features


def _selection_score(features: Sequence[Mapping[str, str]], starts: Sequence[int]) -> tuple[Any, ...]:
    shapes = Counter(item["shape"] for item in features)
    sizes = Counter(item["size"] for item in features)
    shape_values = [shapes[name] for name in ("circle", "square", "triangle", "rectangle", "pentagon")]
    size_values = [sizes[name] for name in ("small", "medium", "large")]
    return (
        min(shape_values),
        -max(shape_values) + min(shape_values),
        min(size_values),
        len({item["pair"] for item in features}),
        len({item["position"] for item in features}),
        len({item["color"] for item in features}),
        tuple(-start for start in starts),
    )


def _choose_batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    subset: str,
    batch_count: int,
    batch_size: int,
) -> list[int]:
    candidates = [
        start
        for start in range(0, len(rows), batch_size)
        if len(rows[start : start + batch_size]) == batch_size
        and all(row["subset"] == subset for row in rows[start : start + batch_size])
    ]
    if len(candidates) < batch_count:
        raise RuntimeError(f"not enough complete {subset} batches")
    best: tuple[Any, ...] | None = None
    best_starts: tuple[int, ...] | None = None
    for starts in combinations(candidates, batch_count):
        features = [
            item
            for start in starts
            for item in _batch_features(rows, start, batch_size)
        ]
        score = _selection_score(features, starts)
        if best is None or score > best:
            best = score
            best_starts = starts
    assert best_starts is not None
    return list(best_starts)


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    objects = [row["objects"][0] for row in rows]
    return {
        "sample_count": len(rows),
        "shape_counts": dict(sorted(Counter(obj["shape"] for obj in objects).items())),
        "color_counts": dict(sorted(Counter(obj["color"] for obj in objects).items())),
        "size_counts": dict(sorted(Counter(obj["size"] for obj in objects).items())),
        "position_counts": dict(sorted(Counter(obj["position_region"] for obj in objects).items())),
        "composition_pair_count": len({row["composition_pair"] for row in rows}),
    }


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": int(metrics["sample_count"]),
        "count_successes": _successes(metrics["count_accuracy"]),
        "count_accuracy": _value(metrics["count_accuracy"]),
        "accepted_shape_successes": _successes(metrics["shape_accuracy"]),
        "shape_accuracy": _value(metrics["shape_accuracy"]),
        "fully_correct_successes": _successes(metrics["fully_correct_rate"]),
        "fully_correct_rate": _value(metrics["fully_correct_rate"]),
        "position_successes": _successes(metrics["position_accuracy"]),
        "position_accuracy": _value(metrics["position_accuracy"]),
        "normalized_template_iou_mean": _value(metrics["normalized_template_iou_mean"]),
        "boundary_score_mean": _value(metrics["boundary_score_mean"]),
        "shape_successes": {
            shape: _successes(group["shape_accuracy"])
            for shape, group in sorted(metrics["shape_groups"].items())
        },
    }


def _delta(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, int]:
    return {
        name: int(current[name]) - int(baseline[name])
        for name in (
            "count_successes",
            "accepted_shape_successes",
            "fully_correct_successes",
            "position_successes",
        )
    }


def _baseline_core(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "predicted_object_count": row["predicted_object_count"],
        "foreground_present": row["foreground_present"],
        "is_fully_correct": row["is_fully_correct"],
        "pairs": row["pairs"],
        "probe": row["probe"],
    }


@torch.no_grad()
def main() -> None:
    spec = _load_json(SPEC_PATH)
    base_config = load_config(ROOT)
    preflight = spec["sampling_preflight"]
    output = ROOT / preflight["output"]
    output.mkdir(parents=True, exist_ok=True)

    eval_dir = ROOT / base_config["output_eval_data"]
    result_dir = ROOT / base_config["output_results"]
    rows = [
        json.loads(line)
        for line in (eval_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    batch_size = int(preflight["batch_size"])
    seen_starts = _choose_batches(
        rows,
        subset="seen_single",
        batch_count=int(preflight["seen_batch_count"]),
        batch_size=batch_size,
    )
    unseen_starts = _choose_batches(
        rows,
        subset="unseen_single",
        batch_count=int(preflight["unseen_batch_count"]),
        batch_size=batch_size,
    )
    starts = sorted(seen_starts + unseen_starts)
    selected_rows = [row for start in starts for row in rows[start : start + batch_size]]

    masks_payload = np.load(eval_dir / "ideal_masks.npz", allow_pickle=False)
    ideal_masks = masks_payload["amodal"]
    masks_payload.close()
    prototypes = load_shape_prototypes(result_dir / "evaluator_shape_templates.npz")
    freeze = _load_json(result_dir / "threshold_freeze.json")
    loaded = load_attempt3_checkpoint(ROOT, base_config, "control")
    scaling = float(
        _load_json(ROOT / base_config["output_data"] / "manifest.json")["latent_scaling_factor"]
    )
    original_rows = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (result_dir / "attempt3_control_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    base_seed = int(base_config["seed"]) + 60000
    setting_reports: dict[str, Any] = {}
    setting_predictions: dict[str, list[dict[str, Any]]] = {}
    baseline_matches: list[bool] = []
    started_all = perf_counter()

    for setting in preflight["settings"]:
        setting_id = str(setting["id"])
        predictions: list[dict[str, Any]] = []
        image_dir = output / "images" / setting_id
        image_dir.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        for start in starts:
            batch_rows = rows[start : start + batch_size]
            sampled = sample_latents_v2(
                loaded["model"],
                loaded["schedule"],
                loaded["tokenizer"],
                [row["prompt"] for row in batch_rows],
                sample_steps=int(setting["sample_steps"]),
                guidance_scale=float(setting["guidance_scale"]),
                seed=base_seed + start,
            )
            images = loaded["vae"].decode(sampled["latent"] / scaling).clamp(0, 1)
            for offset, (row, image) in enumerate(zip(batch_rows, images, strict=True)):
                subset_name = "unseen_single" if row["subset"] == "unseen_single" else "seen_single"
                threshold = freeze["thresholds"][subset_name]
                evaluation = evaluate_image_v2(
                    image,
                    row,
                    ideal_masks[row["ideal_mask_index"]],
                    fully_correct_shape_iou=float(threshold["normalized_template_iou_mean"]),
                    shape_prototypes=prototypes,
                )
                image_path = image_dir / f"sample_{start + offset:03d}.png"
                Image.fromarray(
                    image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy(),
                    mode="RGB",
                ).save(image_path)
                evaluation.update(
                    {
                        "generated_image": image_path.relative_to(ROOT).as_posix(),
                        "generated_image_sha256": sha256_file(image_path),
                        "sample_seed": base_seed + start,
                        "checkpoint": loaded["checkpoint"].relative_to(ROOT).as_posix(),
                        "sampling_setting": setting_id,
                        "sample_steps": int(setting["sample_steps"]),
                        "guidance_scale": float(setting["guidance_scale"]),
                    }
                )
                predictions.append(evaluation)
                if setting_id == "baseline":
                    baseline_matches.append(
                        _baseline_core(evaluation) == _baseline_core(original_rows[row["record_id"]])
                        and int(evaluation["sample_seed"])
                        == int(original_rows[row["record_id"]]["sample_seed"])
                    )
        by_subset = {
            subset: aggregate_evaluations([row for row in predictions if row["subset"] == subset])
            for subset in ("seen_single", "unseen_single")
        }
        setting_predictions[setting_id] = predictions
        setting_reports[setting_id] = {
            "sample_steps": int(setting["sample_steps"]),
            "guidance_scale": float(setting["guidance_scale"]),
            "wall_seconds": perf_counter() - started,
            "seen_single": _compact_metrics(by_subset["seen_single"]),
            "unseen_single": _compact_metrics(by_subset["unseen_single"]),
        }
        _write_jsonl(output / "predictions" / f"{setting_id}.jsonl", predictions)

    baseline = setting_reports["baseline"]
    shortlisted: list[str] = []
    minimum_seen = preflight["shortlist_minimum_deltas_on_seen"]
    minimum_unseen = preflight["shortlist_minimum_deltas_on_unseen"]
    for setting_id, report in setting_reports.items():
        if setting_id == "baseline":
            report["shortlisted"] = False
            report["shortlist_reason"] = "reference setting"
            continue
        seen_delta = _delta(report["seen_single"], baseline["seen_single"])
        unseen_delta = _delta(report["unseen_single"], baseline["unseen_single"])
        report["delta_from_baseline"] = {
            "seen_single": seen_delta,
            "unseen_single": unseen_delta,
        }
        checks = {
            "seen_count": seen_delta["count_successes"] >= int(minimum_seen["count_successes"]),
            "seen_shape": seen_delta["accepted_shape_successes"]
            >= int(minimum_seen["accepted_shape_successes"]),
            "seen_fully_correct": seen_delta["fully_correct_successes"]
            >= int(minimum_seen["fully_correct_successes"]),
            "seen_position": seen_delta["position_successes"]
            >= int(minimum_seen["position_successes"]),
            "unseen_shape": unseen_delta["accepted_shape_successes"]
            >= int(minimum_unseen["accepted_shape_successes"]),
            "unseen_fully_correct": unseen_delta["fully_correct_successes"]
            >= int(minimum_unseen["fully_correct_successes"]),
        }
        report["shortlist_checks"] = checks
        report["shortlisted"] = all(checks.values())
        if report["shortlisted"]:
            shortlisted.append(setting_id)

    def ranking(setting_id: str) -> tuple[int, ...]:
        delta_seen = setting_reports[setting_id]["delta_from_baseline"]["seen_single"]
        delta_unseen = setting_reports[setting_id]["delta_from_baseline"]["unseen_single"]
        return (
            delta_seen["fully_correct_successes"],
            delta_seen["count_successes"],
            delta_seen["accepted_shape_successes"],
            delta_unseen["fully_correct_successes"],
            delta_unseen["accepted_shape_successes"],
        )

    selected = max(shortlisted, key=ranking) if shortlisted else None
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "planning_target": "I005",
        "status": "sampling_preflight_completed",
        "sources": {
            "spec": {"path": SPEC_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(SPEC_PATH)},
            "warmstart": {
                "path": loaded["checkpoint"].relative_to(ROOT).as_posix(),
                "sha256": sha256_file(loaded["checkpoint"]),
            },
            "vae": {
                "path": str(spec["frozen_assets"]["vae"]),
                "sha256": sha256_file(ROOT / spec["frozen_assets"]["vae"]),
            },
            "evaluation_manifest": {
                "path": (eval_dir / "manifest.json").relative_to(ROOT).as_posix(),
                "sha256": sha256_file(eval_dir / "manifest.json"),
            },
            "thresholds": {
                "path": (result_dir / "threshold_freeze.json").relative_to(ROOT).as_posix(),
                "sha256": sha256_file(result_dir / "threshold_freeze.json"),
            },
            "shape_prototypes": {
                "path": (result_dir / "evaluator_shape_templates.npz").relative_to(ROOT).as_posix(),
                "sha256": sha256_file(result_dir / "evaluator_shape_templates.npz"),
            },
        },
        "selection": {
            "policy": "exhaustive complete-batch selection maximizing minimum shape and size coverage, then pair/position/color coverage",
            "batch_size": batch_size,
            "seen_batch_starts": seen_starts,
            "unseen_batch_starts": unseen_starts,
            "record_ids": [row["record_id"] for row in selected_rows],
            "seen_single": _coverage([row for row in selected_rows if row["subset"] == "seen_single"]),
            "unseen_single": _coverage([row for row in selected_rows if row["subset"] == "unseen_single"]),
        },
        "baseline_reproduction": {
            "matched_records": sum(baseline_matches),
            "total_records": len(baseline_matches),
            "verified": len(baseline_matches) == len(selected_rows) and all(baseline_matches),
        },
        "settings": setting_reports,
        "shortlist": {
            "minimum_seen_deltas": minimum_seen,
            "minimum_unseen_deltas": minimum_unseen,
            "ranking": preflight["shortlist_ranking"],
            "eligible_settings": shortlisted,
            "selected_setting": selected,
        },
        "decision": {
            "full_fixed_evaluation_required": selected is not None,
            "clean_refiner_implementation_required_if_no_full_pass": selected is None,
            "reason": (
                "one registered sampler setting met every small-sample shortlist condition"
                if selected is not None
                else "no registered sampler setting jointly improved the predeclared residual metrics"
            ),
        },
        "wall_seconds": perf_counter() - started_all,
    }
    report["verified"] = bool(report["baseline_reproduction"]["verified"])
    atomic_write_json(output / "sampling_preflight_report.json", report)
    if not report["verified"]:
        raise RuntimeError("sampling preflight failed baseline reproduction")
    print(
        json.dumps(
            {
                "output": (output / "sampling_preflight_report.json").relative_to(ROOT).as_posix(),
                "seen_batch_starts": seen_starts,
                "unseen_batch_starts": unseen_starts,
                "baseline_reproduced": report["baseline_reproduction"],
                "eligible_settings": shortlisted,
                "selected_setting": selected,
                "wall_seconds": report["wall_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
