from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

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
from mini_multimodal.ldm.revision_evaluation import _fixed_checks, sample_latents_v2
from mini_multimodal.ldm.revision_training import load_config


ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "configs/p2_w04_i005.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _value(metric: Any) -> float:
    return float(metric["value"] if isinstance(metric, Mapping) and "value" in metric else metric)


def _compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _value(metrics[name])
        for name in (
            "foreground_presence",
            "count_accuracy",
            "color_accuracy",
            "shape_accuracy",
            "normalized_template_iou_mean",
            "boundary_score_mean",
            "size_accuracy",
            "position_accuracy",
            "center_distance_mean",
            "fully_correct_rate",
        )
    } | {
        "shape_groups": {
            shape: _value(group["shape_accuracy"])
            for shape, group in sorted(metrics["shape_groups"].items())
        }
    }


def _examples(
    output: Path,
    predictions: list[dict[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> None:
    improved = [
        row
        for row in predictions
        if row["is_fully_correct"] and not baseline[row["record_id"]]["is_fully_correct"]
    ]
    regressed = [
        row
        for row in predictions
        if not row["is_fully_correct"] and baseline[row["record_id"]]["is_fully_correct"]
    ]
    persistent_failures = [
        row
        for row in predictions
        if not row["is_fully_correct"] and not baseline[row["record_id"]]["is_fully_correct"]
    ]
    selected: list[tuple[str, dict[str, Any]]] = []
    selected.extend(("由错误变为完全正确", row) for row in improved[:4])
    selected.extend(("由完全正确退化为错误", row) for row in regressed[:2])
    needed = 8 - len(selected)
    selected.extend(("仍然错误", row) for row in persistent_failures[:needed])
    payload = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "planning_target": "I005",
        "artifact_role": "sampler-only full-confirmation examples; not a new model iteration",
        "example_count": len(selected),
        "examples": [],
    }
    lines = [
        "# I005采样排除试验：完整评价直观样例",
        "",
        "> 这些样例仍使用I004控制支，只把CFG从2.0改为2.5；它们不是新模型能力样例。",
        "",
    ]
    for index, (category, row) in enumerate(selected, 1):
        old = baseline[row["record_id"]]
        item = {
            "index": index,
            "category": category,
            "record_id": row["record_id"],
            "prompt": row["prompt"],
            "baseline_image": old["generated_image"],
            "candidate_image": row["generated_image"],
            "baseline_fully_correct": bool(old["is_fully_correct"]),
            "candidate_fully_correct": bool(row["is_fully_correct"]),
            "baseline_predicted_object_count": int(old["predicted_object_count"]),
            "candidate_predicted_object_count": int(row["predicted_object_count"]),
            "candidate_pairs": row["pairs"],
        }
        payload["examples"].append(item)
        lines.extend(
            [
                f"## 样例{index}：{category}",
                "",
                f"- 记录：`{row['record_id']}`",
                f"- 输入文本：{row['prompt']}",
                f"- 默认CFG 2.0：对象数{old['predicted_object_count']}，完全正确={old['is_fully_correct']}；图像`{old['generated_image']}`",
                f"- 候选CFG 2.5：对象数{row['predicted_object_count']}，完全正确={row['is_fully_correct']}；图像`{row['generated_image']}`",
                "",
            ]
        )
    atomic_write_json(output / "full_candidate_examples.json", payload)
    (output / "full_candidate_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    spec = _json(SPEC_PATH)
    config = load_config(ROOT)
    preflight_output = ROOT / spec["sampling_preflight"]["output"]
    preflight = _json(preflight_output / "sampling_preflight_report.json")
    selected_id = preflight["shortlist"]["selected_setting"]
    if not selected_id:
        raise RuntimeError("sampler confirmation requires a shortlisted preflight setting")
    setting = next(
        item for item in spec["sampling_preflight"]["settings"] if item["id"] == selected_id
    )
    if not preflight["baseline_reproduction"]["verified"]:
        raise RuntimeError("preflight baseline reproduction did not pass")

    result_dir = ROOT / config["output_results"]
    eval_dir = ROOT / config["output_eval_data"]
    rows = [
        json.loads(line)
        for line in (eval_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    mask_payload = np.load(eval_dir / "ideal_masks.npz", allow_pickle=False)
    masks = mask_payload["amodal"]
    mask_payload.close()
    prototypes = load_shape_prototypes(result_dir / "evaluator_shape_templates.npz")
    freeze = _json(result_dir / "threshold_freeze.json")
    loaded = load_attempt3_checkpoint(ROOT, config, "control")
    scaling = float(_json(ROOT / config["output_data"] / "manifest.json")["latent_scaling_factor"])
    output = preflight_output / "full_confirmation"
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    base_seed = int(config["seed"]) + 60000
    started = perf_counter()
    for start in range(0, len(rows), 8):
        batch_rows = rows[start : start + 8]
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
                masks[row["ideal_mask_index"]],
                fully_correct_shape_iou=float(threshold["normalized_template_iou_mean"]),
                shape_prototypes=prototypes,
            )
            image_path = image_dir / f"candidate_{start + offset:03d}.png"
            Image.fromarray(
                image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), mode="RGB"
            ).save(image_path)
            evaluation.update(
                {
                    "generated_image": image_path.relative_to(ROOT).as_posix(),
                    "generated_image_sha256": sha256_file(image_path),
                    "sample_seed": base_seed + start,
                    "checkpoint": loaded["checkpoint"].relative_to(ROOT).as_posix(),
                    "sampling_setting": selected_id,
                    "sample_steps": int(setting["sample_steps"]),
                    "guidance_scale": float(setting["guidance_scale"]),
                }
            )
            predictions.append(evaluation)

    metrics = {
        subset: aggregate_evaluations([row for row in predictions if row["subset"] == subset])
        for subset in ("seen_single", "unseen_single", "multi_2", "multi_3")
    }
    seen_checks = _fixed_checks(
        metrics["seen_single"], freeze["thresholds"]["seen_single"], "seen_single"
    )
    unseen_checks = _fixed_checks(
        metrics["unseen_single"], freeze["thresholds"]["unseen_single"], "unseen_single"
    )
    reference = freeze["thresholds"]["multi_object_reference"]
    multi_checks = {
        subset: {
            name: _value(metrics[subset][name]) >= float(reference[name])
            for name in ("count_accuracy", "color_accuracy", "shape_accuracy", "position_accuracy")
        }
        for subset in ("multi_2", "multi_3")
    }
    baseline_report = _json(result_dir / "attempt3_control_generation_metrics.json")
    report = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "planning_target": "I005",
        "status": "sampler_only_full_confirmation_completed",
        "model_iteration": "I004/control",
        "new_model_trained": False,
        "setting": setting,
        "checkpoint": loaded["checkpoint"].relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(loaded["checkpoint"]),
        "evaluation_manifest_sha256": sha256_file(eval_dir / "manifest.json"),
        "sample_count": len(predictions),
        "wall_seconds": perf_counter() - started,
        "metrics": metrics,
        "compact_comparison": {
            "baseline_cfg_2_0": {
                subset: _compact(baseline_report["metrics"][subset])
                for subset in ("seen_single", "unseen_single")
            },
            "candidate_cfg_2_5": {
                subset: _compact(metrics[subset])
                for subset in ("seen_single", "unseen_single")
            },
        },
        "checks": {
            "seen_single": seen_checks,
            "unseen_single": unseen_checks,
            "multi_object_nonblocking_reference": multi_checks,
        },
        "seen_single_verified": all(seen_checks.values()),
        "unseen_single_verified": all(unseen_checks.values()),
    }
    report["sampler_only_passed_all_blocking_items"] = bool(
        report["seen_single_verified"] and report["unseen_single_verified"]
    )
    report["decision"] = {
        "clean_refiner_implementation_required": not report[
            "sampler_only_passed_all_blocking_items"
        ],
        "reason": (
            "sampler-only candidate passed every frozen blocking item"
            if report["sampler_only_passed_all_blocking_items"]
            else "sampler-only candidate did not pass every frozen blocking item"
        ),
    }
    _write_jsonl(output / "full_candidate_predictions.jsonl", predictions)
    atomic_write_json(output / "full_candidate_metrics.json", report)
    baseline = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (result_dir / "attempt3_control_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    _examples(output, predictions, baseline)
    print(
        json.dumps(
            {
                "output": (output / "full_candidate_metrics.json").relative_to(ROOT).as_posix(),
                "setting": setting,
                "seen_single_verified": report["seen_single_verified"],
                "unseen_single_verified": report["unseen_single_verified"],
                "sampler_only_passed_all_blocking_items": report[
                    "sampler_only_passed_all_blocking_items"
                ],
                "clean_refiner_implementation_required": report["decision"][
                    "clean_refiner_implementation_required"
                ],
                "wall_seconds": report["wall_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
