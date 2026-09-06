from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/P2-W04_tiny_image_ldm/revision_v2"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(metric: object) -> float:
    if isinstance(metric, dict) and "value" in metric:
        return float(metric["value"])
    return float(metric)


def _compact(row: dict, category: str, explanation: str) -> dict:
    return {
        "category": category,
        "explanation": explanation,
        "record_id": row["record_id"],
        "subset": row["subset"],
        "prompt": row["prompt"],
        "generated_image": row["generated_image"],
        "generated_image_sha256": row["generated_image_sha256"],
        "target_object_count": row["target_object_count"],
        "predicted_object_count": row["predicted_object_count"],
        "count_correct": row["count_correct"],
        "is_fully_correct": row["is_fully_correct"],
        "pairs": row["pairs"],
    }


def _write_checkpoint_manifest() -> None:
    checkpoint_dir = RESULT / "training_grounded/checkpoints"
    items: dict[str, dict] = {}
    for name in ("best", "last"):
        path = checkpoint_dir / f"{name}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        items[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "epoch": int(payload["epoch"]),
            "best_metric": float(payload["best_metric"]),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
            "use": (
                "Second-round fixed evaluation and diagnosis; not a final accepted model."
                if name == "best"
                else "Resume/diagnostic endpoint; not selected for reported generation metrics."
            ),
        }
    manifest = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "attempt": "text_derived_object_grounding_v2",
        "status": "diagnostic_failed_generation_thresholds",
        "selection_metric": "validation noise_mse (minimize)",
        **items,
        "inference_package": None,
        "inference_package_reason": (
            "A final self-contained package is not exported for a checkpoint that failed frozen generation thresholds."
        ),
        "usage_document": "docs/P2-W04_第二轮检查点使用说明.md",
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_summary() -> None:
    generation = _json(RESULT / "grounded_generation_metrics.json")
    denoising = _json(RESULT / "grounded_denoising_metrics.json")
    metrics = generation["metrics"]
    summary = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "attempt": "text_derived_object_grounding_v2",
        "status": "failed_frozen_generation_thresholds",
        "checkpoint": generation["checkpoint"],
        "checkpoint_sha256": generation["checkpoint_sha256"],
        "training_holdout_object_count": generation["training_holdout_object_count"],
        "denoising_verified": bool(denoising["verified_for_generation_evaluation"]),
        "seen_single_verified": generation["seen_single_verified"],
        "unseen_single_verified": generation["unseen_single_verified"],
        "seen_single": {
            name: _value(metrics["seen_single"][name])
            for name in (
                "count_accuracy",
                "color_accuracy",
                "shape_accuracy",
                "size_accuracy",
                "position_accuracy",
                "center_distance_mean",
                "normalized_template_iou_mean",
                "boundary_score_mean",
                "fully_correct_rate",
            )
        },
        "unseen_single": {
            name: _value(metrics["unseen_single"][name])
            for name in (
                "count_accuracy",
                "color_accuracy",
                "shape_accuracy",
                "size_accuracy",
                "position_accuracy",
                "center_distance_mean",
                "normalized_template_iou_mean",
                "boundary_score_mean",
                "fully_correct_rate",
            )
        },
        "decision_required": (
            "Whether to research and design a third-round training-only shape/foreground auxiliary objective."
        ),
        "source_metrics": "results/P2-W04_tiny_image_ldm/revision_v2/grounded_generation_metrics.json",
    }
    (RESULT / "attempt2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_evidence_index() -> None:
    generation = _json(RESULT / "grounded_generation_metrics.json")
    denoising = _json(RESULT / "grounded_denoising_metrics.json")
    freeze = _json(RESULT / "threshold_freeze.json")
    entries: list[dict] = []
    for subset in ("seen_single", "unseen_single"):
        sample_count = generation["metrics"][subset]["sample_count"]
        for name in (
            "count_accuracy",
            "color_accuracy",
            "shape_accuracy",
            "size_accuracy",
            "position_accuracy",
            "center_distance_mean",
            "normalized_template_iou_mean",
            "boundary_score_mean",
            "fully_correct_rate",
        ):
            entries.append(
                {
                    "claim_id": f"ATTEMPT2-{subset.upper()}-{name.upper()}",
                    "acceptance_item_id": "MODEL-LDM-002",
                    "source_file": "grounded_generation_metrics.json",
                    "field_path": f"metrics.{subset}.{name}",
                    "value": _value(generation["metrics"][subset][name]),
                    "scope": subset,
                    "sample_count": sample_count,
                    "threshold": freeze["thresholds"][subset].get(name),
                    "passed": generation["checks"][subset].get(name),
                }
            )
    entries.extend(
        [
            {
                "claim_id": "ATTEMPT2-DENOISING-MSE",
                "acceptance_item_id": "MODEL-LDM-001",
                "source_file": "grounded_denoising_metrics.json",
                "field_path": "core.noise_mse",
                "value": denoising["core"]["noise_mse"],
                "scope": "test",
                "sample_count": denoising["core"]["sample_count"],
                "passed": denoising["checks"]["noise_mse"],
            },
            {
                "claim_id": "ATTEMPT2-GROUNDING-ABLATION-MSE",
                "acceptance_item_id": "MODEL-LDM-003",
                "source_file": "grounded_denoising_metrics.json",
                "field_path": "ablations.disable_object_grounding.noise_mse",
                "value": denoising["ablations"]["disable_object_grounding"]["noise_mse"],
                "scope": "test_ablation",
                "sample_count": denoising["ablations"]["disable_object_grounding"]["sample_count"],
                "passed": denoising["ablations"]["disable_object_grounding"]["noise_mse"]
                > denoising["core"]["noise_mse"],
            },
        ]
    )
    payload = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "attempt": 2,
        "status": "failed_frozen_generation_thresholds",
        "entries": entries,
    }
    (RESULT / "attempt2_evidence_index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_examples() -> None:
    rows = [
        json.loads(line)
        for line in (RESULT / "grounded_generation_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selected: list[dict] = []
    used: set[str] = set()

    for shape in ("triangle", "rectangle"):
        row = next(
            row
            for row in rows
            if row["subset"] == "seen_single"
            and row["is_fully_correct"]
            and row["pairs"][0]["target"]["shape"] == shape
        )
        selected.append(_compact(row, "完整正确", f"{shape}样例满足冻结的全部逐属性判据。"))
        used.add(row["record_id"])

    predicates = [
        (
            "圆形不确定",
            "颜色和位置正确，但生成轮廓没有以足够置信度通过圆形判据。",
            lambda r: r["subset"] == "seen_single"
            and r["pairs"][0]["target"]["shape"] == "circle"
            and r["pairs"][0]["color_correct"]
            and r["pairs"][0]["position_correct"]
            and not r["pairs"][0]["shape_correct"],
        ),
        (
            "正方形错误",
            "颜色和位置正确，但正方形轮廓没有可靠形成。",
            lambda r: r["subset"] == "seen_single"
            and r["pairs"][0]["target"]["shape"] == "square"
            and r["pairs"][0]["color_correct"]
            and r["pairs"][0]["position_correct"]
            and not r["pairs"][0]["shape_correct"],
        ),
        (
            "五边形错误",
            "颜色和位置正确，但五边形边界仍不稳定。",
            lambda r: r["subset"] == "seen_single"
            and r["pairs"][0]["target"]["shape"] == "pentagon"
            and r["pairs"][0]["color_correct"]
            and r["pairs"][0]["position_correct"]
            and not r["pairs"][0]["shape_correct"],
        ),
        (
            "碎片/数量错误",
            "主对象存在，但额外连通区域或断裂使数量判据失败。",
            lambda r: r["subset"] == "seen_single" and not r["count_correct"],
        ),
        (
            "未见组合错误",
            "训练留出隔离有效，但该未见颜色—形状组合没有达到完整正确。",
            lambda r: r["subset"] == "unseen_single" and not r["is_fully_correct"],
        ),
        (
            "多对象边界",
            "三对象文本展示当前对象计数、形状与绑定能力的非阻塞边界。",
            lambda r: r["subset"] == "multi_3" and not r["is_fully_correct"],
        ),
    ]
    for category, explanation, predicate in predicates:
        row = next(row for row in rows if row["record_id"] not in used and predicate(row))
        selected.append(_compact(row, category, explanation))
        used.add(row["record_id"])

    payload = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "attempt": 2,
        "source_predictions": "grounded_generation_predictions.jsonl",
        "examples": selected,
    }
    (RESULT / "attempt2_capability_examples.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# P2-W04第二轮修订模型能力样例",
        "",
        "> 本册来自`grounded_generation_predictions.jsonl`，用于直观理解第二轮模型。模型取得明显进步，但没有通过全部冻结验收阈值。",
        "",
    ]
    for index, row in enumerate(selected, 1):
        relative_image = Path(row["generated_image"]).relative_to(
            "results/P2-W04_tiny_image_ldm/revision_v2"
        ).as_posix()
        first_pair = row["pairs"][0]
        prediction = first_pair["prediction"]
        predicted_text = (
            "未检出"
            if prediction is None
            else f"{prediction['color']} / {prediction['shape']} / {prediction['size']} / {prediction['position_region']}"
        )
        target = first_pair["target"]
        lines.extend(
            [
                f"## {index}. {row['category']}",
                "",
                f"- 记录：`{row['record_id']}`（`{row['subset']}`）",
                f"- 文本条件：{row['prompt']}",
                f"- 目标首对象：{target['color']} / {target['shape']} / {target['size']} / {target['position_region']}",
                f"- 检测首对象：{predicted_text}",
                f"- 数量：目标{row['target_object_count']}，检测{row['predicted_object_count']}；完整正确：{row['is_fully_correct']}",
                f"- 说明：{row['explanation']}",
                "",
                f"![{row['category']}]({relative_image})",
                "",
            ]
        )
    (RESULT / "attempt2_capability_examples.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _write_checkpoint_manifest()
    _write_summary()
    _write_evidence_index()
    _write_examples()


if __name__ == "__main__":
    main()
