from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/P2-W04_tiny_image_ldm/revision_v2"
BRANCHES = {
    "second_round": ("grounded_generation_metrics.json", "grounded_generation_predictions.jsonl"),
    "control": ("attempt3_control_generation_metrics.json", "attempt3_control_predictions.jsonl"),
    "geometry": ("attempt3_geometry_generation_metrics.json", "attempt3_geometry_predictions.jsonl"),
    "geometry_lite": ("attempt3_geometry_lite_generation_metrics.json", "attempt3_geometry_lite_predictions.jsonl"),
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(value: object) -> float:
    return float(value["value"] if isinstance(value, dict) and "value" in value else value)


def _metrics() -> dict[str, dict]:
    return {branch: _json(RESULT / names[0]) for branch, names in BRANCHES.items()}


def _predictions() -> dict[str, dict[str, dict]]:
    return {
        branch: {
            row["record_id"]: row
            for row in (
                json.loads(line)
                for line in (RESULT / names[1]).read_text(encoding="utf-8").splitlines()
            )
        }
        for branch, names in BRANCHES.items()
    }


def _write_comparison() -> None:
    reports = _metrics()
    names = (
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
    comparison: dict[str, dict] = {}
    for branch, report in reports.items():
        comparison[branch] = {
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "seen_single_verified": report["seen_single_verified"],
            "unseen_single_verified": report["unseen_single_verified"],
            "seen_single": {name: _value(report["metrics"]["seen_single"][name]) for name in names},
            "unseen_single": {name: _value(report["metrics"]["unseen_single"][name]) for name in names},
            "seen_shape_groups": {
                shape: _value(group["shape_accuracy"])
                for shape, group in report["metrics"]["seen_single"]["shape_groups"].items()
            },
        }
    baseline = comparison["second_round"]["seen_single"]
    for branch in ("control", "geometry", "geometry_lite"):
        comparison[branch]["seen_delta_vs_second_round"] = {
            name: comparison[branch]["seen_single"][name] - baseline[name] for name in names
        }
    payload = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "attempt": "training_only_geometry_auxiliary_v3",
        "status": "failed_frozen_generation_thresholds",
        "fixed_evaluation_sample_count": 420,
        "same_warmstart_sha256": "636821b5acf0441a31520f46e4237fe77bacbcaaed6c3083d69a28386824446c",
        "branches": comparison,
        "conclusions": {
            "resolution_is_not_hard_ceiling": True,
            "evidence": "control continuation improves seen shape 0.5911->0.7244 and unseen branch passes without changing 64x64 resolution",
            "geometry_auxiliary_uniformly_better_than_control": False,
            "geometry_tradeoff": "strong geometry improves count, square and fully-correct rate but reduces aggregate shape/generalization relative to control",
            "p2_w04_verified": False,
        },
        "source_files": [names[0] for names in BRANCHES.values()],
    }
    (RESULT / "attempt3_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_evidence_index() -> None:
    reports = _metrics()
    freeze = _json(RESULT / "threshold_freeze.json")
    names = (
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
    entries: list[dict] = []
    for branch, report in reports.items():
        source_file = BRANCHES[branch][0]
        for subset in ("seen_single", "unseen_single"):
            for name in names:
                entries.append({
                    "claim_id": f"ATTEMPT3-{branch.upper()}-{subset.upper()}-{name.upper()}",
                    "acceptance_item_id": "MODEL-LDM-002",
                    "source_file": source_file,
                    "field_path": f"metrics.{subset}.{name}",
                    "value": _value(report["metrics"][subset][name]),
                    "scope": subset,
                    "sample_count": report["metrics"][subset]["sample_count"],
                    "threshold": freeze["thresholds"][subset].get(name),
                    "passed": report["checks"][subset].get(name),
                })
    payload = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "attempt": 3,
        "status": "failed_frozen_generation_thresholds",
        "entries": entries,
    }
    (RESULT / "attempt3_evidence_index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_manifests() -> None:
    for branch in ("control", "geometry", "geometry_lite"):
        directory = RESULT / f"training_attempt3_{branch}/checkpoints"
        manifest: dict[str, object] = {
            "schema_version": 1,
            "work_package": "P2-W04",
            "attempt": "training_only_geometry_auxiliary_v3",
            "branch": branch,
            "status": "diagnostic_failed_generation_thresholds",
            "selection_metric": "validation total loss (minimize)",
            "warmstart": "results/P2-W04_tiny_image_ldm/revision_v2/training_grounded/checkpoints/best.pt",
            "warmstart_sha256": "636821b5acf0441a31520f46e4237fe77bacbcaaed6c3083d69a28386824446c",
        }
        for name in ("best", "last"):
            path = directory / f"{name}.pt"
            state = torch.load(path, map_location="cpu", weights_only=False)
            manifest[name] = {
                "path": path.relative_to(ROOT).as_posix(),
                "epoch": int(state["epoch"]),
                "best_metric": float(state["best_metric"]),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
                "use": "Frozen evaluation/diagnosis" if name == "best" else "Resume and endpoint comparison",
            }
        manifest["inference_package"] = None
        manifest["inference_package_reason"] = "No attempt-3 branch passed every frozen generation threshold."
        manifest["usage_document"] = "docs/P2-W04_第三轮检查点使用说明.md"
        (directory / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def _compact(row: dict) -> dict:
    return {
        key: row[key]
        for key in (
            "record_id",
            "subset",
            "prompt",
            "generated_image",
            "generated_image_sha256",
            "target_object_count",
            "predicted_object_count",
            "count_correct",
            "is_fully_correct",
            "pairs",
        )
    }


def _write_examples() -> None:
    rows = _predictions()
    ground, control, geometry, lite = (
        rows["second_round"], rows["control"], rows["geometry"], rows["geometry_lite"]
    )
    ids = list(ground)
    selections: list[tuple[str, str, str]] = []
    stable = next(
        record_id for record_id in ids
        if ground[record_id]["subset"] == "seen_single"
        and all(branch[record_id]["is_fully_correct"] for branch in (ground, control, geometry, lite))
    )
    selections.append((stable, "稳定正确", "四个检查点均完整正确，展示已有能力的稳定部分。"))
    for shape, label in (("circle", "圆形改善"), ("square", "正方形改善"), ("pentagon", "五边形改善")):
        record_id = next(
            record_id for record_id in ids
            if ground[record_id]["subset"] == "seen_single"
            and ground[record_id]["pairs"][0]["target"]["shape"] == shape
            and not ground[record_id]["pairs"][0]["shape_correct"]
            and geometry[record_id]["pairs"][0]["shape_correct"]
        )
        selections.append((record_id, label, f"强几何支修复了第二轮的{shape}判定。"))
    aux_count = next(
        record_id for record_id in ids
        if ground[record_id]["subset"] == "seen_single"
        and not ground[record_id]["count_correct"]
        and geometry[record_id]["count_correct"]
        and geometry[record_id]["is_fully_correct"]
        and not control[record_id]["is_fully_correct"]
    )
    selections.append((aux_count, "几何支联合获益", "强几何支修复数量并达到完整正确，而同轮数控制支仍未完整正确。"))
    unseen = next(
        record_id for record_id in ids
        if ground[record_id]["subset"] == "unseen_single"
        and control[record_id]["is_fully_correct"]
        and not geometry[record_id]["is_fully_correct"]
    )
    selections.append((unseen, "控制支泛化获益", "无辅助续训支在该未见组合上完整正确，显示强辅助监督存在泛化权衡。"))
    persistent = next(
        record_id for record_id in ids
        if ground[record_id]["subset"] == "seen_single"
        and not any(branch[record_id]["is_fully_correct"] for branch in (ground, control, geometry, lite))
    )
    selections.append((persistent, "持续失败", "三种第三轮分支均未使该已见提示达到完整正确。"))
    multi = next(
        record_id for record_id in ids
        if ground[record_id]["subset"] == "multi_3"
        and not any(branch[record_id]["is_fully_correct"] for branch in (ground, control, geometry, lite))
    )
    selections.append((multi, "多对象边界", "多对象仍是非阻塞能力边界，所有分支均未完整正确。"))

    examples = [
        {
            "category": category,
            "explanation": explanation,
            "record_id": record_id,
            "branches": {branch: _compact(rows[branch][record_id]) for branch in BRANCHES},
        }
        for record_id, category, explanation in selections
    ]
    payload = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "attempt": 3,
        "examples": examples,
    }
    (RESULT / "attempt3_capability_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    labels = {
        "second_round": "第二轮",
        "control": "同轮数控制支",
        "geometry": "强几何支",
        "geometry_lite": "轻几何支",
    }
    lines = [
        "# P2-W04第三轮逐提示能力对照",
        "",
        "> 四列使用相同提示、评价样本与初始噪声协议。策展样例只帮助理解差异，正式结论以420样本统计为准。",
        "",
    ]
    for index, example in enumerate(examples, 1):
        base = example["branches"]["second_round"]
        lines.extend([
            f"## {index}. {example['category']}",
            "",
            f"- 记录：`{example['record_id']}`（`{base['subset']}`）",
            f"- 文本条件：{base['prompt']}",
            f"- 说明：{example['explanation']}",
            "",
        ])
        for branch in BRANCHES:
            row = example["branches"][branch]
            pair = row["pairs"][0]
            target = pair["target"]
            pred = pair["prediction"]
            pred_text = "未检出" if pred is None else f"{pred['color']} / {pred['shape']} / {pred['size']} / {pred['position_region']}"
            relative = Path(row["generated_image"]).relative_to("results/P2-W04_tiny_image_ldm/revision_v2").as_posix()
            lines.extend([
                f"### {labels[branch]}",
                "",
                f"目标首对象：{target['color']} / {target['shape']} / {target['size']} / {target['position_region']}；检测：{pred_text}。目标数量{row['target_object_count']}，检测{row['predicted_object_count']}；完整正确：{row['is_fully_correct']}。",
                "",
                f"![{labels[branch]}-{example['category']}]({relative})",
                "",
            ])
    (RESULT / "attempt3_capability_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    # The comparison above is useful for experimental analysis, but it is not a
    # substitute for the canonical per-run capability showcase required by the
    # project governance contract.  The strong-geometry branch is used here
    # because it has the highest fully-correct rate in attempt 3.  It is still
    # explicitly labelled as an unaccepted diagnostic checkpoint.
    showcase_dir = RESULT / "capability_examples"
    showcase_dir.mkdir(parents=True, exist_ok=True)
    canonical_examples: list[dict] = []
    for index, example in enumerate(examples, start=1):
        row = example["branches"]["geometry"]
        prompt_path = showcase_dir / f"example_{index:02d}_prompt.txt"
        prompt_path.write_text(str(row["prompt"]) + "\n", encoding="utf-8")
        expected_objects = [pair["target"] for pair in row["pairs"]]
        canonical_examples.append({
            "index": index,
            "source_record_id": row["record_id"],
            "evaluation_subset": row["subset"],
            "category": example["category"],
            "input_artifacts": [prompt_path.relative_to(ROOT).as_posix()],
            "display_artifacts": [row["generated_image"]],
            "expected_output": {
                "object_count": row["target_object_count"],
                "objects": expected_objects,
            },
            "model_outputs": {
                "branch": "geometry",
                "checkpoint_status": "diagnostic_not_accepted",
                "predicted_object_count": row["predicted_object_count"],
                "count_correct": row["count_correct"],
                "pairs": row["pairs"],
            },
            "is_fully_correct": row["is_fully_correct"],
            "teaching_note": example["explanation"],
        })
    canonical_payload = {
        "schema_version": 2,
        "work_package": "P2-W04",
        "formal_experiment": "third_round_geometry_auxiliary",
        "primary_branch": "geometry",
        "primary_branch_reason": "attempt-3 highest seen-single fully-correct rate; no branch passed all frozen thresholds",
        "example_count": len(canonical_examples),
        "selection_policy": "fixed 420-sample evaluation; representative correct, improved, persistent-failure, unseen-combination and multi-object cases",
        "statistics_warning": "curated examples explain behavior and do not represent the full evaluation distribution",
        "examples": canonical_examples,
    }
    (RESULT / "capability_examples.json").write_text(
        json.dumps(canonical_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metric_labels = {
        "color_correct": "颜色",
        "shape_correct": "形状类别",
        "normalized_template_iou": "轮廓IoU",
        "boundary_score": "边界分数",
        "size_correct": "大小",
        "position_correct": "位置",
    }
    canonical_lines = [
        "# P2-W04第三轮能力样例册",
        "",
        "> 当前代表分支为强几何支，因为它在第三轮已见单对象集上的完全正确率最高；但它没有通过全部冻结验收项，因此这里只把它作为诊断检查点，不称为最终模型。",
        "",
        "> 本册8个样例来自固定420样本评价的逐样本机器记录，只用于直观理解能力与失败，不代表总体分布。指标定义见`../../../docs/P2-W04_指标与术语说明.md`；四分支同提示对照见`attempt3_capability_comparison.md`。",
        "",
    ]
    for example in canonical_examples:
        row = example["model_outputs"]
        prompt_text = (showcase_dir / f"example_{example['index']:02d}_prompt.txt").read_text(
            encoding="utf-8"
        ).strip()
        first_pair = row["pairs"][0]
        target = first_pair["target"]
        prediction = first_pair["prediction"]
        prediction_text = (
            "未检出"
            if prediction is None
            else f"{prediction['color']} / {prediction['shape']} / {prediction['size']} / {prediction['position_region']}"
        )
        checks: list[str] = []
        for field, label in metric_labels.items():
            value = first_pair[field]
            if field in {"normalized_template_iou", "boundary_score"}:
                checks.append(f"{label}={float(value):.4f}")
            else:
                checks.append(f"{label}={'正确' if value else '错误'}")
        image_relative = Path(example["display_artifacts"][0]).relative_to(
            "results/P2-W04_tiny_image_ldm/revision_v2"
        ).as_posix()
        canonical_lines.extend([
            f"## {example['index']}. {example['category']}",
            "",
            f"- 源记录：`{example['source_record_id']}`（`{example['evaluation_subset']}`）",
            f"- 文本条件：{prompt_text}",
            f"- 标准首对象：{target['color']} / {target['shape']} / {target['size']} / {target['position_region']}",
            f"- 模型首对象：{prediction_text}",
            f"- 数量：目标{example['expected_output']['object_count']}，模型{row['predicted_object_count']}，{'正确' if row['count_correct'] else '错误'}",
            f"- 分项结果：{'；'.join(checks)}",
            f"- 完全正确：{'是' if example['is_fully_correct'] else '否'}",
            f"- 说明：{example['teaching_note']}",
            "",
            f"![第三轮能力样例{example['index']}]({image_relative})",
            "",
        ])
    (RESULT / "capability_examples.md").write_text(
        "\n".join(canonical_lines), encoding="utf-8"
    )


def main() -> None:
    _write_comparison()
    _write_evidence_index()
    _write_manifests()
    _write_examples()


if __name__ == "__main__":
    main()
