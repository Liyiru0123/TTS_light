from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("overfit_classification", "48条过拟合分类准确率", "overfit_report.json", "$.greedy_classification_accuracy", "COMP-VLM-001", "固定48条过拟合小集"),
    ("overfit_generation", "48条过拟合生成准确率", "overfit_report.json", "$.greedy_generation_accuracy", "COMP-VLM-001", "固定48条过拟合小集"),
    ("classification_test", "分类VQA测试准确率", "classification_metrics.json", "$.classification_accuracy", "MODEL-VLM-001", "固定测试集"),
    ("classification_multi_object", "多对象分类准确率", "classification_metrics.json", "$.groups.object_group.multi_object.classification_accuracy", "MODEL-VLM-001", "固定测试集多对象子集"),
    ("grounding_relation", "甲乙质心直接关系判断准确率", "classification_metrics.json", "$.grounding_relation_accuracy", "MODEL-VLM-001", "固定测试集关系题子集"),
    ("generation_test", "因果生成测试准确率", "generation_metrics.json", "$.generation_accuracy", "MODEL-VLM-002", "固定测试集"),
    ("generation_color", "颜色题生成准确率（六类最低）", "generation_metrics.json", "$.groups.question_kind.object_color.generation_accuracy", "MODEL-VLM-002", "固定测试集颜色题子集"),
    ("generation_horizontal", "水平关系生成准确率", "generation_metrics.json", "$.groups.question_kind.horizontal_relation.generation_accuracy", "MODEL-VLM-002", "固定测试集水平关系子集"),
    ("generation_vertical", "垂直关系生成准确率", "generation_metrics.json", "$.groups.question_kind.vertical_relation.generation_accuracy", "MODEL-VLM-002", "固定测试集垂直关系子集"),
    ("generation_compositional", "未见组合生成准确率", "generation_metrics.json", "$.groups.variant.compositional.generation_accuracy", "MODEL-VLM-003", "固定测试集未见组合子集"),
    ("generation_counterfactual", "反事实生成准确率", "generation_metrics.json", "$.groups.variant.counterfactual.generation_accuracy", "MODEL-VLM-003", "固定测试集反事实子集"),
    ("generation_multi_object", "多对象生成准确率", "generation_metrics.json", "$.groups.object_group.multi_object.generation_accuracy", "MODEL-VLM-002", "固定测试集多对象子集"),
    ("text_only", "文本单模态分类准确率", "single_modality_report.json", "$.text_only_classification_accuracy", "PED-VLM-001", "固定测试集"),
    ("image_only", "图像单模态分类准确率", "single_modality_report.json", "$.image_only_classification_accuracy", "PED-VLM-001", "固定测试集"),
    ("multimodal_classification", "完整多模态分类准确率", "single_modality_report.json", "$.multimodal_classification_accuracy", "PED-VLM-001", "固定测试集"),
    ("multimodal_advantage", "多模态相对最佳单模态优势", "single_modality_report.json", "$.advantage", "PED-VLM-001", "固定测试集差值"),
    ("ablation_image_masked", "图像遮蔽后生成准确率", "ablation_report.json", "$.ablations.image_masked.generation_accuracy", "PED-VLM-002", "固定测试集图像遮蔽评测"),
    ("ablation_text_masked", "文本遮蔽后生成准确率", "ablation_report.json", "$.ablations.text_masked.generation_accuracy", "PED-VLM-002", "固定测试集文本遮蔽评测"),
    ("ablation_fusion_disabled", "阻断融合后生成准确率", "ablation_report.json", "$.ablations.fusion_disabled.generation_accuracy", "PED-VLM-002", "固定测试集融合阻断评测"),
    ("relation_generation", "位置消融前关系题生成准确率", "ablation_report.json", "$.relation_generation_accuracy", "PED-VLM-003", "固定测试集关系题子集"),
    ("ablation_position_relation", "完全位置消融后关系题生成准确率", "ablation_report.json", "$.no_position_relation_generation_accuracy", "PED-VLM-003", "固定测试集关系题位置消融"),
    ("position_relation_drop", "位置消融关系题准确率下降", "ablation_report.json", "$.position_relation_drop", "PED-VLM-003", "固定测试集关系题差值"),
    ("shortcut_iid", "背景捷径IID验证准确率", "shortcut_report.json", "$.shortcut_iid_validation.accuracy", "PED-VLM-004", "捷径IID验证集"),
    ("shortcut_counterfactual", "背景捷径反事实测试准确率", "shortcut_report.json", "$.counterfactual_test.accuracy", "PED-VLM-004", "反事实测试集"),
    ("main_counterfactual_generation", "主模型反事实生成准确率", "shortcut_report.json", "$.main_counterfactual_generation_accuracy", "PED-VLM-004", "固定测试集反事实子集"),
)

SAMPLE_COUNT_SPECS = {
    "overfit_classification": ("overfit_report.json", ("$.count",), "value"),
    "overfit_generation": ("overfit_report.json", ("$.count",), "value"),
    "classification_test": ("data_readiness_report.json", ("$.split_counts.test",), "value"),
    "classification_multi_object": ("classification_metrics.json", ("$.groups.object_group.multi_object.count",), "value"),
    "grounding_relation": ("classification_metrics.json", ("$.groups.question_kind.horizontal_relation.count", "$.groups.question_kind.vertical_relation.count"), "sum"),
    "generation_test": ("data_readiness_report.json", ("$.split_counts.test",), "value"),
    "generation_color": ("generation_metrics.json", ("$.groups.question_kind.object_color.count",), "value"),
    "generation_horizontal": ("generation_metrics.json", ("$.groups.question_kind.horizontal_relation.count",), "value"),
    "generation_vertical": ("generation_metrics.json", ("$.groups.question_kind.vertical_relation.count",), "value"),
    "generation_compositional": ("generation_metrics.json", ("$.groups.variant.compositional.count",), "value"),
    "generation_counterfactual": ("generation_metrics.json", ("$.groups.variant.counterfactual.count",), "value"),
    "generation_multi_object": ("generation_metrics.json", ("$.groups.object_group.multi_object.count",), "value"),
    "text_only": ("data_readiness_report.json", ("$.split_counts.test",), "value"),
    "image_only": ("data_readiness_report.json", ("$.split_counts.test",), "value"),
    "multimodal_classification": ("data_readiness_report.json", ("$.split_counts.test",), "value"),
    "multimodal_advantage": ("data_readiness_report.json", ("$.split_counts.test",), "value"),
    "ablation_image_masked": ("data_readiness_report.json", ("$.split_counts.test",), "value"),
    "ablation_text_masked": ("data_readiness_report.json", ("$.split_counts.test",), "value"),
    "ablation_fusion_disabled": ("data_readiness_report.json", ("$.split_counts.test",), "value"),
    "relation_generation": ("generation_metrics.json", ("$.groups.question_kind.horizontal_relation.count", "$.groups.question_kind.vertical_relation.count"), "sum"),
    "ablation_position_relation": ("generation_metrics.json", ("$.groups.question_kind.horizontal_relation.count", "$.groups.question_kind.vertical_relation.count"), "sum"),
    "position_relation_drop": ("generation_metrics.json", ("$.groups.question_kind.horizontal_relation.count", "$.groups.question_kind.vertical_relation.count"), "sum"),
    "shortcut_iid": ("shortcut_report.json", ("$.shortcut_iid_validation.count",), "value"),
    "shortcut_counterfactual": ("shortcut_report.json", ("$.counterfactual_test.count",), "value"),
    "main_counterfactual_generation": ("generation_metrics.json", ("$.groups.variant.counterfactual.count",), "value"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_json_path(data: Any, json_path: str) -> Any:
    if not json_path.startswith("$."):
        raise ValueError(f"Unsupported JSON path: {json_path}")
    value = data
    for key in json_path[2:].split("."):
        value = value[key]
    return value


def build_index(results_dir: Path) -> dict[str, Any]:
    cache: dict[str, dict[str, Any]] = {}
    entries = []
    for metric_id, label, source_file, json_path, acceptance_item_id, evaluation_scope in METRICS:
        source = cache.setdefault(source_file, read_json(results_dir / source_file))
        value = resolve_json_path(source, json_path)
        count_source_file, count_paths, count_aggregation = SAMPLE_COUNT_SPECS[metric_id]
        count_source = cache.setdefault(
            count_source_file, read_json(results_dir / count_source_file)
        )
        count_values = [int(resolve_json_path(count_source, path)) for path in count_paths]
        sample_count = count_values[0] if count_aggregation == "value" else sum(count_values)
        entries.append(
            {
                "id": metric_id,
                "label": label,
                "value": value,
                "display_4dp": f"{value:.4f}",
                "source_file": source_file,
                "json_path": json_path,
                "evaluation_scope": evaluation_scope,
                "sample_count": sample_count,
                "sample_count_source_file": count_source_file,
                "sample_count_json_paths": list(count_paths),
                "sample_count_aggregation": count_aggregation,
                "acceptance_item_id": acceptance_item_id,
            }
        )
    return {
        "schema_version": "1.0",
        "purpose": "P2-W03总结报告核心精度与原始机器结果之间的可追溯索引",
        "value_policy": "value保留原始JSON浮点值；报告显示值按四位小数四舍五入",
        "metrics": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the P2-W03 metric evidence index.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/P2-W03_tiny_vlm"),
    )
    args = parser.parse_args()
    index = build_index(args.results_dir)
    output = args.results_dir / "evidence_index.json"
    output.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(output), "metric_count": len(index["metrics"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
