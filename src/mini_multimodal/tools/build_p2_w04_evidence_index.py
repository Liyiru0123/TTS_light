"""Build field-level evidence mappings for P2-W04 quantitative conclusions."""

from __future__ import annotations

import json
from pathlib import Path


SPECS = (
    ("data_extension", "扩展训练对", "data_readiness_report.json", "$.extension_count", "P2-W04训练扩展池", "$.extension_count", "DATA-LDM-001"),
    ("data_train_total", "合并训练对", "data_readiness_report.json", "$.combined_split_counts.train", "P2-W04训练池", "$.combined_split_counts.train", "DATA-LDM-001"),
    ("overfit_ratio", "小集损失比", "overfit_report.json", "$.loss_ratio", "固定过拟合小集", "$.sample_count", "COMP-LDM-001"),
    ("test_noise_mse", "测试噪声MSE", "denoising_metrics.json", "$.noise_mse", "固定测试集均匀时间步", "$.sample_count", "MODEL-LDM-001"),
    ("cross_attention_increase", "关闭Cross-Attention后的MSE增加", "denoising_metrics.json", "$.cross_attention_mse_increase", "固定测试集同噪声消融", "$.sample_count", "UNIT-LDM-002"),
    ("single_color", "单对象颜色准确率", "generation_metrics.json", "$.single_object.color_accuracy", "固定单对象测试提示", "$.single_object.sample_count", "MODEL-LDM-002"),
    ("single_shape", "单对象形状准确率", "generation_metrics.json", "$.single_object.shape_accuracy", "固定单对象测试提示", "$.single_object.sample_count", "MODEL-LDM-002"),
    ("single_foreground", "单对象前景存在率", "generation_metrics.json", "$.single_object.foreground_presence", "固定单对象测试提示", "$.single_object.sample_count", "MODEL-LDM-002"),
    ("composition_color", "未见组合颜色准确率", "generation_metrics.json", "$.unseen_composition.color_accuracy", "固定未见组合提示", "$.unseen_composition.sample_count", "MODEL-LDM-002"),
    ("composition_shape", "未见组合形状准确率", "generation_metrics.json", "$.unseen_composition.shape_accuracy", "固定未见组合提示", "$.unseen_composition.sample_count", "MODEL-LDM-002"),
    ("conditioning_advantage", "条件相对最佳对照优势", "conditioning_ablation.json", "$.advantage_over_best_unconditional_or_shuffled", "124条固定提示同噪声消融", "$.conditional_cfg.sample_count", "PED-LDM-001"),
    ("cfg_semantic", "CFG语义平均分", "conditioning_ablation.json", "$.conditional_cfg.semantic_mean", "124条固定提示", "$.conditional_cfg.sample_count", "PED-LDM-002"),
)


def resolve(value: object, path: str) -> object:
    result = value
    for key in path[2:].split("."):
        result = result[key]  # type: ignore[index]
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    directory = root / "results" / "P2-W04_tiny_image_ldm"
    metrics = []
    for metric_id, label, source_file, value_path, scope, count_path, acceptance in SPECS:
        source = json.loads((directory / source_file).read_text(encoding="utf-8"))
        value = resolve(source, value_path)
        count = int(resolve(source, count_path))
        metrics.append({
            "id": metric_id,
            "label": label,
            "value": value,
            "display_4dp": f"{float(value):.4f}",
            "source_file": source_file,
            "json_path": value_path,
            "evaluation_scope": scope,
            "sample_count": count,
            "sample_count_source_file": source_file,
            "sample_count_json_paths": [count_path],
            "sample_count_aggregation": "value",
            "acceptance_item_id": acceptance,
        })
    payload = {"schema_version": 1, "work_package": "P2-W04", "metric_count": len(metrics), "metrics": metrics}
    (directory / "evidence_index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metric_count": len(metrics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
