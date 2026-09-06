from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


SELECTED = (
    (
        "normal:test:core_scene_000700:q:count:f0",
        "正确：单对象数量题，分类与文本生成一致。",
    ),
    (
        "normal:test:core_scene_000701:q:horizontal_relation:obj_0:obj_1:f0",
        "正确：需要同时定位甲、乙并判断水平方向关系。",
    ),
    (
        "compositional:test:composition_scene_000540:q:object_shape:obj_0:f0",
        "正确：未见组合测试样例，展示组合泛化能力。",
    ),
    (
        "counterfactual:test:shortcut_scene_000445:q:object_shape:obj_0:f0",
        "正确：反事实样例中背景捷径被打破，模型仍输出正确形状。",
    ),
    (
        "normal:test:core_scene_000700:q:object_color:obj_0:f0",
        "错误：分类头正确，但文本Decoder把黄色生成成橙色，直观说明两条输出路径并不等价。",
    ),
    (
        "normal:test:core_scene_000701:q:object_shape:obj_0:f0",
        "错误：分类头和文本Decoder都把矩形判断为五边形。",
    ),
    (
        "normal:test:core_scene_000704:q:vertical_relation:obj_0:obj_2:f0",
        "错误：分类与接地关系判断正确，但文本Decoder生成了相反方向。",
    ),
    (
        "normal:test:core_scene_000720:q:count:f0",
        "错误：分类头和文本Decoder都把2个目标判断成1个。",
    ),
)


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["record_id"]] = row
    return rows


def locate_image(image_root: Path, scene_id: str) -> Path:
    matches = list(image_root.rglob(f"{scene_id}_f000.png"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one image for {scene_id}, found {len(matches)}")
    return matches[0]


def build(project_root: Path) -> dict[str, Any]:
    results_dir = project_root / "results" / "P2-W03_tiny_vlm"
    output_dir = results_dir / "capability_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_predictions(results_dir / "predictions.jsonl")
    image_root = project_root / "data" / "images" / "p1_w04_v1" / "images"

    examples = []
    for index, (record_id, teaching_note) in enumerate(SELECTED, start=1):
        row = predictions[record_id]
        source_image = locate_image(image_root, row["scene_id"])
        display_name = f"example_{index:02d}_{row['scene_id']}.png"
        display_path = output_dir / display_name
        with Image.open(source_image) as image:
            source = image.convert("RGB")
            if source.size != (64, 64):
                raise ValueError(f"unexpected source size: {source_image} {source.size}")
            source.resize((256, 256), Image.Resampling.NEAREST).save(display_path)
        examples.append(
            {
                "index": index,
                "source_record_id": record_id,
                "record_id": record_id,
                "scene_id": row["scene_id"],
                "variant": row["variant"],
                "question_kind": row["question_kind"],
                "object_count": row["object_count"],
                "question": row["question_zh"],
                "target_answer": row["answer_zh"],
                "classification_prediction": row["classification_prediction"],
                "classification_correct": row["classification_correct"],
                "generation_prediction": row["generation_prediction"],
                "generation_correct": row["generation_correct"],
                "generated_token_probabilities": row["generated_token_probabilities"],
                "source_image": source_image.relative_to(project_root).as_posix(),
                "display_image": display_path.relative_to(project_root).as_posix(),
                "input_summary": {"question": row["question_zh"]},
                "input_artifacts": [source_image.relative_to(project_root).as_posix()],
                "display_artifacts": [display_path.relative_to(project_root).as_posix()],
                "expected_output": row["answer_zh"],
                "model_outputs": {
                    "classification": row["classification_prediction"],
                    "generation": row["generation_prediction"],
                },
                "is_fully_correct": bool(
                    row["classification_correct"] and row["generation_correct"]
                ),
                "teaching_note": teaching_note,
            }
        )

    payload = {
        "schema_version": 1,
        "work_package": "P2-W03",
        "source_predictions": "results/P2-W03_tiny_vlm/predictions.jsonl",
        "selection_policy": "固定人工策展：4个正确样例与4个具有不同失败模式的错误样例；不代表总体统计分布",
        "display_policy": "源图为模型实际使用的64×64图像；展示图仅用最近邻放大到256×256，不改变内容",
        "example_count": len(examples),
        "correct_both_count": sum(
            item["classification_correct"] and item["generation_correct"]
            for item in examples
        ),
        "has_error_count": sum(
            not (item["classification_correct"] and item["generation_correct"])
            for item in examples
        ),
        "examples": examples,
    }
    (results_dir / "capability_examples.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# P2-W03 TinyVLM能力样例册",
        "",
        "> 本页是最终测试集逐样本预测的固定策展视图，不替代总体统计指标。源图是模型实际接收的64×64统一世界图像；页面中的图像仅以最近邻放大到256×256，便于观察。",
        "",
        "## 输入与输出要求",
        "",
        "- 输入图像：RGB、64×64、本项目统一世界图像；原图推理时先由P2-W02 VAE编码为4×16×16潜变量。",
        "- 输入问题：字符词表覆盖的中文短问题模板，必须能够在图中唯一指代目标甲和可选参照乙。",
        "- 分类输出：从固定21个答案类别中选择一个。",
        "- 生成输出：字符级因果Decoder从BOS开始逐Token生成，遇到EOS或8个Token停止。",
        "- 本模型不面向任意自然图像、开放词汇问题或长文本回答。",
        "",
        "## 样例总览",
        "",
        "| 样例 | 数据变体 | 题型 | 分类 | 生成 | 总体观察 |",
        "|---:|---|---|---|---|---|",
    ]
    for item in examples:
        classification = "✓" if item["classification_correct"] else "✗"
        generation = "✓" if item["generation_correct"] else "✗"
        overall = "正确" if classification == generation == "✓" else "含错误"
        lines.append(
            f"| {item['index']} | {item['variant']} | {item['question_kind']} | "
            f"{classification} | {generation} | {overall} |"
        )

    for section_title, predicate in (
        ("## 正确样例", lambda item: item["classification_correct"] and item["generation_correct"]),
        ("## 错误与分歧样例", lambda item: not (item["classification_correct"] and item["generation_correct"])),
    ):
        lines.extend(("", section_title, ""))
        for item in filter(predicate, examples):
            relative_display = Path(item["display_image"]).relative_to(
                Path("results/P2-W03_tiny_vlm")
            ).as_posix()
            lines.extend(
                (
                    f"### 样例{item['index']}：{item['scene_id']}",
                    "",
                    f"![样例{item['index']}输入图像]({relative_display})",
                    "",
                    f"- 问题：{item['question']}",
                    f"- 标准答案：`{item['target_answer']}`",
                    f"- 分类输出：`{item['classification_prediction']}`（{'正确' if item['classification_correct'] else '错误'}）",
                    f"- 文本Decoder输出：`{item['generation_prediction']}`（{'正确' if item['generation_correct'] else '错误'}）",
                    "- 生成Token概率：`"
                    + ", ".join(f"{value:.4f}" for value in item["generated_token_probabilities"])
                    + "`",
                    f"- 观察：{item['teaching_note']}",
                    f"- 记录ID：`{item['record_id']}`",
                    f"- 64×64源图：`{item['source_image']}`",
                    "",
                )
            )

    lines.extend(
        (
            "## 如何核对",
            "",
            "本页文字来自`capability_examples.json`，后者逐项引用`predictions.jsonl`中的固定记录。自动测试会检查答案、正确性标记、源图、展示图和记录ID一致。若重新训练模型，应重新生成本样例册，不能保留与新模型不一致的旧样例。",
            "",
        )
    )
    (results_dir / "capability_examples.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    payload = build(project_root)
    print(
        json.dumps(
            {
                "example_count": payload["example_count"],
                "correct_both_count": payload["correct_both_count"],
                "has_error_count": payload["has_error_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
