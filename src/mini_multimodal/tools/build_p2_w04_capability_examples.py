"""Build a traceable correct/error text-to-image capability showcase."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    result_dir = root / "results" / "P2-W04_tiny_image_ldm"
    rows = [
        json.loads(line)
        for line in (result_dir / "generation_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    correct = [row for row in rows if row["is_fully_correct"]]
    errors = [row for row in rows if not row["is_fully_correct"]]
    selected = correct[:4] + errors[:4]
    if len(correct) < 4 or len(errors) < 4:
        raise RuntimeError("capability showcase needs four correct and four error examples")
    asset_dir = result_dir / "capability_examples"
    asset_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for index, row in enumerate(selected, start=1):
        source = Path(row["generated_image"])
        display = asset_dir / f"example_{index:02d}_{row['scene_id']}.png"
        with Image.open(source) as image:
            image.convert("RGB").resize((256, 256), Image.Resampling.NEAREST).save(display)
        prompt_file = asset_dir / f"example_{index:02d}_prompt.txt"
        prompt_file.write_text(row["prompt"] + "\n", encoding="utf-8")
        match = row["semantic_match"]
        if row["is_fully_correct"]:
            note = "成功：对象数量、颜色和形状探针均与文本条件一致。"
        else:
            failures = []
            if not row["probe"]["foreground_present"]:
                failures.append("未检测到前景")
            if not match["count_correct"]:
                failures.append("对象数量不一致")
            if match["color_accuracy"] < 1:
                failures.append("颜色不完全一致")
            if match["shape_accuracy"] < 1:
                failures.append("形状不完全一致")
            note = "失败/边界：" + "、".join(failures) + "。"
        relative_source = source.relative_to(root).as_posix()
        relative_display = display.relative_to(root).as_posix()
        relative_prompt = prompt_file.relative_to(root).as_posix()
        examples.append({
            "index": index,
            "source_record_id": row["record_id"],
            "record_id": row["record_id"],
            "scene_id": row["scene_id"],
            "variant": row["variant"],
            "input_summary": {"prompt": row["prompt"], "seed": row["sample_seed"], "guidance_scale": row["guidance_scale"]},
            "input_artifacts": [relative_prompt],
            "display_artifacts": [relative_display],
            "source_generated_image": relative_source,
            "display_image": relative_display,
            "expected_output": {"objects": row["target_objects"]},
            "model_outputs": {"probe": row["probe"], "semantic_match": match},
            "is_fully_correct": row["is_fully_correct"],
            "teaching_note": note,
        })
    payload = {
        "schema_version": 1,
        "work_package": "P2-W04",
        "source_predictions": "results/P2-W04_tiny_image_ldm/generation_predictions.jsonl",
        "selection_policy": "固定选择4个完全正确样例与4个不同失败样例；不代表总体分布",
        "display_policy": "生成原图为64×64；展示图仅最近邻放大至256×256",
        "example_count": len(examples),
        "correct_count": sum(item["is_fully_correct"] for item in examples),
        "error_count": sum(not item["is_fully_correct"] for item in examples),
        "examples": examples,
    }
    (result_dir / "capability_examples.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P2-W04 TinyImageLDM能力样例册",
        "",
        "> 本页展示固定测试提示在固定噪声和CFG设置下的真实生成结果。它用于直观理解能力与失败，不代替完整测试集统计。原图为64×64；页面只做最近邻放大。",
        "",
        "## 输入与输出要求",
        "",
        "- 输入：项目有限词表覆盖的中文场景描述、随机种子、采样步数和引导强度。",
        "- 输出：64×64 RGB图像、26个潜变量轨迹状态及逐步Cross-Attention。",
        "- 正确性：机器视觉探针比较文本真值与生成图中的对象数量、颜色和形状。探针本身也有边界，因此应结合图像观察。",
        "",
    ]
    for title, predicate in (
        ("## 正确样例", lambda item: item["is_fully_correct"]),
        ("## 错误与边界样例", lambda item: not item["is_fully_correct"]),
    ):
        lines.extend((title, ""))
        for item in filter(predicate, examples):
            display = Path(item["display_image"]).relative_to(Path("results/P2-W04_tiny_image_ldm")).as_posix()
            target = item["expected_output"]["objects"]
            predicted = item["model_outputs"]["probe"]["objects"]
            lines.extend((
                f"### 样例{item['index']}：{item['scene_id']}",
                "",
                f"![样例{item['index']}生成图]({display})",
                "",
                f"- 文本条件：{item['input_summary']['prompt']}",
                f"- 期望对象：`{json.dumps(target, ensure_ascii=False)}`",
                f"- 图像探针：`{json.dumps(predicted, ensure_ascii=False)}`",
                f"- 观察：{item['teaching_note']}",
                f"- 记录ID：`{item['source_record_id']}`",
                f"- 64×64原始生成图：`{item['source_generated_image']}`",
                "",
            ))
    lines.extend((
        "## 如何核对",
        "",
        "样例来自`generation_predictions.jsonl`，机器清单为`capability_examples.json`。自动测试会核对提示、真值、探针输出、正确性标记和图像资产；重新训练后必须重新生成本页。",
        "",
    ))
    (result_dir / "capability_examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"example_count": 8, "correct_count": 4, "error_count": 4}, ensure_ascii=False))


if __name__ == "__main__":
    main()
