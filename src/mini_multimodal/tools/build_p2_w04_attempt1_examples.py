from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/P2-W04_tiny_image_ldm/revision_v2"


def compact(row: dict, category: str, explanation: str) -> dict:
    return {
        "category": category,
        "explanation": explanation,
        "record_id": row["record_id"],
        "subset": row["subset"],
        "prompt": row["prompt"],
        "generated_image": row["generated_image"],
        "target_object_count": row["target_object_count"],
        "predicted_object_count": row["predicted_object_count"],
        "count_correct": row["count_correct"],
        "is_fully_correct": row["is_fully_correct"],
        "pairs": row["pairs"],
    }


def main() -> None:
    rows = [json.loads(line) for line in (RESULT / "generation_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    selected: list[dict] = []
    used: set[str] = set()

    for shape in ("circle", "triangle", "rectangle"):
        row = next(
            row for row in rows
            if row["subset"] == "seen_single"
            and row["is_fully_correct"]
            and row["pairs"][0]["target"]["shape"] == shape
        )
        selected.append(compact(row, "完整正确", f"{shape}样例满足数量、颜色、形状、轮廓、大小和位置全部判据。"))
        used.add(row["record_id"])

    predicates = [
        ("正方形错误", "位置和颜色正确，但形状没有被可靠生成为正方形。", lambda r: r["subset"] == "seen_single" and r["pairs"][0]["target"]["shape"] == "square" and r["pairs"][0]["position_correct"] and not r["pairs"][0]["shape_correct"]),
        ("碎片/数量错误", "主对象可见，但额外彩色碎片使对象数量判错。", lambda r: r["subset"] == "seen_single" and not r["count_correct"]),
        ("大小错误", "位置和颜色正确，但大小类别不符合文本。", lambda r: r["subset"] == "seen_single" and r["pairs"][0]["position_correct"] and r["pairs"][0]["color_correct"] and not r["pairs"][0]["size_correct"]),
        ("未见组合错误", "训练侧组合隔离有效，但该未见颜色—形状组合未达到完整正确。", lambda r: r["subset"] == "unseen_single" and not r["is_fully_correct"]),
        ("多对象边界", "三对象提示用于展示第一轮模型的数量和属性绑定边界。", lambda r: r["subset"] == "multi_3" and not r["is_fully_correct"]),
    ]
    for category, explanation, predicate in predicates:
        row = next(row for row in rows if row["record_id"] not in used and predicate(row))
        selected.append(compact(row, category, explanation))
        used.add(row["record_id"])

    (RESULT / "attempt1_capability_examples.json").write_text(
        json.dumps({"schema_version": 1, "work_package": "P2-W04", "attempt": 1, "examples": selected}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# P2-W04第一轮修订模型能力样例",
        "",
        "> 本册来自`generation_predictions.jsonl`，用于直观理解第一轮模型；它没有通过最终验收。",
        "",
    ]
    for index, row in enumerate(selected, 1):
        relative_image = Path(row["generated_image"]).relative_to("results/P2-W04_tiny_image_ldm/revision_v2").as_posix()
        pair = row["pairs"][0]
        prediction = pair["prediction"]
        predicted_text = "未检出" if prediction is None else f"{prediction['color']} / {prediction['shape']} / {prediction['size']} / {prediction['position_region']}"
        target = pair["target"]
        lines.extend([
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
        ])
    (RESULT / "attempt1_capability_examples.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
