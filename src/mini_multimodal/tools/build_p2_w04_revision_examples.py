"""Build direct historical-checkpoint examples under the calibrated v2 evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _choose(
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    row = next(
        item for item in rows if item["record_id"] not in selected_ids and predicate(item)
    )
    selected_ids.add(row["record_id"])
    return row


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    result_dir = root / "results" / "P2-W04_tiny_image_ldm" / "revision_v2"
    rows = [
        json.loads(line)
        for line in (result_dir / "original_checkpoint_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    selected: list[tuple[str, str, dict[str, Any]]] = []
    selected_ids: set[str] = set()
    for row in [item for item in rows if item["is_fully_correct"]][:4]:
        selected_ids.add(row["record_id"])
        selected.append(("correct", "数量、颜色、形状、轮廓、大小和位置均满足修订判据。", row))
    selectors = (
        ("position_error", "颜色和形状可辨，但位置区域错误，说明原模型没有可靠空间接地。", lambda r: any(p["color_correct"] and p["shape_correct"] and not p["position_correct"] for p in r["pairs"])),
        ("shape_error", "颜色可能正确，但形状类别或轮廓未通过，展示硬分类与连续轮廓的联合要求。", lambda r: any(not p["shape_correct"] for p in r["pairs"])),
        ("fragmentation_or_count_error", "生成图出现多余碎片或对象数错误，不能因某个主色块正确而判为成功。", lambda r: not r["count_correct"] and r["probe"]["predicted_object_count"] > r["target_object_count"]),
        ("multi_object_boundary", "多对象输入用于展示当前模型的数量、匹配和属性绑定边界。", lambda r: r["subset"].startswith("multi_") and not r["is_fully_correct"]),
    )
    for category, note, predicate in selectors:
        selected.append((category, note, _choose(rows, selected_ids, predicate)))
    examples = []
    markdown = [
        "# P2-W04原版检查点：修订评价器能力样例",
        "",
        "> 这些样例使用修订评价器重算原版检查点。原训练数据含组合留出，因此任何`unseen_single`标签都不能解释为真正的未见组合泛化。",
        "",
    ]
    for index, (category, note, row) in enumerate(selected, start=1):
        relative_from_md = Path(row["generated_image"]).relative_to(
            Path("results/P2-W04_tiny_image_ldm/revision_v2")
        ).as_posix()
        expected = row["pairs"][0]["target"] if len(row["pairs"]) == 1 else [
            pair["target"] for pair in row["pairs"]
        ]
        outputs = {
            "predicted_object_count": row["predicted_object_count"],
            "pairs": row["pairs"],
        }
        examples.append({
            "example_id": f"old_v2_{index:02d}",
            "category": category,
            "source_record_id": row["record_id"],
            "subset": row["subset"],
            "prompt": row["prompt"],
            "input_artifacts": ["data/ldm/p2_w04_eval_v2/records.jsonl"],
            "display_artifacts": [row["generated_image"]],
            "expected_output": expected,
            "model_outputs": outputs,
            "is_fully_correct": row["is_fully_correct"],
            "teaching_note": note,
        })
        markdown.extend([
            f"## 样例{index}：{category}",
            "",
            f"- 记录：`{row['record_id']}`；集合：`{row['subset']}`",
            f"- 条件：{row['prompt']}",
            f"- 目标对象：`{json.dumps(expected, ensure_ascii=False)}`",
            f"- 预测对象数：{row['predicted_object_count']}；完全正确：`{str(row['is_fully_correct']).lower()}`",
            f"- 说明：{note}",
            "",
            f"![{row['record_id']}]({relative_from_md})",
            "",
        ])
    payload = {
        "schema_version": 1,
        "status": "historical_checkpoint_diagnostic_only",
        "example_count": len(examples),
        "examples": examples,
    }
    _write_json(result_dir / "original_checkpoint_examples.json", payload)
    (result_dir / "original_checkpoint_examples.md").write_text(
        "\n".join(markdown), encoding="utf-8", newline="\n"
    )
    print(json.dumps({"example_count": len(examples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
