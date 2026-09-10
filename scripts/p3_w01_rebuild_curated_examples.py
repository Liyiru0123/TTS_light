"""Rebuild the P3-W01 I003 capability booklet with stratified examples.

This post-completion documentation repair does not retrain or reevaluate the model.
It only selects already-recorded fixed-test predictions and rebuilds their media.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.video_mllm import VideoQADataset  # noqa: E402


BRANCH = ROOT / "results/P3-W01_tiny_video_mllm"
PREDICTIONS = BRANCH / "predictions/test_predictions.jsonl"

# Curated against the fixed test set. The set covers all four question kinds,
# all three temporal relations, one real error and two informative boundaries.
SELECTION = [
    ("p3w01_test_motion_0002_down", "real_error", "真实分类错误：分类头把向下运动判断为向右，但因果Decoder生成正确。"),
    ("p3w01_test_motion_0009_left", "low_confidence_boundary", "低置信正确样例：两个输出均正确，但分类置信度较低，适合讨论不确定性。"),
    ("p3w01_test_appear_order_0014_first", "type_coverage", "出现顺序样例：前一个对象先出现。"),
    ("p3w01_test_appear_order_0015_second", "type_coverage", "出现顺序样例：后一个对象先出现。"),
    ("p3w01_test_disappear_order_0002_first", "type_coverage", "消失顺序样例：前一个对象先消失。"),
    ("p3w01_test_move_start_order_0009_first", "relation_coverage", "开始移动顺序样例：前一个对象更早开始移动。"),
    ("p3w01_test_move_start_order_0022_second", "relation_coverage", "开始移动顺序样例：后一个对象更早开始移动。"),
    ("p3w01_test_move_start_order_0021_same", "semantic_boundary_simultaneous_onset", "两个对象在同一帧开始移动，是先后关系中的相等边界。"),
]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    predictions = {
        row["record_id"]: row
        for line in PREDICTIONS.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))
    }
    missing = [record_id for record_id, _, _ in SELECTION if record_id not in predictions]
    if missing:
        raise RuntimeError(f"fixed-test predictions are missing curated records: {missing}")

    dataset = VideoQADataset(ROOT, data_dir="data/video_qa/p3_w01_v1", splits="test")
    row_index = {row["record_id"]: index for index, row in enumerate(dataset.rows)}
    media_dir = BRANCH / "examples/media"
    media_dir.mkdir(parents=True, exist_ok=True)
    examples = []

    for record_id, selection_type, teaching_note in SELECTION:
        item = predictions[record_id]
        frames = dataset.videos[row_index[record_id]].astype(np.uint8)
        stem = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:12]
        gif_path = media_dir / f"{stem}.gif"
        montage_path = media_dir / f"{stem}_montage.png"
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=250, loop=0)
        montage = Image.new("RGB", (64 * len(images), 64), (255, 255, 255))
        for frame_index, image in enumerate(images):
            montage.paste(image, (64 * frame_index, 0))
        montage.save(montage_path)

        relative_base = "results/P3-W01_tiny_video_mllm/examples/media"
        examples.append({
            **{key: item.get(key) for key in (
                "record_id", "question_kind", "classification_correct", "generation_correct",
                "is_fully_correct", "classification_confidence", "predicted_move_starts",
                "reference_move_starts", "onset_relation_correct", "relation_attention_top_memory_index",
            )},
            "question": item["question_zh"],
            "expected_output": item["reference_answer"],
            "classification_output": item["classification_answer"],
            "generated_output": item["generated_answer"],
            "input_artifact": f"{relative_base}/{stem}.gif",
            "frame_montage": f"{relative_base}/{stem}_montage.png",
            "source_record": "data/video_qa/p3_w01_v1/records.jsonl",
            "source_video_cache": "data/video_qa/p3_w01_v1/videos.npz",
            "selection_type": selection_type,
            "teaching_note": teaching_note,
        })

    boundary_types = {"real_error", "low_confidence_boundary", "semantic_boundary_simultaneous_onset"}
    payload = {
        "schema_version": 2,
        "work_package": "P3-W01",
        "iteration_id": "I003",
        "branch_id": "spacetime_queries",
        "selection_policy": "按题型、答案关系、正确性和教学边界分层策展，不按单一置信度排序",
        "example_count": len(examples),
        "correct_count": sum(item["is_fully_correct"] for item in examples),
        "error_or_boundary_count": sum(item["selection_type"] in boundary_types for item in examples),
        "question_kind_counts": {
            kind: sum(item["question_kind"] == kind for item in examples)
            for kind in sorted({item["question_kind"] for item in examples})
        },
        "examples": examples,
    }
    write_json(BRANCH / "examples/capability_examples.json", payload)

    lines = [
        "# P3-W01 I003 TinyVideoMLLM直观能力样例", "",
        "> 本册从固定测试集分层策展8条样例，覆盖全部4类问题、先/后/同时三种时序关系、1个真实错误和2个教学边界。它用于直观理解能力，不替代完整测试统计。", "",
    ]
    for number, item in enumerate(examples, 1):
        stem = Path(item["input_artifact"]).stem
        lines += [
            f"## 样例{number}：{item['record_id']}", "",
            f"![输入GIF动画](media/{stem}.gif)", "",
            f"[单独打开GIF动画](media/{stem}.gif)", "",
            f"![8帧蒙太奇](media/{stem}_montage.png)", "",
            f"[单独打开8帧蒙太奇](media/{stem}_montage.png)", "",
            f"- 问题类型：`{item['question_kind']}`",
            f"- 问题：{item['question']}",
            f"- 标准答案：{item['expected_output']}",
            f"- 分类输出：{item['classification_output']}（正确：{item['classification_correct']}；置信度：{item['classification_confidence']:.4f}）",
            f"- Decoder输出：{item['generated_output']}（正确：{item['generation_correct']}）",
            f"- 预测/真实运动起点：{item['predicted_move_starts']} / {item['reference_move_starts']}",
            f"- 选取理由：`{item['selection_type']}`",
            f"- 教学说明：{item['teaching_note']}", "",
        ]
    (BRANCH / "examples/capability_examples.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
