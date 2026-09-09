"""Run one frozen I008 current-candidate inference example on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mini_multimodal.ldm.geometry_evaluation import load_attempt3_checkpoint  # noqa: E402
from mini_multimodal.ldm.revision_training import load_config  # noqa: E402
from mini_multimodal.training import load_checkpoint  # noqa: E402
from mini_multimodal.training.baseline import TinyConvClassifier  # noqa: E402
from mini_multimodal.video_gen.controlled_motion import GeneratedAppearanceMotionModule  # noqa: E402
from mini_multimodal.video_gen.evaluation import evaluate_generated_video  # noqa: E402
from p3_w04_pf005_true_independent_ldm import frame_prompt  # noqa: E402
from p3_w04_pf011_generated_anchor_motion import (  # noqa: E402
    generated_anchor_components,
    sample,
    select_cases,
    target_for,
)


CONFIG = ROOT / "configs/p3_w04_i008_generated_motion.json"
CHECKPOINT = ROOT / (
    "results/P3-W04_video_generation/iterations/I008/branches/"
    "generated_motion_module/checkpoints/best.pt"
)
DEFAULT_RECORD = "new:test:p3w04_test_000002"
LABELS = ("Independent LDM", "Motion OFF", "Motion ON")


def render(videos: list[np.ndarray], frame_index: int) -> Image.Image:
    panels = [Image.fromarray(video[frame_index]).resize((256, 256)) for video in videos]
    canvas = Image.new("RGB", (768, 284), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, LABELS)):
        canvas.paste(panel, (index * 256, 28))
        box = draw.textbbox((0, 0), label)
        draw.text((index * 256 + (256 - (box[2] - box[0])) // 2, 8), label, fill="black")
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", default=DEFAULT_RECORD)
    parser.add_argument(
        "--output-dir",
        default="results/P3-W04_video_generation/inference_example",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["cpu_threads"]))
    base_config = load_config(ROOT)
    loaded = load_attempt3_checkpoint(ROOT, base_config, branch="control")
    scaling = float(
        json.loads((ROOT / base_config["output_data"] / "manifest.json").read_text(encoding="utf-8"))[
            "latent_scaling_factor"
        ]
    )
    records = [
        json.loads(line)
        for line in (ROOT / "data/video/p3_w04_v1/records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    test_rows = select_cases(records, int(cfg["test_cases_per_shape"]), split="test")
    matches = [(index, row) for index, row in enumerate(test_rows) if row["record_id"] == args.record_id]
    if not matches:
        available = ", ".join(row["record_id"] for row in test_rows[:5])
        raise ValueError(f"record-id不在I008固定30条测试中；示例：{available}")
    index, row = matches[0]
    obj = row["objects"][0]
    with torch.no_grad():
        anchor, background, mask, _ = generated_anchor_components(
            loaded, base_config, row, scaling, int(cfg["seed"]) + 2000 + index
        )
        trajectory = torch.tensor(obj["trajectory"][:8], dtype=torch.float32).unsqueeze(0)
        clean = anchor[:, :, None].expand(-1, -1, 8, -1, -1).contiguous()
        package = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model = GeneratedAppearanceMotionModule(**package["model_config"])
        model.load_state_dict(package["model_state"], strict=True)
        model.eval()
        independent = sample(
            loaded,
            base_config,
            [frame_prompt(obj, frame) for frame in range(8)],
            int(cfg["seed"]) + 5000 + index,
        )
        latents = (
            independent,
            anchor[:, :, None].expand(-1, -1, 8, -1, -1)[0].permute(1, 0, 2, 3),
            model(
                anchor,
                background,
                mask,
                trajectory,
                use_temporal_attention=True,
                aligned_video=clean,
            )[0].permute(1, 0, 2, 3),
        )
        videos = [
            loaded["vae"].decode(value / scaling)
            .clamp(0, 1)
            .mul(255)
            .round()
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            for value in latents
        ]

    shape_model = TinyConvClassifier(class_count=5)
    load_checkpoint(
        ROOT / "results/P2-W01_training_framework/checkpoints/best.pt",
        model=shape_model,
        optimizer=None,
        restore_rng=False,
    )
    shape_model.eval()
    target = target_for(row)
    metrics = {
        route: evaluate_generated_video(video.astype(np.float32) / 255.0, [target], shape_model)
        for route, video in zip(("independent", "motion_off", "motion_on"), videos)
    }
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frames = [render(videos, frame) for frame in range(8)]
    gif = output / "comparison.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=350, loop=0)
    result = {
        "schema_version": 1,
        "purpose": "I008当前候选固定单样例推理；不属于正式聚合评价",
        "iteration_id": package["iteration_id"],
        "branch_id": package["branch_id"],
        "record_id": row["record_id"],
        "prompt": row["prompt"],
        "expected_segment_zero_based": obj["segments"][0],
        "display_labels": list(LABELS),
        "output_gif": gif.relative_to(ROOT).as_posix(),
        "metrics": metrics,
    }
    (output / "inference_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"record_id": row["record_id"], "gif": str(gif), "motion_on": metrics["motion_on"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

