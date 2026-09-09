"""PF005: generate true independent-frame videos with the P2-W04 I004 LDM."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.geometry_evaluation import load_attempt3_checkpoint  # noqa: E402
from mini_multimodal.ldm.revision_evaluation import sample_latents_v2  # noqa: E402
from mini_multimodal.ldm.revision_training import load_config  # noqa: E402
from mini_multimodal.training import load_checkpoint  # noqa: E402
from mini_multimodal.training.baseline import TinyConvClassifier  # noqa: E402
from mini_multimodal.video_gen.evaluation import evaluate_generated_video  # noqa: E402

OUTPUT = ROOT / "results/P3-W04_video_generation/preflight/PF005_true_independent_ldm"
SHAPES = ("circle", "square", "triangle", "rectangle", "pentagon")
COLOR_ZH = {"red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色", "purple": "紫色", "orange": "橙色"}
SIZE_ZH = {"small": "小", "medium": "中等大小", "large": "大"}
SHAPE_ZH = {"circle": "圆形", "square": "正方形", "triangle": "三角形", "rectangle": "矩形", "pentagon": "五边形"}
REGION_ZH = {
    (0, 0): "左上区域", (1, 0): "上方区域", (2, 0): "右上区域",
    (0, 1): "左侧区域", (1, 1): "中央区域", (2, 1): "右侧区域",
    (0, 2): "左下区域", (1, 2): "下方区域", (2, 2): "右下区域",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def region_name(point: list[float]) -> str:
    x, y = (float(point[0]), float(point[1]))
    column = 0 if x < 1 / 3 else 2 if x > 2 / 3 else 1
    row = 0 if y < 1 / 3 else 2 if y > 2 / 3 else 1
    return REGION_ZH[(column, row)]


def frame_prompt(obj: dict, frame_index: int) -> str:
    region = region_name(obj["trajectory"][frame_index])
    return (
        f"第1帧中有1个可见对象。对象1是{COLOR_ZH[obj['color']]}的"
        f"{SIZE_ZH[obj['size']]}{SHAPE_ZH[obj['shape']]}，位于{region}。"
    )


def main() -> int:
    torch.set_num_threads(8)
    config = load_config(ROOT)
    loaded = load_attempt3_checkpoint(ROOT, config, branch="control")
    shape_model = TinyConvClassifier(class_count=5)
    load_checkpoint(
        ROOT / "results/P2-W01_training_framework/checkpoints/best.pt",
        model=shape_model,
        optimizer=None,
        restore_rng=False,
    )
    shape_model.eval()
    scaling = float(json.loads((ROOT / config["output_data"] / "manifest.json").read_text(encoding="utf-8"))["latent_scaling_factor"])
    records = [
        json.loads(line)
        for line in (ROOT / "data/video/p3_w04_v1/records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    eligible = [
        row for row in records
        if row["split"] == "test"
        and row.get("source") == "p3w04_new"
        and len(row["objects"]) == 1
        and len(row["objects"][0].get("segments", [])) == 1
    ]
    selected = {shape: next(row for row in eligible if row["objects"][0]["shape"] == shape) for shape in SHAPES}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    examples = []
    for index, (shape, row) in enumerate(selected.items()):
        obj = row["objects"][0]
        prompts = [frame_prompt(obj, frame) for frame in range(8)]
        sampled = sample_latents_v2(
            loaded["model"], loaded["schedule"], loaded["tokenizer"], prompts,
            sample_steps=int(config["diffusion"]["sample_steps"]),
            guidance_scale=float(config["diffusion"]["guidance_scale"]),
            seed=20260822 + index,
        )
        with torch.no_grad():
            decoded = loaded["vae"].decode(sampled["latent"] / scaling).clamp(0, 1)
        frames = decoded.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        images = [Image.fromarray(frame).resize((256, 256), Image.Resampling.NEAREST) for frame in frames]
        gif_path = OUTPUT / f"{shape}_independent.gif"
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=350, loop=0)
        npz_path = OUTPUT / f"{shape}_independent.npz"
        np.savez_compressed(npz_path, frames=frames, latent=sampled["latent"].cpu().numpy())
        direction, start, end, curve = obj["segments"][0]
        target = {
            "object_id": "obj_0",
            "shape": obj["shape"],
            "color": obj["color"],
            "size": obj["size"],
            "trajectory": obj["trajectory"][:8],
        }
        metrics = evaluate_generated_video(frames.astype(np.float32) / 255.0, [target], shape_model)
        examples.append({
            "shape": shape,
            "record_id": row["record_id"],
            "source_video_prompt": row["prompt"],
            "frame_prompts": prompts,
            "expected_motion": {"direction": direction, "start": start, "end": end, "curve": curve},
            "gif": gif_path.relative_to(ROOT).as_posix(),
            "frames": npz_path.relative_to(ROOT).as_posix(),
            "pixel_metrics": metrics,
        })
    metric_keys = (
        "count_accuracy", "color_accuracy", "size_accuracy", "shape_accuracy",
        "trajectory_mae", "direction_accuracy", "timing_accuracy", "background_flicker",
    )
    aggregate = {
        key: float(np.mean([example["pixel_metrics"][key] for example in examples]))
        for key in metric_keys
    }
    report = {
        "schema_version": 1,
        "work_package": "P3-W04",
        "preflight_id": "PF005",
        "route": "true_independent_frames_from_p2_w04_i004_control",
        "p2_w04_checkpoint": loaded["checkpoint"].relative_to(ROOT).as_posix(),
        "p2_w04_checkpoint_sha256": sha256(loaded["checkpoint"]),
        "sample_steps": int(config["diffusion"]["sample_steps"]),
        "guidance_scale": float(config["diffusion"]["guidance_scale"]),
        "program_rgb_anchor_used": False,
        "independent_noise_per_frame": True,
        "aggregate_pixel_metrics": aggregate,
        "examples": examples,
    }
    (OUTPUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
