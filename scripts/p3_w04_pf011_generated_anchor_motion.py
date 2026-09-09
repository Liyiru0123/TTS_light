"""PF011: P2-W04-generated appearance plus explicit controllable motion.

This is a route preflight, not a model iteration.  The target object is never
drawn by this script: its appearance comes from the frozen P2-W04 I004/control
LDM.  The new video path only propagates the generated latent foreground.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mini_multimodal.ldm.geometry_evaluation import load_attempt3_checkpoint  # noqa: E402
from mini_multimodal.ldm.revision_evaluation import sample_latents_v2  # noqa: E402
from mini_multimodal.ldm.revision_training import load_config  # noqa: E402
from mini_multimodal.training import load_checkpoint  # noqa: E402
from mini_multimodal.training.baseline import TinyConvClassifier  # noqa: E402
from mini_multimodal.video_gen.controlled_motion import GeneratedAppearanceMotionModule, motion_controls  # noqa: E402
from mini_multimodal.video_gen.evaluation import evaluate_generated_video, foreground_mask  # noqa: E402
from p3_w04_pf005_true_independent_ldm import frame_prompt  # noqa: E402
from p3_w04_pf006_train_fair_ldm_adapters import render_comparison  # noqa: E402


OUTPUT = ROOT / "results/P3-W04_video_generation/preflight/PF011_generated_anchor_motion"
SHAPES = ("circle", "square", "triangle", "rectangle", "pentagon")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cases(rows: list[dict], per_shape: int = 4, split: str = "test") -> list[dict]:
    eligible = [
        row for row in rows
        if row["split"] == split and row.get("source") == "p3w04_new"
        and len(row["objects"]) == 1 and len(row["objects"][0].get("segments", [])) == 1
    ]
    selected = []
    for shape in SHAPES:
        candidates = [row for row in eligible if row["objects"][0]["shape"] == shape]
        delayed_early = [
            row for row in candidates
            if int(row["objects"][0]["segments"][0][1]) > 0
            and int(row["objects"][0]["segments"][0][2]) < 7
        ]
        boundary_cases = [
            row for row in candidates
            if int(row["objects"][0]["segments"][0][1]) > 0
            or int(row["objects"][0]["segments"][0][2]) < 7
        ]
        if not delayed_early:
            raise RuntimeError(f"no delayed-start/early-stop case for {shape}")
        ordered = delayed_early + [row for row in boundary_cases if row not in delayed_early]
        if len(ordered) < per_shape:
            raise RuntimeError(f"only {len(ordered)} boundary cases for {shape}")
        selected.extend(ordered[:per_shape])
    return selected


@torch.no_grad()
def sample(loaded: dict, config: dict, prompts: list[str], seed: int) -> torch.Tensor:
    return sample_latents_v2(
        loaded["model"], loaded["schedule"], loaded["tokenizer"], prompts,
        sample_steps=int(config["diffusion"]["sample_steps"]),
        guidance_scale=float(config["diffusion"]["guidance_scale"]), seed=seed,
    )["latent"]


@torch.no_grad()
def generated_anchor_components(
    loaded: dict, config: dict, row: dict, scaling: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    obj = row["objects"][0]
    anchor = sample(loaded, config, [frame_prompt(obj, 0)], seed)
    decoded = loaded["vae"].decode(anchor / scaling).clamp(0, 1)
    image = decoded[0].permute(1, 2, 0).cpu().numpy()
    observed = foreground_mask(image)
    if int(observed.sum()) < 16:
        raise RuntimeError("generated LDM anchor has no detectable foreground")

    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0)
    background_rgb = np.median(border, axis=0).astype(np.float32)
    background_image = torch.from_numpy(background_rgb).view(1, 3, 1, 1).expand(1, 3, 64, 64)
    background_mean, _ = loaded["vae"].encode(background_image)
    background = background_mean * scaling

    mask = torch.from_numpy(observed.astype(np.float32)).view(1, 1, 64, 64)
    mask = F.avg_pool2d(mask, kernel_size=4)
    mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1).clamp(0, 1)
    return anchor, background, mask, image


def target_for(row: dict) -> dict:
    obj = row["objects"][0]
    return {
        "object_id": "obj_0", "shape": obj["shape"], "color": obj["color"],
        "size": obj["size"], "trajectory": obj["trajectory"][:8],
    }


def main() -> int:
    torch.set_num_threads(8)
    started = time.time()
    config = load_config(ROOT)
    loaded = load_attempt3_checkpoint(ROOT, config, branch="control")
    scaling = float(json.loads((ROOT / config["output_data"] / "manifest.json").read_text(encoding="utf-8"))["latent_scaling_factor"])
    shape_model = TinyConvClassifier(class_count=5)
    load_checkpoint(ROOT / "results/P2-W01_training_framework/checkpoints/best.pt",
                    model=shape_model, optimizer=None, restore_rng=False)
    shape_model.eval()
    rows = [json.loads(line) for line in (ROOT / "data/video/p3_w04_v1/records.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = select_cases(rows)
    module = GeneratedAppearanceMotionModule().eval()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    route_metrics = {name: [] for name in ("independent", "temporal_off", "temporal_on")}
    examples = []
    for index, row in enumerate(cases):
        obj = row["objects"][0]
        seed = 20262100 + index
        independent = sample(loaded, config, [frame_prompt(obj, frame) for frame in range(8)], seed)
        anchor, background, mask, anchor_image = generated_anchor_components(loaded, config, row, scaling, seed)
        trajectory = torch.tensor(obj["trajectory"][:8], dtype=torch.float32).unsqueeze(0)
        velocity, moving = motion_controls(trajectory)
        off = anchor[:, :, None].expand(-1, -1, 8, -1, -1)
        on = module(anchor, background, mask, trajectory, use_temporal_attention=True)
        latents = {
            "independent": independent,
            "temporal_off": off[0].permute(1, 0, 2, 3),
            "temporal_on": on[0].permute(1, 0, 2, 3),
        }
        videos = {
            name: loaded["vae"].decode(latent / scaling).clamp(0, 1).mul(255).round().byte()
            .permute(0, 2, 3, 1).cpu().numpy()
            for name, latent in latents.items()
        }
        target = target_for(row)
        metrics = {}
        for name, video in videos.items():
            metrics[name] = evaluate_generated_video(video.astype(np.float32) / 255.0, [target], shape_model)
            route_metrics[name].append(metrics[name])
        animation = [render_comparison([videos[name] for name in route_metrics], frame) for frame in range(8)]
        artifact_stem = f"{obj['shape']}_{index:02d}"
        gif_path = OUTPUT / f"{artifact_stem}_comparison.gif"
        animation[0].save(gif_path, save_all=True, append_images=animation[1:], duration=350, loop=0)
        sheet_path = OUTPUT / f"{artifact_stem}_sheet.png"
        sheet = Image.new("RGB", (8 * 96, 3 * 116), "white")
        sheet_draw = ImageDraw.Draw(sheet)
        for route_index, name in enumerate(route_metrics):
            sheet_draw.text((4, route_index * 116 + 2), name, fill="black")
            for frame_index, frame in enumerate(videos[name]):
                tile = Image.fromarray(frame).resize((96, 96), Image.Resampling.NEAREST)
                sheet.paste(tile, (frame_index * 96, route_index * 116 + 20))
        sheet.save(sheet_path)
        npz_path = OUTPUT / f"{artifact_stem}_comparison.npz"
        np.savez_compressed(npz_path, **videos, anchor_image=anchor_image,
                            foreground_mask=mask[0, 0].numpy(), trajectory=trajectory[0].numpy())
        examples.append({
            "shape": obj["shape"], "record_id": row["record_id"], "prompt": row["prompt"],
            "segment": obj["segments"][0], "moving_state": moving[0, :, 0].tolist(),
            "velocity": velocity[0].tolist(), "gif": gif_path.relative_to(ROOT).as_posix(),
            "sheet": sheet_path.relative_to(ROOT).as_posix(),
            "arrays": npz_path.relative_to(ROOT).as_posix(), "metrics": metrics,
        })

    keys = ("count_accuracy", "color_accuracy", "size_accuracy", "shape_accuracy", "trajectory_mae",
            "direction_accuracy", "timing_accuracy", "background_flicker")
    aggregate = {
        route: {key: float(np.mean([item[key] for item in values])) for key in keys}
        for route, values in route_metrics.items()
    }
    checks = {
        "timing_nonzero_and_at_least_0p60": aggregate["temporal_on"]["timing_accuracy"] >= 0.60,
        "direction_at_least_0p80": aggregate["temporal_on"]["direction_accuracy"] >= 0.80,
        "trajectory_at_most_0p06": aggregate["temporal_on"]["trajectory_mae"] <= 0.06,
        "trajectory_better_than_independent": aggregate["temporal_on"]["trajectory_mae"] < aggregate["independent"]["trajectory_mae"],
        "shape_inheritance_margin": aggregate["temporal_on"]["shape_accuracy"] >= aggregate["independent"]["shape_accuracy"] - 0.10,
        "no_program_rgb_target_shape": True,
    }
    report = {
        "schema_version": 1, "work_package": "P3-W04", "preflight_id": "PF011",
        "route": "P2-W04-generated appearance anchor + explicit latent motion field + AnimateDiff-style temporal attention",
        "program_rgb_target_shape_used": False, "appearance_source": "P2-W04 I004/control LDM",
        "p2_w04_checkpoint": loaded["checkpoint"].relative_to(ROOT).as_posix(),
        "p2_w04_checkpoint_sha256": sha256(loaded["checkpoint"]),
        "control_channels": ["position_xy", "velocity_xy", "moving_static"],
        "temporal_attention_gate": module.temporal.output_gate.detach().tolist(),
        "temporal_attention_note": "zero-initialized in route preflight; PF011 first isolates whether explicit motion can execute without redrawing appearance",
        "aggregate": aggregate, "checks": checks, "passed": all(checks.values()),
        "wall_seconds": time.time() - started, "examples": examples,
    }
    (OUTPUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "checks": checks, "passed": report["passed"],
                      "wall_seconds": report["wall_seconds"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
