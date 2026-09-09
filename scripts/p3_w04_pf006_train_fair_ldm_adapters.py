"""PF006: fair spatial-only versus temporal latent-adapter preflight."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.geometry_evaluation import load_attempt3_checkpoint  # noqa: E402
from mini_multimodal.ldm.revision_evaluation import sample_latents_v2  # noqa: E402
from mini_multimodal.ldm.revision_training import load_config  # noqa: E402
from mini_multimodal.training import load_checkpoint  # noqa: E402
from mini_multimodal.training.baseline import TinyConvClassifier  # noqa: E402
from mini_multimodal.video_gen.evaluation import evaluate_generated_video  # noqa: E402
from mini_multimodal.video_gen.ldm_temporal_adapter import LDMVideoAdapter  # noqa: E402

OUTPUT = ROOT / "results/P3-W04_video_generation/preflight/PF006_fair_ldm_adapters"
SHAPES = ("circle", "square", "triangle", "rectangle", "pentagon")
COLOR_ZH = {"red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色", "purple": "紫色", "orange": "橙色"}
SIZE_ZH = {"small": "小", "medium": "中等大小", "large": "大"}
SHAPE_ZH = {"circle": "圆形", "square": "正方形", "triangle": "三角形", "rectangle": "矩形", "pentagon": "五边形"}
REGION_ZH = {
    (0, 0): "左上区域", (1, 0): "上方区域", (2, 0): "右上区域",
    (0, 1): "左侧区域", (1, 1): "中央区域", (2, 1): "右侧区域",
    (0, 2): "左下区域", (1, 2): "下方区域", (2, 2): "右下区域",
}


def region_name(point: list[float]) -> str:
    x, y = float(point[0]), float(point[1])
    column = 0 if x < 1 / 3 else 2 if x > 2 / 3 else 1
    row = 0 if y < 1 / 3 else 2 if y > 2 / 3 else 1
    return REGION_ZH[(column, row)]


def frame_prompts(obj: dict) -> list[str]:
    return [
        f"第1帧中有1个可见对象。对象1是{COLOR_ZH[obj['color']]}的"
        f"{SIZE_ZH[obj['size']]}{SHAPE_ZH[obj['shape']]}，位于{region_name(obj['trajectory'][frame])}。"
        for frame in range(8)
    ]


@torch.no_grad()
def independent_latents(loaded: dict, config: dict, row: dict, seed: int) -> torch.Tensor:
    sampled = sample_latents_v2(
        loaded["model"], loaded["schedule"], loaded["tokenizer"], frame_prompts(row["objects"][0]),
        sample_steps=int(config["diffusion"]["sample_steps"]),
        guidance_scale=float(config["diffusion"]["guidance_scale"]), seed=seed,
    )
    return sampled["latent"].permute(1, 0, 2, 3).contiguous()


def select_balanced(rows: list[dict], split: str, per_shape: int) -> list[dict]:
    eligible = [
        row for row in rows
        if row["split"] == split and row.get("source") == "p3w04_new"
        and len(row["objects"]) == 1 and len(row["objects"][0].get("segments", [])) == 1
    ]
    selected = []
    for shape in SHAPES:
        selected.extend([row for row in eligible if row["objects"][0]["shape"] == shape][:per_shape])
    return selected


def target_latent(row: dict) -> torch.Tensor:
    latent = np.load(ROOT / "data/video/p3_w04_v1" / row["latent_path"], allow_pickle=False)["latent"]
    return torch.from_numpy(latent.astype(np.float32)).permute(1, 0, 2, 3).contiguous()


def loss_terms(output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction = (output - target).square().mean()
    output_delta = output[:, :, 1:] - output[:, :, :-1]
    target_delta = target[:, :, 1:] - target[:, :, :-1]
    temporal = (output_delta - target_delta).square().mean()
    return reconstruction + 0.5 * temporal, reconstruction, temporal


def render_comparison(groups: list[np.ndarray], frame: int) -> Image.Image:
    labels = ("Independent LDM", "Temporal OFF", "Temporal ON")
    canvas = Image.new("RGB", (768, 284), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (video, label) in enumerate(zip(groups, labels, strict=True)):
        image = Image.fromarray(video[frame]).resize((256, 256), Image.Resampling.NEAREST)
        canvas.paste(image, (column * 256, 28))
        draw.text((column * 256 + 6, 7), label, fill="black")
    return canvas


def main() -> int:
    torch.set_num_threads(8)
    torch.manual_seed(20260822)
    config = load_config(ROOT)
    loaded = load_attempt3_checkpoint(ROOT, config, branch="control")
    scaling = float(json.loads((ROOT / config["output_data"] / "manifest.json").read_text(encoding="utf-8"))["latent_scaling_factor"])
    rows = [json.loads(line) for line in (ROOT / "data/video/p3_w04_v1/records.jsonl").read_text(encoding="utf-8").splitlines()]
    train_rows = select_balanced(rows, "train", per_shape=4)
    test_rows = select_balanced(rows, "test", per_shape=1)
    if len(train_rows) != 20 or len(test_rows) != 5:
        raise RuntimeError("balanced preflight requires 20 train and 5 test videos")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source_train = torch.stack([
        independent_latents(loaded, config, row, 20260900 + index)
        for index, row in enumerate(train_rows)
    ])
    target_train = torch.stack([target_latent(row) for row in train_rows])

    torch.manual_seed(17)
    off = LDMVideoAdapter(base_channels=16, use_temporal=False)
    torch.manual_seed(17)
    on = LDMVideoAdapter(base_channels=16, use_temporal=True)
    on_state = on.state_dict()
    for name, value in off.state_dict().items():
        if name in on_state and on_state[name].shape == value.shape:
            on_state[name] = value.clone()
    on.load_state_dict(on_state, strict=True)
    optimizers = {
        "temporal_off": torch.optim.Adam(off.parameters(), lr=0.001),
        "temporal_on": torch.optim.Adam(on.parameters(), lr=0.001),
    }
    models = {"temporal_off": off, "temporal_on": on}
    generator = torch.Generator().manual_seed(91)
    history = {name: [] for name in models}
    updates = 160
    for update in range(updates):
        indices = torch.randint(0, len(train_rows), (2,), generator=generator)
        source, target = source_train[indices], target_train[indices]
        for name, model in models.items():
            model.train()
            output = model(source)
            loss, reconstruction, temporal = loss_terms(output, target)
            optimizers[name].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizers[name].step()
            history[name].append({"loss": float(loss.item()), "reconstruction": float(reconstruction.item()), "temporal": float(temporal.item())})
        if (update + 1) % 40 == 0:
            print(update + 1, {name: round(float(np.mean([x["loss"] for x in values[-20:]])), 5) for name, values in history.items()}, flush=True)

    checkpoints = OUTPUT / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    torch.save({"model_state": off.state_dict(), "use_temporal": False}, checkpoints / "temporal_off.pt")
    torch.save({"model_state": on.state_dict(), "use_temporal": True}, checkpoints / "temporal_on.pt")

    shape_model = TinyConvClassifier(class_count=5)
    load_checkpoint(ROOT / "results/P2-W01_training_framework/checkpoints/best.pt", model=shape_model, optimizer=None, restore_rng=False)
    shape_model.eval()
    route_metrics = {name: [] for name in ("independent", "temporal_off", "temporal_on")}
    examples = []
    for index, row in enumerate(test_rows):
        source = independent_latents(loaded, config, row, 20261000 + index).unsqueeze(0)
        with torch.no_grad():
            latents = {"independent": source, "temporal_off": off(source), "temporal_on": on(source)}
            videos = {
                name: loaded["vae"].decode(value[0].permute(1, 0, 2, 3) / scaling).clamp(0, 1)
                .mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                for name, value in latents.items()
            }
        obj = row["objects"][0]
        target = {"object_id": "obj_0", "shape": obj["shape"], "color": obj["color"],
                  "size": obj["size"], "trajectory": obj["trajectory"][:8]}
        metrics = {}
        for name, video in videos.items():
            metrics[name] = evaluate_generated_video(video.astype(np.float32) / 255.0, [target], shape_model)
            route_metrics[name].append(metrics[name])
        animation = [render_comparison([videos[name] for name in route_metrics], frame) for frame in range(8)]
        path = OUTPUT / f"{obj['shape']}_comparison.gif"
        animation[0].save(path, save_all=True, append_images=animation[1:], duration=350, loop=0)
        examples.append({"shape": obj["shape"], "record_id": row["record_id"], "prompt": row["prompt"],
                         "gif": path.relative_to(ROOT).as_posix(), "metrics": metrics})

    keys = ("count_accuracy", "color_accuracy", "size_accuracy", "shape_accuracy", "trajectory_mae",
            "direction_accuracy", "timing_accuracy", "background_flicker")
    aggregate = {
        route: {key: float(np.mean([item[key] for item in values])) for key in keys}
        for route, values in route_metrics.items()
    }
    report = {
        "schema_version": 1, "work_package": "P3-W04", "preflight_id": "PF006",
        "source": "P2-W04 I004/control TinyImageLDM independent per-frame latents",
        "program_rgb_anchor_used": False, "program_video_used_as_training_target_only": True,
        "fairness": {"same_source_latents": True, "same_target_latents": True, "same_batch_order": True,
                     "same_updates": updates, "off_separately_trained": True},
        "train_videos": len(train_rows), "test_videos": len(test_rows), "aggregate": aggregate,
        "training": {name: {"initial_loss": float(np.mean([x["loss"] for x in values[:20]])),
                             "final_loss": float(np.mean([x["loss"] for x in values[-20:]]))}
                     for name, values in history.items()},
        "wall_seconds": time.time() - started, "examples": examples,
    }
    (OUTPUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "training": report["training"], "wall_seconds": report["wall_seconds"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
