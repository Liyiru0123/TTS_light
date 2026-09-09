"""PF012: causal preflight for trainable temporal attention.

The target is a P2-W04-generated appearance anchor repeated in aligned latent
space.  Independent per-frame latent perturbations model the failure mode of
framewise generation.  No program-rendered RGB target shape is used.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
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
from p3_w04_pf006_train_fair_ldm_adapters import render_comparison  # noqa: E402
from p3_w04_pf011_generated_anchor_motion import (  # noqa: E402
    generated_anchor_components,
    select_cases,
    target_for,
)


OUTPUT = ROOT / "results/P3-W04_video_generation/preflight/PF012_temporal_attention"


@torch.no_grad()
def build_items(loaded: dict, config: dict, rows: list[dict], scaling: float, seed_base: int) -> list[dict]:
    items = []
    for index, row in enumerate(rows):
        anchor, background, mask, _ = generated_anchor_components(
            loaded, config, row, scaling, seed_base + index,
        )
        trajectory = torch.tensor(row["objects"][0]["trajectory"][:8], dtype=torch.float32).unsqueeze(0)
        clean = anchor[:, :, None].expand(-1, -1, 8, -1, -1).contiguous()
        items.append({"row": row, "anchor": anchor, "background": background,
                      "mask": mask, "trajectory": trajectory, "clean": clean})
    return items


def corrupt(clean: torch.Tensor, mask: torch.Tensor, generator: torch.Generator, std: float) -> torch.Tensor:
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    # Perturb the observed generated foreground and a narrow halo.  The target
    # remains the LDM anchor, never a renderer-produced shape.
    weight = (0.2 + 0.8 * mask[:, :, None]).expand_as(clean)
    return clean + float(std) * noise * weight


def mean_metric(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def main() -> int:
    torch.set_num_threads(8)
    started = time.time()
    config = load_config(ROOT)
    loaded = load_attempt3_checkpoint(ROOT, config, branch="control")
    scaling = float(json.loads((ROOT / config["output_data"] / "manifest.json").read_text(encoding="utf-8"))["latent_scaling_factor"])
    records = [json.loads(line) for line in (ROOT / "data/video/p3_w04_v1/records.jsonl").read_text(encoding="utf-8").splitlines()]
    train_items = build_items(loaded, config, select_cases(records, per_shape=2, split="train"), scaling, 20262200)
    test_items = build_items(loaded, config, select_cases(records, per_shape=4, split="test"), scaling, 20262300)

    module = GeneratedAppearanceMotionModule()
    for parameter in module.temporal.parameters():
        parameter.requires_grad = False
    module.temporal.output_gate.requires_grad = True
    optimizer = torch.optim.Adam([module.temporal.output_gate], lr=0.01)
    generator = torch.Generator().manual_seed(20262400)
    history = []
    updates = 240
    for update in range(updates):
        indices = torch.randint(0, len(train_items), (2,), generator=generator).tolist()
        clean = torch.cat([train_items[i]["clean"] for i in indices])
        mask = torch.cat([train_items[i]["mask"] for i in indices])
        trajectory = torch.cat([train_items[i]["trajectory"] for i in indices])
        noisy = corrupt(clean, mask, generator, std=0.12)
        predicted = module.temporal(noisy, trajectory)
        weight = 1.0 + 4.0 * mask[:, :, None]
        reconstruction = ((predicted - clean).square() * weight).mean()
        temporal = (predicted[:, :, 1:] - predicted[:, :, :-1]).square().mean()
        loss = reconstruction + 0.25 * temporal
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
        optimizer.step()
        history.append(float(loss.item()))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUTPUT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save({"model_state": module.state_dict(), "updates": updates,
                "target_source": "P2-W04-generated latent appearance"}, checkpoint_dir / "temporal_attention.pt")

    shape_model = TinyConvClassifier(class_count=5)
    load_checkpoint(ROOT / "results/P2-W01_training_framework/checkpoints/best.pt",
                    model=shape_model, optimizer=None, restore_rng=False)
    shape_model.eval()
    metrics = {name: [] for name in ("stress_off", "stress_on", "clean_off", "clean_on")}
    latent_delta = {"stress_off": [], "stress_on": []}
    examples = []
    eval_generator = torch.Generator().manual_seed(20262500)
    with torch.no_grad():
        for index, item in enumerate(test_items):
            noisy = corrupt(item["clean"], item["mask"], eval_generator, std=0.12)
            aligned_on = module.temporal(noisy, item["trajectory"])
            latent_delta["stress_off"].append(float((noisy[:, :, 1:] - noisy[:, :, :-1]).abs().mean()))
            latent_delta["stress_on"].append(float((aligned_on[:, :, 1:] - aligned_on[:, :, :-1]).abs().mean()))
            latents = {
                "stress_off": module(item["anchor"], item["background"], item["mask"], item["trajectory"],
                                     use_temporal_attention=False, aligned_video=noisy),
                "stress_on": module(item["anchor"], item["background"], item["mask"], item["trajectory"],
                                    use_temporal_attention=True, aligned_video=noisy),
                "clean_off": module(item["anchor"], item["background"], item["mask"], item["trajectory"],
                                    use_temporal_attention=False, aligned_video=item["clean"]),
                "clean_on": module(item["anchor"], item["background"], item["mask"], item["trajectory"],
                                   use_temporal_attention=True, aligned_video=item["clean"]),
            }
            videos = {
                name: loaded["vae"].decode(value[0].permute(1, 0, 2, 3) / scaling).clamp(0, 1)
                .mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                for name, value in latents.items()
            }
            target = target_for(item["row"])
            row_metrics = {}
            for name, video in videos.items():
                row_metrics[name] = evaluate_generated_video(video.astype(np.float32) / 255.0, [target], shape_model)
                metrics[name].append(row_metrics[name])
            if index % 4 == 0:
                display_routes = ("stress_off", "stress_on", "clean_on")
                animation = [render_comparison([videos[name] for name in display_routes], frame) for frame in range(8)]
                gif = OUTPUT / f"{item['row']['objects'][0]['shape']}_comparison.gif"
                animation[0].save(gif, save_all=True, append_images=animation[1:], duration=350, loop=0)
                examples.append({"record_id": item["row"]["record_id"],
                                 "shape": item["row"]["objects"][0]["shape"],
                                 "gif": gif.relative_to(ROOT).as_posix(), "metrics": row_metrics})

    keys = ("count_accuracy", "color_accuracy", "size_accuracy", "shape_accuracy", "trajectory_mae",
            "direction_accuracy", "timing_accuracy", "background_flicker")
    aggregate = {route: {key: mean_metric(values, key) for key in keys} for route, values in metrics.items()}
    latent_flicker = {route: float(np.mean(values)) for route, values in latent_delta.items()}
    checks = {
        "gate_nonzero": float(module.temporal.output_gate.detach().abs().mean()) > 0.05,
        "stress_latent_flicker_improves": latent_flicker["stress_on"] < 0.75 * latent_flicker["stress_off"],
        "stress_pixel_flicker_improves": aggregate["stress_on"]["background_flicker"] < aggregate["stress_off"]["background_flicker"],
        "clean_timing": aggregate["clean_on"]["timing_accuracy"] >= 0.90,
        "clean_direction": aggregate["clean_on"]["direction_accuracy"] >= 0.90,
        "clean_trajectory": aggregate["clean_on"]["trajectory_mae"] <= 0.06,
        "clean_shape_inherited": aggregate["clean_on"]["shape_accuracy"] >= aggregate["clean_off"]["shape_accuracy"] - 0.02,
        "clean_size_inherited": aggregate["clean_on"]["size_accuracy"] >= aggregate["clean_off"]["size_accuracy"] - 0.02,
        "no_program_rgb_target_shape": True,
    }
    report = {
        "schema_version": 1, "work_package": "P3-W04", "preflight_id": "PF012",
        "route": "trainable motion-aligned temporal attention on P2-W04-generated appearance",
        "program_rgb_target_shape_used": False, "updates": updates,
        "train_case_count": len(train_items), "test_case_count": len(test_items),
        "train_loss": {"initial_20_mean": float(np.mean(history[:20])),
                       "final_20_mean": float(np.mean(history[-20:]))},
        "temporal_gate": module.temporal.output_gate.detach().tolist(),
        "trainable_parameters": sum(p.numel() for p in module.temporal.parameters() if p.requires_grad),
        "latent_flicker": latent_flicker, "aggregate": aggregate,
        "checks": checks, "passed": all(checks.values()),
        "wall_seconds": time.time() - started, "examples": examples,
    }
    (OUTPUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"loss": report["train_loss"], "gate": report["temporal_gate"],
                      "latent_flicker": latent_flicker, "aggregate": aggregate,
                      "checks": checks, "passed": report["passed"],
                      "wall_seconds": report["wall_seconds"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
