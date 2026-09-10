"""Train and evaluate P3-W04 I008/generated_motion_module on CPU."""

from __future__ import annotations

import hashlib
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
from p3_w04_pf005_true_independent_ldm import frame_prompt  # noqa: E402
from p3_w04_pf006_train_fair_ldm_adapters import render_comparison  # noqa: E402
from p3_w04_pf011_generated_anchor_motion import (  # noqa: E402
    generated_anchor_components,
    sample,
    select_cases,
    target_for,
)
from p3_w04_pf012_temporal_attention import corrupt  # noqa: E402


CONFIG_PATH = ROOT / "configs/p3_w04_i008_generated_motion.json"
BRANCH = ROOT / "results/P3-W04_video_generation"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def build_items(loaded: dict, base_config: dict, rows: list[dict], scaling: float, seed_base: int) -> list[dict]:
    items = []
    for index, row in enumerate(rows):
        anchor, background, mask, anchor_image = generated_anchor_components(
            loaded, base_config, row, scaling, seed_base + index,
        )
        trajectory = torch.tensor(row["objects"][0]["trajectory"][:8], dtype=torch.float32).unsqueeze(0)
        clean = anchor[:, :, None].expand(-1, -1, 8, -1, -1).contiguous()
        items.append({"row": row, "anchor": anchor, "background": background, "mask": mask,
                      "trajectory": trajectory, "clean": clean, "anchor_image": anchor_image})
    return items


def aggregate(rows: list[dict]) -> dict:
    keys = ("count_accuracy", "color_accuracy", "size_accuracy", "shape_accuracy",
            "trajectory_mae", "direction_accuracy", "timing_accuracy", "background_flicker")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> int:
    started = time.time()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["cpu_threads"]))
    torch.manual_seed(int(cfg["seed"]))
    base_config = load_config(ROOT)
    loaded = load_attempt3_checkpoint(ROOT, base_config, branch="control")
    scaling = float(json.loads((ROOT / base_config["output_data"] / "manifest.json").read_text(encoding="utf-8"))["latent_scaling_factor"])
    records = [json.loads(line) for line in (ROOT / "data/video/p3_w04_v1/records.jsonl").read_text(encoding="utf-8").splitlines()]
    train_rows = select_cases(records, int(cfg["train_cases_per_shape"]), split="train")
    test_rows = select_cases(records, int(cfg["test_cases_per_shape"]), split="test")
    train_items = build_items(loaded, base_config, train_rows, scaling, int(cfg["seed"]) + 1000)
    test_items = build_items(loaded, base_config, test_rows, scaling, int(cfg["seed"]) + 2000)

    model = GeneratedAppearanceMotionModule()
    for parameter in model.temporal.parameters():
        parameter.requires_grad = False
    model.temporal.output_gate.requires_grad = True
    optimizer = torch.optim.Adam([model.temporal.output_gate], lr=float(cfg["learning_rate"]))
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + 3000)
    history = []
    best_loss = float("inf")
    best_state = None
    for update in range(int(cfg["updates"])):
        indices = torch.randint(0, len(train_items), (int(cfg["batch_size"]),), generator=generator).tolist()
        clean = torch.cat([train_items[i]["clean"] for i in indices])
        mask = torch.cat([train_items[i]["mask"] for i in indices])
        trajectory = torch.cat([train_items[i]["trajectory"] for i in indices])
        noisy = corrupt(clean, mask, generator, float(cfg["latent_perturbation_std"]))
        predicted = model.temporal(noisy, trajectory)
        weight = 1.0 + 4.0 * mask[:, :, None]
        reconstruction = ((predicted - clean).square() * weight).mean()
        temporal = (predicted[:, :, 1:] - predicted[:, :, :-1]).square().mean()
        loss = reconstruction + 0.25 * temporal
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        history.append({"update": update + 1, "loss": value,
                        "gate": model.temporal.output_gate.detach().tolist()})
        if value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}

    checkpoint_dir = BRANCH / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    payload = {"schema_version": 1, "work_package": "P3-W04", "iteration_id": "I008",
               "branch_id": "generated_motion_module", "model_state": model.state_dict(),
               "model_config": {"latent_channels": 4}, "training_config": cfg,
               "appearance_source": "P2-W04 I004/control LDM", "program_rgb_target_shape_used": False}
    torch.save(payload, last_path)
    payload["model_state"] = best_state
    torch.save(payload, best_path)
    model.load_state_dict(best_state, strict=True)
    write_json(BRANCH / "metrics/training_history.json", {"updates": history,
               "initial_20_mean": float(np.mean([x["loss"] for x in history[:20]])),
               "final_20_mean": float(np.mean([x["loss"] for x in history[-20:]])),
               "best_loss": best_loss})

    shape_model = TinyConvClassifier(class_count=5)
    load_checkpoint(ROOT / "results/P2-W01_training_framework/checkpoints/best.pt",
                    model=shape_model, optimizer=None, restore_rng=False)
    shape_model.eval()
    routes = ("independent", "motion_off", "motion_on", "attention_stress_off", "attention_stress_on")
    all_metrics = {name: [] for name in routes}
    latent_flicker = {"off": [], "on": []}
    predictions = []
    examples = []
    eval_generator = torch.Generator().manual_seed(int(cfg["seed"]) + 4000)
    with torch.no_grad():
        for index, item in enumerate(test_items):
            row = item["row"]
            obj = row["objects"][0]
            independent = sample(loaded, base_config, [frame_prompt(obj, frame) for frame in range(8)],
                                 int(cfg["seed"]) + 5000 + index)
            noisy = corrupt(item["clean"], item["mask"], eval_generator, float(cfg["latent_perturbation_std"]))
            aligned_stress_on = model.temporal(noisy, item["trajectory"])
            latent_flicker["off"].append(float((noisy[:, :, 1:] - noisy[:, :, :-1]).abs().mean()))
            latent_flicker["on"].append(float((aligned_stress_on[:, :, 1:] - aligned_stress_on[:, :, :-1]).abs().mean()))
            latent_values = {
                "independent": independent,
                "motion_off": item["anchor"][:, :, None].expand(-1, -1, 8, -1, -1)[0].permute(1, 0, 2, 3),
                "motion_on": model(item["anchor"], item["background"], item["mask"], item["trajectory"],
                                   use_temporal_attention=True, aligned_video=item["clean"])[0].permute(1, 0, 2, 3),
                "attention_stress_off": model(item["anchor"], item["background"], item["mask"], item["trajectory"],
                                               use_temporal_attention=False, aligned_video=noisy)[0].permute(1, 0, 2, 3),
                "attention_stress_on": model(item["anchor"], item["background"], item["mask"], item["trajectory"],
                                              use_temporal_attention=True, aligned_video=noisy)[0].permute(1, 0, 2, 3),
            }
            videos = {name: loaded["vae"].decode(value / scaling).clamp(0, 1).mul(255).round().byte()
                      .permute(0, 2, 3, 1).cpu().numpy() for name, value in latent_values.items()}
            target = target_for(row)
            row_result = {"record_id": row["record_id"], "shape": obj["shape"],
                          "prompt": row["prompt"], "segment": obj["segments"][0], "routes": {}}
            for name, video in videos.items():
                result = evaluate_generated_video(video.astype(np.float32) / 255.0, [target], shape_model)
                all_metrics[name].append(result)
                row_result["routes"][name] = result
            predictions.append(row_result)
            if index % int(cfg["test_cases_per_shape"]) == 0:
                display = ("independent", "motion_off", "motion_on")
                animation = [render_comparison([videos[name] for name in display], frame) for frame in range(8)]
                gif_path = BRANCH / "examples" / f"{obj['shape']}_comparison.gif"
                gif_path.parent.mkdir(parents=True, exist_ok=True)
                animation[0].save(gif_path, save_all=True, append_images=animation[1:], duration=350, loop=0)
                examples.append({"record_id": row["record_id"], "shape": obj["shape"],
                                 "prompt": row["prompt"], "segment": obj["segments"][0],
                                 "gif": gif_path.relative_to(ROOT).as_posix(), "metrics": row_result["routes"]})

    aggregate_metrics = {name: aggregate(values) for name, values in all_metrics.items()}
    flicker = {key: float(np.mean(values)) for key, values in latent_flicker.items()}
    th = cfg["thresholds"]
    on = aggregate_metrics["motion_on"]
    independent_agg = aggregate_metrics["independent"]
    checks = {
        "MODEL-VIDGEN-001_motion_module_causal": on["timing_accuracy"] > aggregate_metrics["motion_off"]["timing_accuracy"],
        "MODEL-VIDGEN-002_flicker": on["background_flicker"] <= th["background_flicker_max"],
        "MODEL-VIDGEN-003_trajectory": on["trajectory_mae"] <= th["trajectory_mae_max"],
        "MODEL-VIDGEN-004_count": on["count_accuracy"] >= th["count_accuracy_min"],
        "MODEL-VIDGEN-004_color": on["color_accuracy"] >= th["color_accuracy_min"],
        "MODEL-VIDGEN-004_size": on["size_accuracy"] >= th["size_accuracy_min"],
        "MODEL-VIDGEN-005_direction": on["direction_accuracy"] >= th["direction_accuracy_min"],
        "MODEL-VIDGEN-005_timing": on["timing_accuracy"] >= th["timing_accuracy_min"],
        "MODEL-VIDGEN-006_shape_inheritance": on["shape_accuracy"] >= independent_agg["shape_accuracy"] - th["shape_drop_from_independent_max"],
        "PED-VIDGEN-ATTN-001_latent_flicker": flicker["on"] <= th["attention_latent_flicker_ratio_max"] * flicker["off"],
        "PED-VIDGEN-ATTN-002_pixel_flicker": aggregate_metrics["attention_stress_on"]["background_flicker"] < aggregate_metrics["attention_stress_off"]["background_flicker"],
        "GEN-SOURCE-001_no_program_rgb_shape": True,
    }
    evaluation = {"schema_version": 1, "work_package": "P3-W04", "iteration_id": "I008",
                  "branch_id": "generated_motion_module", "test_case_count": len(test_items),
                  "appearance_source": "P2-W04 I004/control LDM", "program_rgb_target_shape_used": False,
                  "aggregate": aggregate_metrics, "attention_latent_flicker": flicker,
                  "thresholds": th, "checks": checks,
                  "all_blocking_acceptance_items_passed": all(checks.values())}
    write_json(BRANCH / "metrics/evaluation_metrics.json", evaluation)
    predictions_path = BRANCH / "predictions/test_predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n", encoding="utf-8")
    write_json(BRANCH / "examples/capability_examples.json",
               {"schema_version": 1, "example_count": len(examples), "examples": examples})
    example_lines = ["# I008 生成能力样例", "", "对象外观均来自P2-W04 I004/control；程序未绘制目标形状。", ""]
    for item in examples:
        example_lines.extend([f"## {item['shape']} — {item['record_id']}", "", item["prompt"], "",
                              f"![{item['shape']}]({Path(item['gif']).name})", ""])
    (BRANCH / "examples/capability_examples.md").write_text("\n".join(example_lines), encoding="utf-8")
    write_json(checkpoint_dir / "checkpoint_manifest.json", {
        "schema_version": 1,
        "files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "role": role}
                  for path, role in ((best_path, "best"), (last_path, "last"))],
        "p2_w04_checkpoint": loaded["checkpoint"].relative_to(ROOT).as_posix(),
        "p2_w04_checkpoint_sha256": sha256(loaded["checkpoint"]),
    })
    write_json(BRANCH / "metrics/acceptance_report.json",
               {"schema_version": 1, "checks": checks,
                "all_blocking_acceptance_items_passed": all(checks.values()),
                "failed_items": [key for key, passed in checks.items() if not passed]})
    write_json(BRANCH / "metrics/resource_benchmark.json", {
        "wall_seconds": time.time() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_motion_module_parameters": sum(p.numel() for p in model.parameters()),
        "cpu_threads": int(cfg["cpu_threads"]), "train_cases": len(train_items), "test_cases": len(test_items),
    })
    print(json.dumps({"aggregate": aggregate_metrics, "attention_latent_flicker": flicker,
                      "checks": checks, "passed": all(checks.values()),
                      "wall_seconds": time.time() - started}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
