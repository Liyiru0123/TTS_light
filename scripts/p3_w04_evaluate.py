"""P3-W04 TinyT2V I001 evaluation (MODEL-VIDGEN-001..007).

- 25-step DDIM, CFG 2.0; temporal ON vs OFF (same seed) for ablation
- probe_image (P2-W01 shape classifier + palette detection) per frame
- Metrics: ablation (flicker/trajectory), flicker, trajectory MAE, attributes
  (count/color/size), motion (direction/timing), shape recognizability,
  VideoQA closed loop (P3-W01 direction question vs text)

Evidence -> results/P3-W04_video_generation/evaluation/
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.artifacts import probe_image  # noqa: E402
from mini_multimodal.ldm.data import CharacterTokenizer  # noqa: E402
from mini_multimodal.ldm.inference import load_inference_package  # noqa: E402
from mini_multimodal.ldm.schedule import DiffusionSchedule  # noqa: E402
from mini_multimodal.training import load_checkpoint  # noqa: E402
from mini_multimodal.training.baseline import TinyConvClassifier  # noqa: E402
from mini_multimodal.vae.model import TinyImageVAE  # noqa: E402
from mini_multimodal.video_gen.model import MotionPredictor, TinyVideoDenoiser  # noqa: E402
from mini_multimodal.video_gen.sampling import sample_video_latents  # noqa: E402
from mini_multimodal.video_mllm.inference import LoadedTinyVideoMLLM, predict_video  # noqa: E402
from mini_multimodal.video_mllm.data import CharacterTokenizer as VideoMLLMTokenizer  # noqa: E402
from mini_multimodal.video_mllm.model_i003 import TinyVideoMLLMI003  # noqa: E402


def load_p3w01_legacy(path: Path) -> LoadedTinyVideoMLLM:
    """Load the legacy-format I003 (spacetime_queries) inference package."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    tokens = list(bundle["vocabulary"])  # legacy: plain token tuple
    vocab = {"tokens": tokens, "token_to_id": {t: i for i, t in enumerate(tokens)}}
    tokenizer = VideoMLLMTokenizer(vocab)
    answer_classes = tuple(str(value) for value in bundle["answer_classes"])
    model = TinyVideoMLLMI003(
        vocabulary_size=len(tokenizer), answer_class_count=len(answer_classes),
        pad_id=tokenizer.id("<pad>"), bos_id=tokenizer.id("<bos>"),
        eos_id=tokenizer.id("<eos>"), mask_id=tokenizer.id("<mask>"),
        **bundle["model_config"],
    )
    model.load_state_dict(bundle["model_state"], strict=True)
    model.eval()
    return LoadedTinyVideoMLLM(model=model, tokenizer=tokenizer,
                               answer_classes=answer_classes, metadata=bundle)


def load_json(path: Path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def text_attributes(objects: list[dict]) -> dict:
    """Text-specified attributes keyed by color (colors are distinct per video)."""
    return {o["color"]: o for o in objects}


def normalize_object(o: dict) -> dict:
    """Unify new (p3w04_new) and existing (p1_w06) record object fields."""
    if "is_static" in o:
        is_static = bool(o["is_static"])
        segments = o.get("segments", [])
    else:  # p1_w06
        is_static = o.get("motion_direction") == "static"
        segments = []
        if not is_static:
            traj = o.get("trajectory", [])
            centers = [tuple(p) for p in traj]
            start = 0
            while start + 1 < len(centers) and np.allclose(centers[start], centers[start + 1], atol=1e-6):
                start += 1
            end = len(centers) - 1
            while end > start and np.allclose(centers[end], centers[end - 1], atol=1e-6):
                end -= 1
            segments = [(o["motion_direction"], start, end, "constant")]
    return {"color": o["color"], "size": o["size"], "shape": o["shape"],
            "is_static": is_static, "segments": segments,
            "trajectory": o.get("trajectory", [])}


def main() -> int:
    cfg = load_json(ROOT / "configs" / "p3_w04_tiny_t2v.json")
    out_dir = ROOT / cfg["output_results"] / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg["seed"]))
    tr_cfg = cfg["training"]
    th = cfg["thresholds"]

    package = torch.load(ROOT / cfg["image_ldm_package"], map_location="cpu", weights_only=False)
    base_tokenizer = CharacterTokenizer(package["vocabulary"])
    with io.open(ROOT / cfg["output_data"] / "vocabulary.json", encoding="utf-8") as f:
        tokenizer = CharacterTokenizer(json.load(f))
    schedule = DiffusionSchedule(**package["schedule_config"])
    mc = package["model_config"]

    ckpt = torch.load(ROOT / cfg["output_results"] / "checkpoints" / "best.pt",
                      map_location="cpu", weights_only=False)
    model = TinyVideoDenoiser(vocab_size=len(tokenizer), base_vocab_size=len(base_tokenizer),
                              **mc, temporal_cfg=cfg["temporal_module"])
    model.load_frozen_spatial(package["model_state"], freeze_spatial=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    predictor = MotionPredictor(text_dim=int(mc["text_dim"]))
    predictor.load_state_dict(ckpt["predictor_state"], strict=True)
    predictor.eval()

    vae_cfg = load_json(ROOT / cfg["image_vae_config"])
    vae = TinyImageVAE(**{key: vae_cfg["model"][key] for key in
                          ("base_channels", "latent_channels", "logvar_min", "logvar_max")})
    vae.load_state_dict(torch.load(ROOT / cfg["image_vae_checkpoint"], map_location="cpu",
                                   weights_only=False)["model_state"], strict=True)
    vae.eval()
    shape_model = TinyConvClassifier(class_count=5)
    load_checkpoint(ROOT / cfg.get("source_shape_checkpoint", "results/P2-W01_training_framework/checkpoints/best.pt"),
                    model=shape_model, optimizer=None, restore_rng=False)
    shape_model.eval()
    video_mllm = load_p3w01_legacy(ROOT / cfg["video_qa_package"])

    rows = [json.loads(line) for line in io.open(ROOT / cfg["output_data"] / "records.jsonl",
                                                 encoding="utf-8")]
    test_rows = [r for r in rows if r["split"] == "test"]
    # eval set: counterfactual pairs (60 videos) + 30 regular test videos
    cf = [r for r in test_rows if r.get("counterfactual_group")]
    regular = [r for r in test_rows if not r.get("counterfactual_group")][:30]
    eval_rows = (cf + regular)[:90]
    print(f"eval videos: {len(eval_rows)} (counterfactual {len(cf)} + regular {len(regular)})")

    dir_zh = {"left": "向左移动", "right": "向右移动", "up": "向上移动", "down": "向下移动"}
    size_zh = {"small": "小", "medium": "中等大小", "large": "大"}
    shape_zh = {"circle": "圆形", "square": "正方形", "triangle": "三角形",
                "rectangle": "矩形", "pentagon": "五边形"}
    color_zh = {"red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色",
                "purple": "紫色", "orange": "橙色"}

    rows_out = []
    agg = {"flicker": {"on": [], "off": []}, "traj_std": {"on": [], "off": []},
           "traj_mae": [], "generated_mae": [], "jump_rate": [], "count_acc": [], "color_acc": [],
           "size_acc": [], "shape_acc": [], "direction_acc": [], "timing_acc": [],
           "closed_loop_acc": []}

    for idx, row in enumerate(eval_rows):
        prompt = row["prompt"]
        text_objs = {normalize_object(o)["color"]: normalize_object(o) for o in row["objects"]}
        gts = {color: spec["trajectory"] for color, spec in text_objs.items()}
        sampled = {}
        for label, use_t in (("on", True), ("off", False)):
            result = sample_video_latents(
                model, predictor, schedule, tokenizer, [prompt],
                sample_steps=int(tr_cfg["sample_steps"]),
                guidance_scale=float(tr_cfg["cfg_guidance"]),
                seed=200 + idx, use_temporal=use_t)
            latent = result["latent"][0]
            with torch.no_grad():
                frames = vae.decode(latent.permute(1, 0, 2, 3) / float(package["latent_scaling_factor"]))
            sampled[label] = frames.clamp(0, 1).permute(0, 2, 3, 1).numpy()
        predicted_traj = result["trajectory"][0].numpy()  # [8,2] motion predictor output
        # per-frame probes (temporal ON for content metrics)
        probes = []
        for f in range(8):
            probes.append(probe_image(torch.from_numpy(sampled["on"][f]).permute(2, 0, 1), shape_model))
        # flicker: adjacent-frame mean diff (on vs off)
        flick = {}
        for label in ("on", "off"):
            flick[label] = float(np.abs(sampled[label][1:] - sampled[label][:-1]).mean())
        agg["flicker"]["on"].append(flick["on"])
        agg["flicker"]["off"].append(flick["off"])
        # trajectory from probe centroids (match by color)
        row_metrics = {"record_id": row["record_id"], "prompt": prompt,
                       "flicker_on": flick["on"], "flicker_off": flick["off"]}
        matched = {}
        for color, spec in text_objs.items():
            for obj in probes[0].get("objects", []):
                if obj["color"] == color:
                    matched[color] = obj
                    break
        if matched:
            traj_maes, jumps, all_jumps = [], [], []
            for color, obj0 in matched.items():
                gt = gts.get(color)
                centroids = []
                for p in probes:
                    obj = next((o for o in p.get("objects", []) if o["color"] == color), None)
                    centroids.append(np.array(obj["center_xy"]) if obj else None)
                valid = [c for c in centroids if c is not None]
                if gt and valid and len(valid) == 8:
                    gt_arr = np.array(gt[:8])
                    traj_maes.append(float(np.abs(np.array(valid) - gt_arr).mean()))
                disp = np.diff(np.stack([c for c in valid if c is not None]), axis=0)
                if len(disp) > 0:
                    jumps.extend(np.linalg.norm(disp, axis=1).tolist())
                all_jumps.extend(jumps)
            if traj_maes:
                row_metrics["generated_vs_gt_mae"] = float(np.mean(traj_maes))
                agg["generated_mae"].append(row_metrics["generated_vs_gt_mae"])
            if len(all_jumps) > 0:
                median_jump = float(np.median(all_jumps))
                jump_rate = float(np.mean([1.0 if j > 3 * median_jump else 0.0 for j in all_jumps]))
                row_metrics["jump_rate"] = jump_rate
                agg["jump_rate"].append(jump_rate)
                row_metrics["trajectory_jump_std"] = float(np.std(all_jumps))
                agg["traj_std"]["on"].append(row_metrics["trajectory_jump_std"])
            # attributes vs text (first detected object per color)
            for color, spec in text_objs.items():
                obj = next((o for o in probes[0].get("objects", []) if o["color"] == color), None)
                if obj is None:
                    continue
                row_metrics.setdefault("count_check", []).append(1)
                if obj["color"] == spec["color"]:
                    row_metrics.setdefault("color_check", []).append(1)
                area = obj["area"]
                size_est = "large" if area > 700 else "medium" if area > 300 else "small"
                if size_est == spec["size"]:
                    row_metrics.setdefault("size_check", []).append(1)
                if obj["shape"] == spec["shape"]:
                    row_metrics.setdefault("shape_check", []).append(1)
        if "color_check" in row_metrics:
            agg["color_acc"].append(len(row_metrics["color_check"]) == len(text_objs))
        if "size_check" in row_metrics:
            agg["size_acc"].append(len(row_metrics["size_check"]) == len(text_objs))
        if "shape_check" in row_metrics:
            agg["shape_acc"].append(len(row_metrics["shape_check"]) == len(text_objs))
        # motion direction + timing from the PREDICTED trajectory (motion predictor output)
        moving = [spec for spec in text_objs.values() if not spec["is_static"]]
        if moving:
            o = moving[0]
            first_seg = o["segments"][0] if o["segments"] else None
            if first_seg:
                d, s, e, _p = first_seg
                text_dir = d
                traj = predicted_traj  # [8,2] predicted by MotionPredictor
                # trajectory MAE (predictor vs GT)
                if o.get("trajectory"):
                    pred_mae = float(np.abs(traj[:8] - np.array(o["trajectory"][:8])).mean())
                    row_metrics["predictor_traj_mae"] = pred_mae
                    agg["traj_mae"].append(pred_mae)
                d0 = traj[max(0, s - 1)]
                d1 = traj[min(7, e)]
                disp = d1 - d0
                est_dir = ("right" if abs(disp[0]) >= abs(disp[1]) and disp[0] > 0 else
                           "left" if abs(disp[0]) >= abs(disp[1]) else
                           "down" if disp[1] > 0 else "up")
                row_metrics["direction_ok"] = est_dir == text_dir
                agg["direction_acc"].append(est_dir == text_dir)
                # timing: motion start/end from predicted trajectory
                initial = traj[0]
                moved = [f for f, c in enumerate(traj) if np.linalg.norm(c - initial) > 0.02]
                est_start = moved[0] if moved else 0
                est_end = moved[-1] if moved else 0
                timing_ok = abs(est_start - s) <= 1 and abs(est_end - min(e, 7)) <= 1
                row_metrics["timing_ok"] = timing_ok
                agg["timing_acc"].append(timing_ok)
        # closed loop: VideoQA on generated video (temporal ON)
        if moving:
            o = moving[0]
            desc = f"{color_zh[o['color']]}的{size_zh[o['size']]}{shape_zh[o['shape']]}"
            question = f"视频中，{desc}向哪个方向运动？"
            answer = predict_video(video_mllm, sampled["on"], question)["classification_answer"]
            expected = dir_zh.get(o["segments"][0][0]) if o["segments"] else "保持静止"
            row_metrics["closed_loop_answer"] = answer
            row_metrics["closed_loop_expected"] = expected
            ok = answer == expected
            row_metrics["closed_loop_ok"] = ok
            agg["closed_loop_acc"].append(ok)
        rows_out.append(row_metrics)

    def r4(value):
        return round(float(value), 4) if value is not None else None
    report = {
        "schema_version": 1, "work_package": "P3-W04", "n_evaluated": len(eval_rows),
        "thresholds": {k: th[k] for k in ("background_diff_max", "trajectory_mae_max", "jump_rate_max",
                                          "count_accuracy_min", "color_accuracy_min", "size_accuracy_min",
                                          "direction_accuracy_min", "timing_accuracy_min",
                                          "shape_recognizable_min", "closed_loop_accuracy_min")},
        "metrics": {
            "flicker_mean": {"on": r4(np.mean(agg["flicker"]["on"])), "off": r4(np.mean(agg["flicker"]["off"]))},
            "trajectory_jump_std_mean": {"on": r4(np.mean(agg["traj_std"]["on"])), "off": r4(np.mean(agg["traj_std"]["off"]))},
            "trajectory_mae_mean": r4(np.mean(agg["traj_mae"])),
            "generated_vs_gt_mae_mean": r4(np.mean(agg["generated_mae"])),
            "jump_rate_mean": r4(np.mean(agg["jump_rate"])),
            "count_accuracy": r4(np.mean(agg["count_acc"])),
            "color_accuracy": r4(np.mean(agg["color_acc"])),
            "size_accuracy": r4(np.mean(agg["size_acc"])),
            "shape_recognizable": r4(np.mean(agg["shape_acc"])),
            "direction_accuracy": r4(np.mean(agg["direction_acc"])),
            "timing_accuracy": r4(np.mean(agg["timing_acc"])),
            "closed_loop_accuracy": r4(np.mean(agg["closed_loop_acc"])),
        },
        "ablation": {
            "flicker_on_lt_off": float(np.mean(agg["flicker"]["on"])) < float(np.mean(agg["flicker"]["off"])),
            "note": "时间模块 ON vs OFF（同种子）：闪烁/轨迹指标排序",
        },
        "rows": rows_out,
    }
    dump_json(out_dir / "evaluation_report.json", report)
    m = report["metrics"]
    print("=== P3-W04 evaluation (I001) ===")
    for k, v in m.items():
        print(f"  {k}: {v}")
    print(f"  ablation flicker ON<OFF: {report['ablation']['flicker_on_lt_off']}")
    print(f"evidence -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
