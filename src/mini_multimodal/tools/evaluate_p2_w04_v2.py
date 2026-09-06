"""Evaluate the already-trained P2-W04 v2.0 checkpoint without retraining.

Loads best.pt (trained with spatial position encoding) and runs:
denoising -> generation (with position) -> conditioning ablation -> CFG -> reproducibility -> resource
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter

import psutil
import torch
from torch.utils.data import DataLoader

from mini_multimodal.ldm.artifacts import (
    _aggregate_generation,
    _load_shape_model,
    _load_vae,
    _model_kwargs,
    _new_trainer,
    _save_forward_noise,
    _save_training_curve,
    _save_trajectory,
    _schedule,
    _state_digest,
    _write_json,
    _write_jsonl,
    evaluate_denoising,
    generate_records,
)
from mini_multimodal.ldm.data import CharacterTokenizer, LatentTextDataset, collate_latent_text, verify_p2_w04_data
from mini_multimodal.ldm.inference import sample_latents
from mini_multimodal.training import load_checkpoint, set_deterministic_seed


def main() -> None:
    started = perf_counter()
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
    result_dir = root / config["output_results"]
    result_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))
    data_report = verify_p2_w04_data(root, config["output_data"])
    vocabulary = json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocabulary)
    train = LatentTextDataset(root, data_dir=config["output_data"], split="train", max_text_length=int(config["model"]["max_text_length"]))
    validation = LatentTextDataset(root, data_dir=config["output_data"], split="validation", max_text_length=int(config["model"]["max_text_length"]))
    test = LatentTextDataset(root, data_dir=config["output_data"], split="test", max_text_length=int(config["model"]["max_text_length"]))
    thresholds = config["thresholds"]

    # Load best checkpoint (already trained)
    from mini_multimodal.ldm.model import TinyImageDenoiser
    model = TinyImageDenoiser(**_model_kwargs(config, len(tokenizer), vocabulary))
    best_path = result_dir / "training" / "checkpoints" / "best.pt"
    best_state = load_checkpoint(best_path, model=model, optimizer=None, restore_rng=False)
    model.eval()
    print(f"Loaded best checkpoint: epoch={best_state['epoch']}, best_metric={best_state['best_metric']:.6f}")

    schedule = _schedule(config)

    def _loader(dataset, shuffle=False, s=seed):
        return DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=shuffle, num_workers=0, collate_fn=collate_latent_text, generator=None)

    # 1. Denoising evaluation
    print("Running denoising evaluation...")
    from mini_multimodal.ldm.model import build_position_token_ids as _bpt
    _pos_strength = float(config["diffusion"].get("position_bias_strength", 0.0))
    _pos_ids = _bpt(vocabulary)
    denoising = evaluate_denoising(model, schedule, _loader(test), seed=seed + 101, position_bias_strength=_pos_strength, position_token_ids=_pos_ids)
    no_cross = evaluate_denoising(model, schedule, _loader(test), seed=seed + 101, disable_cross_attention=True, position_bias_strength=_pos_strength, position_token_ids=_pos_ids)
    denoising.update({
        "best_epoch": int(best_state["epoch"]),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "cross_attention_disabled_noise_mse": no_cross["noise_mse"],
        "cross_attention_mse_increase": no_cross["noise_mse"] - denoising["noise_mse"],
        "checks": {
            "noise_mse": denoising["noise_mse"] <= thresholds["test_noise_mse"],
            "baseline_improvement": denoising["improvement"] >= thresholds["noise_baseline_improvement"],
        },
    })
    denoising["verified"] = all(denoising["checks"].values())
    _write_json(result_dir / "denoising_metrics.json", denoising)
    print(f"  noise_mse={denoising['noise_mse']:.4f} cross_attn_increase={denoising['cross_attention_mse_increase']:.4f} verified={denoising['verified']}")

    # 2. Generation evaluation (with position)
    print("Running generation evaluation...")
    vae = _load_vae(root, config)
    shape_model = _load_shape_model(root, config)
    single_rows = [row for row in test.rows if len(row["objects"]) == 1 and row["variant"] != "compositional"][: int(config["data"]["generation_evaluation_count"])]
    composition_rows = [row for row in test.rows if len(row["objects"]) == 1 and row["variant"] == "compositional"][: int(config["data"]["composition_evaluation_count"])]
    evaluation_rows = single_rows + composition_rows
    generated_dir = result_dir / "generated_samples"
    full_records = generate_records(
        model, schedule, tokenizer, vae, shape_model, evaluation_rows,
        scaling_factor=test.scaling_factor,
        sample_steps=int(config["diffusion"]["sample_steps"]),
        guidance_scale=float(config["diffusion"]["guidance_scale"]),
        seed=seed + 200,
        save_dir=generated_dir,
        prefix="conditional",
        position_bias_strength=float(config["diffusion"].get("position_bias_strength", 0.0)),
    )
    single_metrics = _aggregate_generation(full_records[: len(single_rows)])
    composition_metrics = _aggregate_generation(full_records[len(single_rows):])
    generation = {
        "single_object": single_metrics,
        "unseen_composition": composition_metrics,
        "checks": {
            "single_color_accuracy": single_metrics["color_accuracy"] >= thresholds["single_color_accuracy"],
            "single_shape_accuracy": single_metrics["shape_accuracy"] >= thresholds["single_shape_accuracy"],
            "single_position_accuracy": single_metrics["position_accuracy"] >= thresholds["single_position_accuracy"],
            "foreground_presence": single_metrics["foreground_presence"] >= thresholds["foreground_presence"],
            "composition_color_accuracy": composition_metrics["color_accuracy"] >= thresholds["composition_color_accuracy"],
            "composition_shape_accuracy": composition_metrics["shape_accuracy"] >= thresholds["composition_shape_accuracy"],
            "composition_position_accuracy": composition_metrics["position_accuracy"] >= thresholds["composition_position_accuracy"],
        },
    }
    generation["verified"] = all(generation["checks"].values())
    _write_json(result_dir / "generation_metrics.json", generation)
    _write_jsonl(result_dir / "generation_predictions.jsonl", full_records)
    print(f"  single: color={single_metrics['color_accuracy']:.4f} shape={single_metrics['shape_accuracy']:.4f} pos={single_metrics['position_accuracy']:.4f} fg={single_metrics['foreground_presence']:.4f}")
    print(f"  compos: color={composition_metrics['color_accuracy']:.4f} shape={composition_metrics['shape_accuracy']:.4f} pos={composition_metrics['position_accuracy']:.4f}")
    print(f"  verified={generation['verified']}")

    # 3. Conditioning ablation
    print("Running conditioning ablation...")
    shuffled_prompts = [row["prompt"] for row in evaluation_rows[1:]] + [evaluation_rows[0]["prompt"]]
    ablations = {}
    for name, scale, prompts in (
        ("no_guidance", 1.0, None),
        ("unconditional", 0.0, None),
        ("shuffled_text", float(config["diffusion"]["guidance_scale"]), shuffled_prompts),
    ):
        records = generate_records(
            model, schedule, tokenizer, vae, shape_model, evaluation_rows,
            scaling_factor=test.scaling_factor,
            sample_steps=int(config["diffusion"]["sample_steps"]),
            guidance_scale=scale,
            seed=seed + 200,
            prompt_override=prompts,
        )
        ablations[name] = _aggregate_generation(records)
    full_metrics = _aggregate_generation(full_records)
    best_control = max(ablations["unconditional"]["semantic_mean"], ablations["shuffled_text"]["semantic_mean"])
    conditioning = {
        "conditional_cfg": full_metrics,
        **ablations,
        "advantage_over_best_unconditional_or_shuffled": full_metrics["semantic_mean"] - best_control,
        "threshold": thresholds["conditioning_advantage"],
    }
    conditioning["verified"] = conditioning["advantage_over_best_unconditional_or_shuffled"] >= thresholds["conditioning_advantage"]
    _write_json(result_dir / "conditioning_ablation.json", conditioning)
    print(f"  advantage={conditioning['advantage_over_best_unconditional_or_shuffled']:.4f} verified={conditioning['verified']}")

    # 4. CFG guidance trajectory
    print("Running CFG guidance trajectory...")
    trajectory = _save_trajectory(
        result_dir / "denoising_trajectory.png", model, schedule, tokenizer, vae,
        evaluation_rows[0]["prompt"], test.scaling_factor, config,
    )
    guidance = {
        "conditional_cfg_semantic_mean": full_metrics["semantic_mean"],
        "no_guidance_semantic_mean": ablations["no_guidance"]["semantic_mean"],
        **trajectory,
    }
    guidance["verified"] = (
        guidance["conditional_cfg_semantic_mean"] >= guidance["no_guidance_semantic_mean"]
        and guidance["exported_state_count"] >= 6
        and all(v > 0 for v in guidance["adjacent_latent_mae"])
    )
    _write_json(result_dir / "guidance_report.json", guidance)

    # 5. Reproducibility
    print("Running reproducibility check...")
    overfit_ids = [row["record_id"] for row in train.rows[: int(config["data"]["overfit_count"])]]
    overfit = LatentTextDataset(root, data_dir=config["output_data"], record_ids=overfit_ids, max_text_length=int(config["model"]["max_text_length"]))
    overfit_loader = DataLoader(overfit, batch_size=len(overfit), shuffle=False, collate_fn=collate_latent_text)
    digests, sequences = [], []
    for suffix in ("a", "b"):
        set_deterministic_seed(seed + 77, cpu_threads=int(config["cpu_threads"]))
        repro_model, _, repro_trainer = _new_trainer(result_dir / f"repro_{suffix}", config, tokenizer, epochs=2)
        repro_history = repro_trainer.fit(overfit_loader, overfit_loader)
        digests.append(_state_digest(repro_model))
        sequences.append(repro_history)
    reproducibility = {
        "state_sha256": digests,
        "state_equal": digests[0] == digests[1],
        "metrics_equal": sequences[0] == sequences[1],
    }
    reproducibility["verified"] = reproducibility["state_equal"] and reproducibility["metrics_equal"]
    _write_json(result_dir / "reproducibility_report.json", reproducibility)
    print(f"  reproducible={reproducibility['verified']}")

    # 6. Training curve + forward noise
    history = [json.loads(line) for line in (result_dir / "training" / "training_log.jsonl").read_text(encoding="utf-8").splitlines()]
    _save_training_curve(result_dir / "training_curve.png", history)
    _save_forward_noise(result_dir / "forward_noise_process.png", schedule, test, vae)

    # 7. Resource benchmark
    elapsed = perf_counter() - started
    process = psutil.Process()
    peak_rss = int(getattr(process.memory_info(), "peak_wset", process.memory_info().rss))
    resource = {
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "wall_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "parameter_count": denoising["parameter_count"],
        "limits": {
            "max_wall_seconds": thresholds["max_wall_seconds"],
            "max_peak_rss_bytes": thresholds["max_peak_rss_bytes"],
            "max_parameters": thresholds["max_parameters"],
        },
    }
    resource["verified"] = (
        elapsed <= thresholds["max_wall_seconds"]
        and peak_rss <= thresholds["max_peak_rss_bytes"]
        and denoising["parameter_count"] < thresholds["max_parameters"]
    )
    _write_json(result_dir / "resource_benchmark.json", resource)
    print(f"  wall={elapsed:.1f}s peak_rss={peak_rss // 1024 // 1024}MB params={denoising['parameter_count']} verified={resource['verified']}")

    # Summary
    summary = {
        "data_verified": data_report["verified"],
        "overfit_verified": json.loads((result_dir / "overfit_report.json").read_text(encoding="utf-8"))["verified"],
        "denoising_verified": denoising["verified"],
        "generation_verified": generation["verified"],
        "conditioning_verified": conditioning["verified"],
        "guidance_verified": guidance["verified"],
        "reproducibility_verified": reproducibility["verified"],
        "resource_verified": resource["verified"],
    }
    summary["verified"] = all(summary.values())
    _write_json(result_dir / "experiment_summary.json", summary)
    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()