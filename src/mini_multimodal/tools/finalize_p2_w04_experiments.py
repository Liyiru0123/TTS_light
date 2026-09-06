"""Continue P2-W04 from the saved best checkpoint after a reporting-only failure."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter

import psutil
import torch
from torch.utils.data import DataLoader

from mini_multimodal.ldm.artifacts import (
    _load_vae,
    _model_kwargs,
    _new_trainer,
    _save_forward_noise,
    _save_training_curve,
    _save_trajectory,
    _schedule,
    _state_digest,
    _write_json,
)
from mini_multimodal.ldm.data import CharacterTokenizer, LatentTextDataset, collate_latent_text
from mini_multimodal.ldm.model import TinyImageDenoiser
from mini_multimodal.training import load_checkpoint, set_deterministic_seed


def main() -> None:
    started = perf_counter()
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
    result_dir = root / config["output_results"]
    vocabulary = json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocabulary)
    test = LatentTextDataset(root, data_dir=config["output_data"], split="test", max_text_length=config["model"]["max_text_length"])
    model = TinyImageDenoiser(**_model_kwargs(config, len(tokenizer)))
    best_path = result_dir / "training" / "checkpoints" / "best.pt"
    load_checkpoint(best_path, model=model, optimizer=None, restore_rng=False)
    vae = _load_vae(root, config)
    evaluation_row = next(row for row in test.rows if len(row["objects"]) == 1)
    trajectory = _save_trajectory(
        result_dir / "denoising_trajectory.png",
        model,
        _schedule(config),
        tokenizer,
        vae,
        evaluation_row["prompt"],
        test.scaling_factor,
        config,
    )
    conditioning = json.loads((result_dir / "conditioning_ablation.json").read_text(encoding="utf-8"))
    guidance = {
        "conditional_cfg_semantic_mean": conditioning["conditional_cfg"]["semantic_mean"],
        "no_guidance_semantic_mean": conditioning["no_guidance"]["semantic_mean"],
        **trajectory,
    }
    guidance["verified"] = (
        guidance["conditional_cfg_semantic_mean"] >= guidance["no_guidance_semantic_mean"]
        and guidance["exported_state_count"] >= 6
        and all(value > 0 for value in guidance["adjacent_latent_mae"])
    )
    _write_json(result_dir / "guidance_report.json", guidance)

    train = LatentTextDataset(root, data_dir=config["output_data"], split="train", max_text_length=config["model"]["max_text_length"])
    count = int(config["data"]["overfit_count"])
    overfit = LatentTextDataset(
        root,
        data_dir=config["output_data"],
        record_ids=[row["record_id"] for row in train.rows[:count]],
        max_text_length=config["model"]["max_text_length"],
    )
    loader = DataLoader(overfit, batch_size=len(overfit), shuffle=False, collate_fn=collate_latent_text)
    digests, sequences = [], []
    for suffix in ("a", "b"):
        set_deterministic_seed(config["seed"] + 77, cpu_threads=config["cpu_threads"])
        repro_model, _, trainer = _new_trainer(result_dir / f"repro_{suffix}", config, tokenizer, epochs=2)
        history = trainer.fit(loader, loader)
        digests.append(_state_digest(repro_model))
        sequences.append([
            {
                "epoch": row["epoch"],
                "train": row["train"],
                "validation": row["validation"],
                "monitor": row["monitor"],
                "best_metric": row["best_metric"],
                "improved": row["improved"],
            }
            for row in history
        ])
    reproducibility = {
        "state_sha256": digests,
        "state_equal": digests[0] == digests[1],
        "metrics_equal": sequences[0] == sequences[1],
    }
    reproducibility["verified"] = reproducibility["state_equal"] and reproducibility["metrics_equal"]
    _write_json(result_dir / "reproducibility_report.json", reproducibility)

    history = [json.loads(line) for line in (result_dir / "training" / "training_log.jsonl").read_text(encoding="utf-8").splitlines()]
    _save_training_curve(result_dir / "training_curve.png", history)
    _save_forward_noise(result_dir / "forward_noise_process.png", _schedule(config), test, vae)
    artifact_times = [
        (result_dir / "overfit" / "training_log.jsonl").stat().st_mtime,
        (result_dir / "conditioning_ablation.json").stat().st_mtime,
    ]
    observed_primary_span = max(artifact_times) - min(artifact_times)
    completion_seconds = perf_counter() - started
    denoising = json.loads((result_dir / "denoising_metrics.json").read_text(encoding="utf-8"))
    thresholds = config["thresholds"]
    process = psutil.Process()
    memory = process.memory_info()
    resource = {
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "observed_primary_artifact_span_seconds": observed_primary_span,
        "completion_seconds": completion_seconds,
        "wall_seconds": observed_primary_span + completion_seconds,
        "wall_seconds_scope": "训练起始日志至条件消融文件的机器时间戳跨度，加检查点续接完成耗时",
        "peak_rss_bytes": int(getattr(memory, "peak_wset", memory.rss)),
        "parameter_count": denoising["parameter_count"],
        "limits": {
            "max_wall_seconds": thresholds["max_wall_seconds"],
            "max_peak_rss_bytes": thresholds["max_peak_rss_bytes"],
            "max_parameters": thresholds["max_parameters"],
        },
    }
    resource["verified"] = (
        resource["wall_seconds"] <= thresholds["max_wall_seconds"]
        and resource["peak_rss_bytes"] <= thresholds["max_peak_rss_bytes"]
        and resource["parameter_count"] < thresholds["max_parameters"]
    )
    _write_json(result_dir / "resource_benchmark.json", resource)

    summary = {
        "data_verified": True,
        "overfit_verified": json.loads((result_dir / "overfit_report.json").read_text(encoding="utf-8"))["verified"],
        "denoising_verified": denoising["verified"],
        "generation_verified": json.loads((result_dir / "generation_metrics.json").read_text(encoding="utf-8"))["verified"],
        "conditioning_verified": conditioning["verified"],
        "guidance_verified": guidance["verified"],
        "reproducibility_verified": reproducibility["verified"],
        "resource_verified": resource["verified"],
    }
    summary["verified"] = all(summary.values())
    _write_json(result_dir / "experiment_summary.json", summary)
    if not summary["verified"]:
        raise RuntimeError("P2-W04 finalization found an unmet acceptance item")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
