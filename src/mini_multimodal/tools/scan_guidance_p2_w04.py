"""Temporary diagnostic: scan guidance_scale on the existing best.pt (no retrain).

Prints single-object and composition color/shape/position accuracy for several
guidance scales so we can pick one that clears the shape >= 0.70 threshold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from mini_multimodal.ldm.artifacts import (  # noqa: E402
    _aggregate_generation,
    _load_shape_model,
    _load_vae,
    _model_kwargs,
    _schedule,
    generate_records,
)
from mini_multimodal.ldm.data import CharacterTokenizer, LatentTextDataset  # noqa: E402
from mini_multimodal.ldm.model import TinyImageDenoiser, build_position_token_ids  # noqa: E402
from mini_multimodal.training import load_checkpoint, set_deterministic_seed  # noqa: E402

GUIDANCE_CANDIDATES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs" / "p2_w04_tiny_image_ldm.json").read_text(encoding="utf-8"))
    seed = int(config["seed"])
    set_deterministic_seed(seed, cpu_threads=int(config["cpu_threads"]))
    vocabulary = json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocabulary)
    test = LatentTextDataset(root, data_dir=config["output_data"], split="test", max_text_length=int(config["model"]["max_text_length"]))

    model = TinyImageDenoiser(**_model_kwargs(config, len(tokenizer), vocabulary))
    best_path = root / config["output_results"] / "training" / "checkpoints" / "best.pt"
    state = load_checkpoint(best_path, model=model, optimizer=None, restore_rng=False)
    model.eval()
    print(f"Loaded best.pt epoch={state['epoch']} metric={state['best_metric']:.5f}")

    schedule = _schedule(config)
    vae = _load_vae(root, config)
    shape_model = _load_shape_model(root, config)
    pos_strength = float(config["diffusion"].get("position_bias_strength", 0.0))
    sample_steps = int(config["diffusion"]["sample_steps"])
    scaling = test.scaling_factor

    single_rows = [r for r in test.rows if len(r["objects"]) == 1 and r["variant"] != "compositional"][: int(config["data"]["generation_evaluation_count"])]
    comp_rows = [r for r in test.rows if len(r["objects"]) == 1 and r["variant"] == "compositional"][: int(config["data"]["composition_evaluation_count"])]
    print(f"single_rows={len(single_rows)} comp_rows={len(comp_rows)}")

    print(f"{'guid':>5} | {'s_color':>7} {'s_shape':>7} {'s_pos':>7} {'s_full':>7} | {'c_color':>7} {'c_shape':>7} {'c_pos':>7}")
    for g in GUIDANCE_CANDIDATES:
        recs = generate_records(
            model, schedule, tokenizer, vae, shape_model, single_rows + comp_rows,
            scaling_factor=scaling, sample_steps=sample_steps, guidance_scale=g,
            seed=seed + 200, position_bias_strength=pos_strength,
        )
        s = _aggregate_generation(recs[: len(single_rows)])
        c = _aggregate_generation(recs[len(single_rows):])
        print(f"{g:>5} | {s['color_accuracy']:>7.3f} {s['shape_accuracy']:>7.3f} {s['position_accuracy']:>7.3f} {s['fully_correct_rate']:>7.3f} | {c['color_accuracy']:>7.3f} {c['shape_accuracy']:>7.3f} {c['position_accuracy']:>7.3f}")


if __name__ == "__main__":
    main()
