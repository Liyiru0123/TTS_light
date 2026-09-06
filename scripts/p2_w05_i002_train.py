"""Train and evaluate the preregistered P2-W05 I002 robust fine-tune."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from mini_multimodal.asr import ASRDataset, CharacterCTCTokenizer, TinyASR, collate_asr
from mini_multimodal.training.checkpoint import load_checkpoint, save_checkpoint
from p2_w05_train import evaluate, evaluate_items, loss_for, perturb_item, set_seed, sha256, write_json


ROOT = Path(__file__).resolve().parents[1]


def loader(data_dir: str, ids: list[str], batch_size: int, *, shuffle: bool, seed: int, augmentation: dict | None = None) -> DataLoader:
    dataset = ASRDataset(ROOT, data_dir=data_dir, record_ids=ids, augmentation=augmentation, augmentation_seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(seed), collate_fn=collate_asr, num_workers=0)


def train_epoch(model, data_loader, optimizer, criterion, clip: float, epoch: int) -> float:
    if hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    model.train(); total = 0.0; count = 0
    for batch in data_loader:
        optimizer.zero_grad(set_to_none=True); loss, _ = loss_for(model, batch, criterion)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip); optimizer.step()
        total += float(loss.detach()) * len(batch["record_ids"]); count += len(batch["record_ids"])
    return total / max(1, count)


def make_model(config: dict, vocabulary_size: int) -> TinyASR:
    return TinyASR(vocabulary_size=vocabulary_size, **config["model"])


def augmented_repro(config, splits, tokenizer, augmentation):
    ids = splits["train_multistyle"][:16]; runs = []
    for _ in range(2):
        set_seed(config["seed"] + 17, config["cpu_threads"])
        model = make_model(config, len(tokenizer)); load_checkpoint(ROOT / config["initial_checkpoint"], model=model, restore_rng=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
        criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        data_loader = loader(config["output_data"], ids, 8, shuffle=False, seed=config["seed"] + 17, augmentation=augmentation)
        losses = []
        for epoch in range(2):
            losses.append(train_epoch(model, data_loader, optimizer, criterion, config["training"]["gradient_clip_norm"], epoch))
        payload = b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
        runs.append({"losses": losses, "parameter_sha256": hashlib.sha256(payload).hexdigest()})
    return {"runs": runs, "identical": runs[0] == runs[1]}


def main() -> None:
    config = json.loads((ROOT / "configs/p2_w05_i002.json").read_text(encoding="utf-8"))
    source_splits = json.loads((ROOT / config["source_asr_data"] / "splits.json").read_text(encoding="utf-8"))
    i002_splits = json.loads((ROOT / config["output_data"] / "splits.json").read_text(encoding="utf-8"))
    source_rows = [json.loads(line) for line in (ROOT / config["source_asr_data"] / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    source_by_id = {row["record_id"]: row for row in source_rows}
    vocabulary = json.loads((ROOT / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterCTCTokenizer(vocabulary)
    branch = ROOT / config["output_results"] / "iterations/I002/branches/robust_finetune"
    checkpoints, metrics_dir, predictions_dir, examples_dir, logs_dir, tests_dir = [branch / name for name in ("checkpoints", "metrics", "predictions", "examples", "logs", "tests")]
    for path in (checkpoints, metrics_dir, predictions_dir, examples_dir, logs_dir, tests_dir): path.mkdir(parents=True, exist_ok=True)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    augmentation = {
        "probability": config["augmentation"]["specaugment_probability"],
        "frequency_mask_count": config["augmentation"]["frequency_mask_count"],
        "frequency_mask_max_width": config["augmentation"]["frequency_mask_max_width"],
        "time_mask_count": config["augmentation"]["time_mask_count"],
        "time_mask_max_width": config["augmentation"]["time_mask_max_width"]
    }
    set_seed(config["seed"], config["cpu_threads"]); started = time.perf_counter(); process = psutil.Process(os.getpid())
    write_json(metrics_dir / "reproducibility_report.json", augmented_repro(config, i002_splits, tokenizer, augmentation))

    # Scratch small-set diagnostic with the preregistered larger update budget.
    set_seed(config["seed"] + 1, config["cpu_threads"])
    overfit_model = make_model(config, len(tokenizer)); overfit_optimizer = torch.optim.AdamW(overfit_model.parameters(), lr=0.001)
    overfit_loader = loader(config["source_asr_data"], source_splits["regular"]["train"][:config["training"]["overfit_count"]], config["training"]["overfit_count"], shuffle=False, seed=config["seed"] + 1)
    overfit_losses = []
    for epoch in range(config["training"]["overfit_epochs"]):
        overfit_losses.append(train_epoch(overfit_model, overfit_loader, overfit_optimizer, criterion, config["training"]["gradient_clip_norm"], epoch))
    overfit_metrics, _ = evaluate(overfit_model, overfit_loader, tokenizer, criterion)
    write_json(metrics_dir / "overfit_report.json", {"updates": len(overfit_losses), "initial_loss": overfit_losses[0], "final_loss": overfit_losses[-1], **overfit_metrics})

    set_seed(config["seed"], config["cpu_threads"])
    model = make_model(config, len(tokenizer)); initial = load_checkpoint(ROOT / config["initial_checkpoint"], model=model, restore_rng=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    train_loader = loader(config["output_data"], i002_splits["train_multistyle"], config["training"]["batch_size"], shuffle=True, seed=config["seed"], augmentation=augmentation)
    clean_validation_loader = loader(config["source_asr_data"], source_splits["regular"]["validation"], config["training"]["batch_size"], shuffle=False, seed=config["seed"])
    noise_validation_loader = loader(config["output_data"], i002_splits["validation_noise_15db"], config["training"]["batch_size"], shuffle=False, seed=config["seed"])
    history, best_score, patience = [], float("inf"), 0
    for epoch in range(config["training"]["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config["training"]["gradient_clip_norm"], epoch)
        clean_metrics, _ = evaluate(model, clean_validation_loader, tokenizer, criterion)
        noise_metrics, _ = evaluate(model, noise_validation_loader, tokenizer, criterion)
        score = config["selection"]["clean_validation_weight"] * clean_metrics["cer"] + config["selection"]["noise_validation_weight"] * noise_metrics["cer"]
        row = {"epoch": epoch, "train_ctc_loss": train_loss, "clean_validation_cer": clean_metrics["cer"], "noise15_validation_cer": noise_metrics["cer"], "selection_score": score}
        history.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        save_checkpoint(checkpoints / "last.pt", model=model, optimizer=optimizer, epoch=epoch, best_metric=min(best_score, score), config=config)
        if score < best_score:
            best_score = score; patience = 0
            save_checkpoint(checkpoints / "best.pt", model=model, optimizer=optimizer, epoch=epoch, best_metric=score, config=config)
        else: patience += 1
        if patience >= config["training"]["early_stopping_patience"]: break
    load_checkpoint(checkpoints / "best.pt", model=model, restore_rng=False)
    write_json(logs_dir / "training_history.json", {"initial_checkpoint_epoch": initial["epoch"], "history": history})
    plt.figure(figsize=(7,4)); plt.plot([x["epoch"] for x in history],[x["clean_validation_cer"] for x in history],label="clean CER"); plt.plot([x["epoch"] for x in history],[x["noise15_validation_cer"] for x in history],label="noise15 CER"); plt.xlabel("epoch"); plt.ylabel("CER"); plt.legend(); plt.tight_layout(); plt.savefig(logs_dir / "training_curve.png", dpi=150); plt.close()

    evaluation = {}
    views = (("regular_validation","regular","validation"),("regular_test","regular","test"),("asset_nonoverlap_test","asset_nonoverlap","test"),("text_nonoverlap_test","text_nonoverlap","test"))
    clean_predictions = []
    for name, group, split in views:
        data_loader = loader(config["source_asr_data"], source_splits[group][split], config["training"]["batch_size"], shuffle=False, seed=config["seed"])
        evaluation[name], predictions = evaluate(model, data_loader, tokenizer, criterion, include_predictions=name=="regular_test")
        if predictions: clean_predictions = predictions
    for name, ids in source_splits["challenge"].items():
        evaluation[name] = evaluate(model, loader(config["source_asr_data"], ids, config["training"]["batch_size"], shuffle=False, seed=config["seed"]), tokenizer, criterion)[0]
    write_json(metrics_dir / "evaluation_metrics.json", evaluation)
    (predictions_dir / "clean_test_predictions.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True)+"\n" for item in clean_predictions), encoding="utf-8", newline="\n")

    stress_ids = source_splits["text_nonoverlap"]["test"][:config["evaluation"]["stress_sample_count"]]
    clean_stress = evaluate(model, loader(config["source_asr_data"], stress_ids, 8, shuffle=False, seed=config["evaluation"]["stress_seed"]), tokenizer, criterion)[0]
    robustness = {"clean": clean_stress}; noise_predictions = {}
    for snr in [*config["evaluation"]["additional_noise_snr_db"], config["evaluation"]["noise_snr_db"]]:
        items = [perturb_item(ROOT, source_by_id[item], "noise", snr, config["evaluation"]["stress_seed"] + index, tokenizer) for index,item in enumerate(stress_ids)]
        key = f"noise_{snr:g}db"; robustness[key] = evaluate_items(model, items, tokenizer, criterion)
        # Store per-sample outputs for the primary 15 dB challenge.
        if snr == config["evaluation"]["noise_snr_db"]:
            data_loader = DataLoader(items, batch_size=8, collate_fn=collate_asr)
            from p2_w05_train import evaluate as evaluate_predictions
            _, preds = evaluate_predictions(model, data_loader, tokenizer, criterion, include_predictions=True); noise_predictions = preds
    for factor in config["evaluation"]["speed_factors"]:
        items = [perturb_item(ROOT, source_by_id[item], "speed", factor, config["evaluation"]["stress_seed"] + index, tokenizer) for index,item in enumerate(stress_ids)]
        robustness[f"speed_{factor}"] = evaluate_items(model, items, tokenizer, criterion)
    for value in robustness.values(): value["cer_increase_over_clean"] = value["cer"] - clean_stress["cer"]
    write_json(metrics_dir / "robustness_metrics.json", robustness)
    (predictions_dir / "noise15_test_predictions.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True)+"\n" for item in noise_predictions), encoding="utf-8", newline="\n")

    ordered = sorted(source_splits["regular"]["test"], key=lambda item: source_by_id[item]["duration_seconds"]); length_metrics = {}
    for name, ids in zip(("short","medium","long"), np.array_split(ordered,3)):
        result = evaluate(model, loader(config["source_asr_data"], list(ids), config["training"]["batch_size"], shuffle=False, seed=config["seed"]), tokenizer, criterion)[0]
        length_metrics[name] = {**result, "min_duration": min(source_by_id[item]["duration_seconds"] for item in ids), "max_duration": max(source_by_id[item]["duration_seconds"] for item in ids)}
    write_json(metrics_dir / "length_group_metrics.json", length_metrics)

    elapsed = time.perf_counter()-started; memory = process.memory_info(); parameter_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    write_json(metrics_dir / "resource_benchmark.json", {"wall_seconds":elapsed,"peak_rss_bytes":memory.peak_wset if hasattr(memory,"peak_wset") else memory.rss,"trainable_parameters":parameter_count,"cpu_threads":config["cpu_threads"]})
    manifest = {name:{"bytes":(checkpoints/name).stat().st_size,"sha256":sha256(checkpoints/name)} for name in ("best.pt","last.pt")}
    write_json(checkpoints / "checkpoint_manifest.json", manifest)
    best_state = torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=False)
    write_json(metrics_dir / "run_summary.json", {"best_epoch":best_state["epoch"],"best_selection_score":best_state["best_metric"],"epochs_completed":len(history),"checkpoint_manifest":manifest})


if __name__ == "__main__":
    main()
