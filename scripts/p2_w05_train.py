"""Train, evaluate and package the preregistered P2-W05 I001 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from mini_multimodal.asr import ASRDataset, ASRMetricAccumulator, CharacterCTCTokenizer, TinyASR, collate_asr, ctc_greedy_decode, forced_greedy_spans
from mini_multimodal.asr.metrics import _edit_distance
from mini_multimodal.speech.features import log_mel
from mini_multimodal.training.checkpoint import load_checkpoint, save_checkpoint


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def make_model(config: dict, vocabulary_size: int) -> TinyASR:
    return TinyASR(vocabulary_size=vocabulary_size, **config["model"])


def make_loader(root: Path, data_dir: str, ids: list[str], batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    dataset = ASRDataset(root, data_dir=data_dir, record_ids=ids)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, collate_fn=collate_asr, num_workers=0)


def loss_for(model: TinyASR, batch: dict, criterion) -> tuple[torch.Tensor, object]:
    output = model(batch["logmels"], batch["input_lengths"])
    loss = criterion(output.logits.log_softmax(-1).transpose(0, 1), batch["targets"], output.output_lengths, batch["target_lengths"])
    return loss, output


def evaluate(model, loader, tokenizer, criterion, *, include_predictions=False):
    model.eval(); accumulator = ASRMetricAccumulator(); total_loss = 0.0; total_items = 0; predictions = []
    with torch.no_grad():
        for batch in loader:
            loss, output = loss_for(model, batch, criterion)
            hypotheses = ctc_greedy_decode(output.logits, output.output_lengths, tokenizer)
            accumulator.update(batch["transcripts"], hypotheses)
            total_loss += float(loss) * len(hypotheses); total_items += len(hypotheses)
            if include_predictions:
                for index, (reference, hypothesis) in enumerate(zip(batch["transcripts"], hypotheses)):
                    predictions.append({
                        "record_id": batch["record_ids"][index], "reference": reference, "prediction": hypothesis,
                        "character_edits": _edit_distance(reference, hypothesis), "reference_characters": len(reference),
                        "cer": _edit_distance(reference, hypothesis) / max(1, len(reference)),
                        "pool": batch["pools"][index], "variant": batch["variants"][index]
                    })
    return {"ctc_loss": total_loss / max(1, total_items), **accumulator.compute()}, predictions


def train_epoch(model, loader, optimizer, criterion, clip):
    model.train(); total = 0.0; count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_for(model, batch, criterion)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip); optimizer.step()
        total += float(loss.detach()) * len(batch["record_ids"]); count += len(batch["record_ids"])
    return total / count


def short_repro(root, config, splits, tokenizer):
    ids = splits["regular"]["train"][:16]
    sequences, digests = [], []
    for _ in range(2):
        set_seed(config["seed"] + 17, config["cpu_threads"])
        model = make_model(config, len(tokenizer)); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        loader = make_loader(root, config["output_data"], ids, 8, shuffle=False, seed=config["seed"])
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); loss, _ = loss_for(model, batch, criterion); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        payload = b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
        sequences.append(losses); digests.append(hashlib.sha256(payload).hexdigest())
    return {"loss_sequences": sequences, "parameter_sha256": digests, "identical": sequences[0] == sequences[1] and digests[0] == digests[1]}


def perturb_item(root: Path, row: dict, mode: str, value: float, seed: int, tokenizer) -> dict:
    waveform, sample_rate = sf.read(root / row["audio_path"], dtype="float32")
    if mode == "noise":
        generator = np.random.default_rng(seed)
        power = float(np.mean(waveform ** 2)); noise_power = power / (10 ** (value / 10))
        waveform = waveform + generator.normal(0.0, np.sqrt(noise_power), len(waveform)).astype(np.float32)
    elif mode == "speed":
        new_length = max(512, int(round(len(waveform) / value)))
        waveform = np.interp(np.linspace(0, len(waveform) - 1, new_length), np.arange(len(waveform)), waveform).astype(np.float32)
    feature = log_mel(waveform, sample_rate).T
    feature = (feature - feature.mean(axis=0, keepdims=True)) / np.maximum(feature.std(axis=0, keepdims=True), 1e-5)
    return {"record_id": row["record_id"], "logmel": torch.from_numpy(feature), "target": torch.tensor(tokenizer.encode(row["normalized_transcript"])), "transcript": row["normalized_transcript"], "pool": row["pool"], "variant": f"{mode}_{value}", "duration_seconds": len(waveform) / sample_rate}


def evaluate_items(model, items, tokenizer, criterion, batch_size=8):
    class Items(torch.utils.data.Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, index): return items[index]
    return evaluate(model, DataLoader(Items(), batch_size=batch_size, collate_fn=collate_asr), tokenizer, criterion)[0]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); args = parser.parse_args()
    root = args.root.resolve(); config = json.loads((root / "configs/p2_w05_tiny_asr.json").read_text(encoding="utf-8"))
    result_root = root / config["output_results"]; branch = result_root / "iterations/I001/branches/baseline"; checkpoints = branch / "checkpoints"; results = branch / "results"
    checkpoints.mkdir(parents=True, exist_ok=True); results.mkdir(parents=True, exist_ok=True)
    splits = json.loads((root / config["output_data"] / "splits.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / config["output_data"] / "records.jsonl").read_text(encoding="utf-8").splitlines()]; by_id = {row["record_id"]: row for row in rows}
    tokenizer = CharacterCTCTokenizer(json.loads((root / config["output_data"] / "vocabulary.json").read_text(encoding="utf-8")))
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    set_seed(config["seed"], config["cpu_threads"]); started = time.perf_counter(); process = psutil.Process(os.getpid())

    write_json(results / "reproducibility_report.json", short_repro(root, config, splits, tokenizer))

    # Pre-registered small-set learning check.
    set_seed(config["seed"] + 1, config["cpu_threads"])
    overfit_model = make_model(config, len(tokenizer)); overfit_optimizer = torch.optim.AdamW(overfit_model.parameters(), lr=config["training"]["learning_rate"])
    overfit_loader = make_loader(root, config["output_data"], splits["regular"]["train"][:config["training"]["overfit_count"]], config["training"]["overfit_count"], shuffle=False, seed=config["seed"])
    overfit_losses = []
    for _ in range(config["training"]["overfit_epochs"]): overfit_losses.append(train_epoch(overfit_model, overfit_loader, overfit_optimizer, criterion, config["training"]["gradient_clip_norm"]))
    overfit_metrics, _ = evaluate(overfit_model, overfit_loader, tokenizer, criterion)
    write_json(results / "overfit_report.json", {"epochs": len(overfit_losses), "initial_loss": overfit_losses[0], "final_loss": overfit_losses[-1], **overfit_metrics})

    set_seed(config["seed"], config["cpu_threads"])
    model = make_model(config, len(tokenizer)); optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    train_loader = make_loader(root, config["output_data"], splits["regular"]["train"], config["training"]["batch_size"], shuffle=True, seed=config["seed"])
    validation_loader = make_loader(root, config["output_data"], splits["regular"]["validation"], config["training"]["batch_size"], shuffle=False, seed=config["seed"])
    history, best_cer, patience = [], float("inf"), 0
    for epoch in range(config["training"]["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config["training"]["gradient_clip_norm"])
        validation, _ = evaluate(model, validation_loader, tokenizer, criterion)
        history.append({"epoch": epoch, "train_ctc_loss": train_loss, "validation_ctc_loss": validation["ctc_loss"], "validation_cer": validation["cer"], "validation_domain_wer": validation["domain_wer"]})
        save_checkpoint(checkpoints / "last.pt", model=model, optimizer=optimizer, epoch=epoch, best_metric=min(best_cer, validation["cer"]), config=config)
        if validation["cer"] < best_cer:
            best_cer = validation["cer"]; patience = 0
            save_checkpoint(checkpoints / "best.pt", model=model, optimizer=optimizer, epoch=epoch, best_metric=best_cer, config=config)
        else:
            patience += 1
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if patience >= config["training"]["early_stopping_patience"]: break
    load_checkpoint(checkpoints / "best.pt", model=model, optimizer=None, restore_rng=False)
    write_json(results / "training_history.json", history)
    plt.figure(figsize=(7, 4)); plt.plot([x["epoch"] for x in history], [x["train_ctc_loss"] for x in history], label="train CTC"); plt.plot([x["epoch"] for x in history], [x["validation_ctc_loss"] for x in history], label="validation CTC"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout(); plt.savefig(results / "training_curve.png", dpi=150); plt.close()

    metrics = {}
    for group, split_name in (("regular_validation", ("regular", "validation")), ("regular_test", ("regular", "test")), ("asset_nonoverlap_validation", ("asset_nonoverlap", "validation")), ("asset_nonoverlap_test", ("asset_nonoverlap", "test")), ("text_nonoverlap_validation", ("text_nonoverlap", "validation")), ("text_nonoverlap_test", ("text_nonoverlap", "test"))):
        ids = splits[split_name[0]][split_name[1]]; loader = make_loader(root, config["output_data"], ids, config["training"]["batch_size"], shuffle=False, seed=config["seed"])
        metrics[group], predictions = evaluate(model, loader, tokenizer, criterion, include_predictions=group == "regular_test")
        if predictions:
            (results / "test_predictions.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in predictions), encoding="utf-8", newline="\n")
    for name, ids in splits["challenge"].items():
        loader = make_loader(root, config["output_data"], ids, config["training"]["batch_size"], shuffle=False, seed=config["seed"])
        metrics[name] = evaluate(model, loader, tokenizer, criterion)[0]
    write_json(results / "evaluation_metrics.json", metrics)

    # Length groups based on fixed test-set duration tertiles.
    test_ids = splits["regular"]["test"]; ordered = sorted(test_ids, key=lambda item: by_id[item]["duration_seconds"]); groups = np.array_split(ordered, 3)
    length_metrics = {}
    for name, ids in zip(("short", "medium", "long"), groups):
        loader = make_loader(root, config["output_data"], list(ids), config["training"]["batch_size"], shuffle=False, seed=config["seed"])
        length_metrics[name] = {**evaluate(model, loader, tokenizer, criterion)[0], "min_duration": min(by_id[item]["duration_seconds"] for item in ids), "max_duration": max(by_id[item]["duration_seconds"] for item in ids)}
    write_json(results / "length_group_metrics.json", length_metrics)

    stress_ids = splits["text_nonoverlap"]["test"][:config["evaluation"]["stress_sample_count"]]
    clean_loader = make_loader(root, config["output_data"], stress_ids, 8, shuffle=False, seed=config["seed"]); clean = evaluate(model, clean_loader, tokenizer, criterion)[0]
    stress = {"clean": clean}
    noise_items = [perturb_item(root, by_id[item], "noise", config["evaluation"]["noise_snr_db"], config["seed"] + index, tokenizer) for index, item in enumerate(stress_ids)]
    stress["noise_15db"] = evaluate_items(model, noise_items, tokenizer, criterion)
    for factor in config["evaluation"]["speed_factors"]:
        items = [perturb_item(root, by_id[item], "speed", factor, config["seed"] + index, tokenizer) for index, item in enumerate(stress_ids)]
        stress[f"speed_{factor}"] = evaluate_items(model, items, tokenizer, criterion)
    for value in stress.values(): value["cer_increase_over_clean"] = value["cer"] - clean["cer"]
    write_json(results / "robustness_metrics.json", stress)

    # Fixed correct and error examples with model-path evidence.
    predictions = [json.loads(line) for line in (results / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = sorted([item for item in predictions if item["character_edits"] > 0], key=lambda x: (-x["cer"], x["record_id"]))[:5]
    selected = sorted([item for item in predictions if item["character_edits"] == 0], key=lambda x: x["record_id"])[: max(5, 10 - len(errors))] + errors
    example_records = []
    for item in selected:
        dataset = ASRDataset(root, data_dir=config["output_data"], record_ids=[item["record_id"]]); batch = collate_asr([dataset[0]])
        with torch.no_grad(): output = model(batch["logmels"], batch["input_lengths"], return_attention=True)
        item = dict(item); item["greedy_spans"] = forced_greedy_spans(output.logits[0], int(output.output_lengths[0]), tokenizer); example_records.append(item)
    write_json(results / "capability_examples.json", example_records)
    lines = ["# P2-W05 TinyASR直观能力样例", "", "> 全部样例由当前检查点在固定测试集上机器生成；前5条优先为正确样例，后5条为错误样例。", ""]
    for index, item in enumerate(example_records, 1):
        lines += [f"## 样例{index}：{'正确' if item['character_edits']==0 else '错误'}", "", f"- 记录ID：`{item['record_id']}`", f"- 参考：{item['reference']}", f"- 预测：{item['prediction'] or '（空）'}", f"- 该句CER：{item['cer']:.4f}", f"- 贪心路径非空字符段数：{len(item['greedy_spans'])}", ""]
    (results / "capability_examples.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    elapsed = time.perf_counter() - started; parameter_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    checkpoint_manifest = {name: {"bytes": (checkpoints / name).stat().st_size, "sha256": sha256(checkpoints / name)} for name in ("best.pt", "last.pt")}
    write_json(checkpoints / "checkpoint_manifest.json", checkpoint_manifest)
    write_json(results / "resource_benchmark.json", {"wall_seconds": elapsed, "peak_rss_bytes": process.memory_info().peak_wset if hasattr(process.memory_info(), "peak_wset") else process.memory_info().rss, "trainable_parameters": parameter_count, "cpu_threads": config["cpu_threads"]})
    write_json(results / "run_summary.json", {"best_epoch": int(torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=False)["epoch"]), "best_validation_cer": best_cer, "epochs_completed": len(history), "checkpoint_manifest": checkpoint_manifest})


if __name__ == "__main__":
    main()
