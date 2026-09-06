"""P3-W03 evaluation: full TinyTTS pipeline on test units.

Gate C metrics:
- predicted duration vs ground-truth duration (mean relative error, overflow rate)
- CER: generated audio (predicted duration) back-translated by ASR
- CER gt: generated audio (ground-truth duration)
- CER baseline: frozen-chain direct decode of posterior mean (chain ceiling)
- CER gap = CER_pred - CER_gt (duration planning effect, MODEL-SPG-002)
- CER gap to baseline = CER_gt - CER_baseline (CFM quality, MODEL-SPG-001)

Evidence -> results/P3-W03_tiny_tts/evaluation/
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

from mini_multimodal.asr.metrics import ASRMetricAccumulator  # noqa: E402
from mini_multimodal.tts import TinySpeechFlow, DurationOverflowError  # noqa: E402
from mini_multimodal.tts.asr_eval import load_asr, transcribe  # noqa: E402
from mini_multimodal.tts.cfm import text_to_token_ids, ode_infer  # noqa: E402
from mini_multimodal.tts.frozen_chain import load_frozen_chain  # noqa: E402


def load_json(path: Path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_unit_mel(unit: dict, root: Path) -> torch.Tensor:
    mel80 = np.load(root / unit["source_logmel80_path"], allow_pickle=False)
    return torch.tensor(mel80[:, unit["mel_start"] : unit["mel_end"]], dtype=torch.float32)


def main() -> int:
    cfg = load_json(ROOT / "configs" / "p3_w03_tiny_tts.json")
    out_dir = ROOT / cfg["output_results"] / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg["seed"]))

    chain = load_frozen_chain(cfg, ROOT)
    asr, tokenizer = load_asr(ROOT / cfg["asr_package"])
    with io.open(ROOT / cfg["output_data"] / "vocabulary.json", encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]

    ckpt = torch.load(ROOT / cfg["output_results"] / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model = TinySpeechFlow(
        vocab_size=len(vocab),
        text_cfg=cfg["text_encoder"],
        duration_cfg=cfg["duration"],
        condition_cfg=cfg["condition"],
        unet_cfg=cfg["unet"],
        t_max=int(cfg["container"]["t_max"]),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    manifest = [json.loads(l) for l in io.open(ROOT / cfg["output_data"] / "unit_manifest.jsonl", encoding="utf-8")]
    test_units = [u for u in manifest if u["split"] == "test"]

    def generate(unit, durations):
        """Generate waveform with given (gt/predicted) durations. Returns (wave, T_valid)."""
        token_ids = torch.tensor(text_to_token_ids(unit["token_chars"], vocab), dtype=torch.long)
        h = model.encode_text(token_ids)
        H = model.regulator(h, durations)
        C = model.condition_frames(H)
        T_valid = int(durations.sum().item())
        x1 = ode_infer(model, C, T_valid, t_max=64, ode_steps=20)
        latent = x1.reshape(4, 20, 64)
        mel = chain.decode_latent(latent)
        valid_mel = min(4 * T_valid, 256)
        wave = chain.synthesize(mel[:, :valid_mel])[0, 0].numpy()
        return wave, T_valid

    acc_pred = ASRMetricAccumulator()
    acc_gt = ASRMetricAccumulator()
    acc_base = ASRMetricAccumulator()
    rows = []
    n_overflow = 0
    n_skip = 0
    dur_errors = []
    sample_count = min(len(test_units), 300)
    for u in test_units[:sample_count]:
        token_ids = torch.tensor(text_to_token_ids(u["token_chars"], vocab), dtype=torch.long)
        h = model.encode_text(token_ids)
        d_pred = model.duration.predict_durations(h).long()  # [N] int
        d_gt = torch.tensor(u["token_durations"], dtype=torch.long)
        T_pred = int(d_pred.sum().item())
        T_gt = int(d_gt.sum().item())
        # duration error (per-token, over aligned positions)
        n = min(len(d_pred), len(d_gt))
        if n > 0:
            rel_err = ((d_pred[:n].float() - d_gt[:n].float()).abs() / (d_gt[:n].float() + 1e-9)).mean().item()
            dur_errors.append(rel_err)
        if T_pred > 64 or T_pred < 4:
            n_overflow += 1
            n_skip += 1
            continue
        try:
            wave_pred, _ = generate(u, d_pred)
            wave_gt, _ = generate(u, d_gt)
        except DurationOverflowError:
            n_overflow += 1
            n_skip += 1
            continue
        hyp_pred = transcribe(wave_pred, asr, tokenizer)
        hyp_gt = transcribe(wave_gt, asr, tokenizer)
        # baseline: frozen chain direct decode of posterior mean (chain ceiling)
        mel = load_unit_mel(u, ROOT)
        mean = chain.encode_unit(mel)  # [4,20,64]
        mel_base = chain.decode_latent(mean)
        valid_base = min(4 * u["t_valid"], 256)
        wave_base = chain.synthesize(mel_base[:, :valid_base])[0, 0].numpy()
        hyp_base = transcribe(wave_base, asr, tokenizer)
        acc_pred.update([u["text"]], [hyp_pred])
        acc_gt.update([u["text"]], [hyp_gt])
        acc_base.update([u["text"]], [hyp_base])
        rows.append({
            "unit_id": u["unit_id"], "text": u["text"], "hyp_pred": hyp_pred, "hyp_gt": hyp_gt,
            "hyp_base": hyp_base,
            "T_gt": T_gt, "T_pred": T_pred, "d_gt": u["token_durations"],
            "d_pred": d_pred.tolist(),
        })

    cer_pred = acc_pred.compute()["cer"]
    cer_gt = acc_gt.compute()["cer"]
    cer_base = acc_base.compute()["cer"]
    cer_gap = cer_pred - cer_gt
    cer_gap_to_baseline = cer_gt - cer_base
    report = {
        "schema_version": 1, "work_package": "P3-W03",
        "n_evaluated": sample_count, "n_skipped": n_skip, "n_overflow": n_overflow,
        "cer_pred_duration": round(cer_pred, 4), "cer_gt_duration": round(cer_gt, 4),
        "cer_baseline": round(cer_base, 4),
        "cer_gap": round(cer_gap, 4), "cer_gap_to_baseline": round(cer_gap_to_baseline, 4),
        "duration_mean_rel_error": round(float(np.mean(dur_errors)) if dur_errors else float("nan"), 4),
        "duration_overflow_rate": n_overflow / sample_count if sample_count else 0.0,
        "thresholds": {
            "duration_mean_rel_error_max": cfg["thresholds"]["duration_mean_rel_error_max"],
            "duration_cer_gap_max": cfg["thresholds"]["duration_cer_gap_max"],
            "cer_gap_to_baseline_max": cfg["thresholds"]["cer_gap_to_baseline_max"],
        },
        "rows": rows,
    }
    dump_json(out_dir / "evaluation_report.json", report)
    print("=== P3-W03 evaluation ===")
    print(f"evaluated {len(rows)}/{sample_count} test units, skipped {n_skip} (overflow {n_overflow})")
    print(f"CER_pred_duration: {cer_pred:.4f}")
    print(f"CER_gt_duration: {cer_gt:.4f}")
    print(f"CER_baseline (frozen chain): {cer_base:.4f}")
    print(f"CER_gap (pred - gt): {cer_gap:+.4f} (threshold {cfg['thresholds']['duration_cer_gap_max']})")
    print(f"CER_gap_to_baseline (gt - baseline): {cer_gap_to_baseline:+.4f} (threshold {cfg['thresholds']['cer_gap_to_baseline_max']})")
    print(f"duration_mean_rel_error: {report['duration_mean_rel_error']:.4f} (threshold {cfg['thresholds']['duration_mean_rel_error_max']})")
    print(f"evidence -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
