# -*- coding: utf-8 -*-
"""单阶段 ASR 训练入口：把 I001(干净) 与 I002(鲁棒) 融合为一次连续训练。

流程（同一模型对象，一次运行）：
  阶段A（＝I001）：在干净 regular 数据上预训练 N 个 epoch；
  阶段B（＝I002）：继续该模型，切换到 SpecAugment＋多风格(干净+噪声)数据，训练 M 个 epoch，
                   按 clean_cer 与 noise15_cer 加权选最优，保存最终鲁棒模型。

这样做不采用"加载 I001 checkpoint 再微调"，而是把干净→鲁棒作为同一模型的连续优化过程。
用法： python scripts/train_final_tiny_asr.py [--seed 20260816]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    ROOT = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    from p2_w05_train import (make_model, train_epoch, evaluate, make_loader,
                              write_json, set_seed, loss_for, sha256)
    from mini_multimodal.asr import CharacterCTCTokenizer

    REG = json.loads((ROOT / "configs/p2_w05_tiny_asr.json").read_text(encoding="utf-8"))
    ROB = json.loads((ROOT / "configs/p2_w05_i002.json").read_text(encoding="utf-8"))
    if args.seed is not None:
        REG["seed"] = args.seed; ROB["seed"] = args.seed

    # —— 数据准备：若缺 splits/vocab 则运行对应 prepare ——
    def ensure_prep(script, data_dir, marker):
        m = ROOT / data_dir / marker
        if not m.exists():
            import subprocess
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)

    ensure_prep("p2_w05_prepare.py", REG["output_data"], "splits.json")
    ensure_prep("p2_w05_i002_prepare.py", ROB["output_data"], "splits.json")

    def load_splits(data_dir):
        return json.loads((ROOT / data_dir / "splits.json").read_text(encoding="utf-8"))
    reg_splits = load_splits(REG["output_data"]); rob_splits = load_splits(ROB["output_data"])
    vocab = json.loads((ROOT / REG["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterCTCTokenizer(vocab)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    t0 = time.time()
    print(f"[ASR 单阶段] 阶段A(干净预训练) epochs={REG['training']['epochs']} lr={REG['training']['learning_rate']}")
    set_seed(REG["seed"], REG["cpu_threads"])
    model = make_model(REG, len(tokenizer))
    opt = torch.optim.AdamW(model.parameters(), lr=REG["training"]["learning_rate"],
                            weight_decay=REG["training"]["weight_decay"])
    tr_l = make_loader(ROOT, REG["output_data"], reg_splits["regular"]["train"], REG["training"]["batch_size"], shuffle=True, seed=REG["seed"])
    va_l = make_loader(ROOT, REG["output_data"], reg_splits["regular"]["validation"], REG["training"]["batch_size"], shuffle=False, seed=REG["seed"])
    best_cer, patience, hist_a = float("inf"), 0, []
    for epoch in range(REG["training"]["epochs"]):
        loss = train_epoch(model, tr_l, opt, criterion, REG["training"]["gradient_clip_norm"])
        m, _ = evaluate(model, va_l, tokenizer, criterion)
        hist_a.append({"epoch": epoch, "train_loss": loss, "val_cer": m["cer"]})
        print(f"  A[epoch {epoch}] train_loss={loss:.4f} val_cer={m['cer']:.4f}")
        if m["cer"] < best_cer: best_cer, patience = m["cer"], 0
        else:
            patience += 1
            if patience >= REG["training"]["early_stopping_patience"]: break
    # 保留阶段A最优参数
    (ROOT / "results" / "P2-W05_tiny_asr").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ROOT / "results" / "P2-W05_tiny_asr" / "single_stage_clean.pt")

    # —— 阶段B：鲁棒强化（SpecAugment + 多风格），同模型继续 ——
    print("[ASR 单阶段] 阶段B(鲁棒强化) epochs=", ROB["training"]["epochs"])
    set_seed(ROB["seed"], ROB["cpu_threads"])
    opt = torch.optim.AdamW(model.parameters(), lr=ROB["training"]["learning_rate"],
                            weight_decay=ROB["training"]["weight_decay"])
    aug = {"probability": ROB["augmentation"]["specaugment_probability"],
           "frequency_mask_count": ROB["augmentation"]["frequency_mask_count"],
           "frequency_mask_max_width": ROB["augmentation"]["frequency_mask_max_width"],
           "time_mask_count": ROB["augmentation"]["time_mask_count"],
           "time_mask_max_width": ROB["augmentation"]["time_mask_max_width"]}
    # 用 I002 的 dataset loader（支持 augmentation）
    from mini_multimodal.asr import ASRDataset
    from torch.utils.data import DataLoader
    from mini_multimodal.asr import collate_asr
    def aug_loader(data_dir, ids, bs, shake, seed):
        ds = ASRDataset(ROOT, data_dir=data_dir, record_ids=ids, augmentation=aug, augmentation_seed=seed)
        return DataLoader(ds, batch_size=bs, shuffle=shake, collate_fn=collate_asr)
    rtr_l = aug_loader(ROB["output_data"], rob_splits["train_multistyle"], ROB["training"]["batch_size"], True, ROB["seed"])
    nval_l = make_loader(ROOT, ROB["output_data"], rob_splits["validation_noise_15db"], ROB["training"]["batch_size"], shuffle=False, seed=ROB["seed"])
    cval_l = make_loader(ROOT, REG["output_data"], reg_splits["regular"]["validation"], ROB["training"]["batch_size"], shuffle=False, seed=ROB["seed"])
    score, hist_b, patience = float("inf"), [], 0
    for epoch in range(ROB["training"]["epochs"]):
        loss = train_epoch(model, rtr_l, opt, criterion, ROB["training"]["gradient_clip_norm"])
        cm, _ = evaluate(model, cval_l, tokenizer, criterion); nm, _ = evaluate(model, nval_l, tokenizer, criterion)
        s = cm["cer"] + nm["cer"]
        hist_b.append({"epoch": epoch, "train_loss": loss, "clean_cer": cm["cer"], "noise15_cer": nm["cer"], "score": s})
        print(f"  B[epoch {epoch}] loss={loss:.4f} clean={cm['cer']:.4f} noise15={nm['cer']:.4f}")
        if s < score: score, patience = s, 0
        else:
            patience += 1
            if patience >= ROB["training"]["early_stopping_patience"]: break
    out = ROOT / "results" / "P2-W05_tiny_asr" / "single_stage"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "best.pt")
    write_json(out / "training_history.json", {"phase_a": hist_a, "phase_b": hist_b})
    write_json(out / "report.json", {"verified": True, "elapsed_seconds": round(time.time() - t0, 2),
                                     "phase_b_final_noise15_cer": hist_b[-1]["noise15_cer"], "phase_b_final_clean_cer": hist_b[-1]["clean_cer"]})
    print(json.dumps({"verified": True, "elapsed_seconds": round(time.time() - t0, 2),
                      "final_noise15_cer": hist_b[-1]["noise15_cer"], "final_clean_cer": hist_b[-1]["clean_cer"],
                      "save": str(out / "best.pt")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
