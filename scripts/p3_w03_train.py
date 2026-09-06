"""P3-W03 I002 training: TinySpeechFlow (CFM + duration predictor).

I002 change (vs I001): duration augmentation for CFM conditioning.
- CFM loss: velocity MSE with (possibly rate-scaled) duration condition
  (I001 used ground-truth durations only -> rate scaling was out-of-distribution,
  causing content drop/dup at x0.7/x1.3 in the rate demo)
- Duration loss: MSE(log(d_pred+1), log(d_gt+1)) unchanged (on ground truth)

Augmentation: with prob (1 - clean_probability), sample factor f ~ U[fmin,fmax],
scale durations, and resample target x1 per char so (condition, target) stay
consistent. This makes rate control in-distribution at inference.

Data: DATA-SPG-001 units (data/speech/p3_w03_tiny_tts_v1/).
Frozen chain: P3-W02 VAE (posterior mean x1) + HiFi-GAN.
"""

from __future__ import annotations

import io
import json
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.tts import TinySpeechFlow  # noqa: E402
from mini_multimodal.tts.cfm import scale_durations, text_to_token_ids  # noqa: E402
from mini_multimodal.tts.frozen_chain import load_frozen_chain  # noqa: E402


def load_json(path: Path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class UnitCache:
    """Precompute x1 (posterior mean) + token ids for all units."""

    def __init__(self, cfg: dict, root: Path, chain, vocab: dict):
        self.chain = chain
        self.vocab = vocab
        manifest = [json.loads(l) for l in io.open(root / cfg["output_data"] / "unit_manifest.jsonl", encoding="utf-8")]
        self.units = [u for u in manifest if u["split"] == "train"]
        self.x1s = {}
        for u in self.units:
            mel80 = np.load(root / u["source_logmel80_path"], allow_pickle=False)
            mel = torch.tensor(mel80[:, u["mel_start"] : u["mel_end"]], dtype=torch.float32)
            x1 = chain.encode_unit(mel).reshape(80, 64).clone()
            x1[:, u["t_valid"]:] = 0.0
            self.x1s[u["unit_id"]] = x1

    def __len__(self):
        return len(self.units)

    def __getitem__(self, idx: int) -> dict:
        u = self.units[idx]
        return {
            "token_ids": torch.tensor(text_to_token_ids(u["token_chars"], self.vocab), dtype=torch.long),
            "durations": torch.tensor(u["token_durations"], dtype=torch.long),
            "x1": self.x1s[u["unit_id"]],
            "t_valid": u["t_valid"],
            "unit_id": u["unit_id"],
        }


def resample_x1(x1: torch.Tensor, d, d_aug, t_valid: int) -> torch.Tensor:
    """Resample latent frames per char so target content positions match d_aug.

    x1: [80, 64] posterior mean (valid [0, t_valid)). d / d_aug: [N] latent
    durations (original / augmented). Zero-duration tokens contribute no frames.
    """
    out = torch.zeros_like(x1)
    s, s_aug = 0, 0
    for k in range(len(d)):
        dk, dk_a = int(d[k]), int(d_aug[k])
        if dk == 0 or dk_a == 0:
            continue
        seg = x1[:, s : s + dk]
        if dk_a == dk:
            seg_a = seg
        elif dk_a > dk:
            reps = (dk_a + dk - 1) // dk
            seg_a = seg.repeat(1, reps)[:, :dk_a]
        else:
            idx = np.linspace(0, dk - 1, dk_a).astype(int)
            seg_a = seg[:, idx]
        out[:, s_aug : s_aug + dk_a] = seg_a
        s += dk
        s_aug += dk_a
    return out


def collate(items: list[dict]) -> dict:
    max_n = max(len(it["token_ids"]) for it in items)
    B = len(items)
    tok = torch.zeros(B, max_n, dtype=torch.long)
    dur = torch.zeros(B, max_n, dtype=torch.long)
    tok_mask = torch.zeros(B, max_n)
    x1 = torch.zeros(B, 80, 64)
    mask = torch.zeros(B, 64)
    for i, it in enumerate(items):
        n = len(it["token_ids"])
        tok[i, :n] = it["token_ids"]
        dur[i, :n] = it["durations"]
        tok_mask[i, :n] = 1.0
        x1[i] = it["x1"]
        mask[i, : it["t_valid"]] = 1.0
    return {"token_ids": tok, "durations": dur, "tok_mask": tok_mask, "x1": x1, "mask": mask,
            "t_valid": torch.tensor([it["t_valid"] for it in items]), "unit_ids": [it["unit_id"] for it in items]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p3_w03_tiny_tts.json")
    args = parser.parse_args()
    cfg = load_json(ROOT / args.config)
    out_dir = ROOT / cfg["output_results"]
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]) % (2**32))
    t0 = time.time()

    chain = load_frozen_chain(cfg, ROOT)
    with io.open(ROOT / cfg["output_data"] / "vocabulary.json", encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]
    cache = UnitCache(cfg, ROOT, chain, vocab)
    print(f"train units: {len(cache)}")

    model = TinySpeechFlow(
        vocab_size=len(vocab),
        text_cfg=cfg["text_encoder"],
        duration_cfg=cfg["duration"],
        condition_cfg=cfg["condition"],
        unet_cfg=cfg["unet"],
        t_max=int(cfg["container"]["t_max"]),
    )
    if cfg.get("parent_checkpoint"):
        parent = torch.load(ROOT / cfg["parent_checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(parent["model_state"], strict=True)
    n_params = model.count_parameters(trainable=True)
    print(f"trainable params: {n_params}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    batch_size = int(cfg["training"]["batch_size"])
    epochs = int(cfg["training"]["epochs"])
    n_batches = max(1, len(cache) // batch_size)
    model.train()

    history = []
    best_loss = float("inf")
    for epoch in range(epochs):
        perm = torch.randperm(len(cache))
        epoch_losses = []
        for b in range(n_batches):
            idxs = perm[b * batch_size : (b + 1) * batch_size]
            items = [cache[i] for i in idxs]
            batch = collate(items)
            t = (torch.rand(1) * 0.999 + 0.001).item()
            # Encode each unpadded text independently. Valid tokens must never
            # attend to batch padding in the bidirectional text Transformer.
            h_list = []
            for i in range(batch_size):
                n_i = int(batch["tok_mask"][i].sum().item())
                h_list.append(model.encode_text(batch["token_ids"][i, :n_i]))
            aug_cfg = cfg.get("duration_augmentation", {})
            use_aug = bool(aug_cfg.get("enabled", False))
            clean_p = float(aug_cfg.get("clean_probability", 0.5))
            fmin, fmax = float(aug_cfg.get("factor_min", 0.7)), float(aug_cfg.get("factor_max", 1.3))
            x1_target = batch["x1"].clone()
            mask_target = batch["mask"].clone()
            C_list = []
            for i in range(batch_size):
                n_i = int(batch["tok_mask"][i].sum().item())
                h_i = h_list[i]
                d_i = batch["durations"][i, :n_i]
                if use_aug and np.random.rand() >= clean_p:
                    f = float(np.random.uniform(fmin, fmax))
                    try:
                        d_aug = scale_durations(d_i, f, t_max=64)
                    except ValueError:
                        d_aug = d_i.clone()
                    T_aug = int(d_aug.sum().item())
                    x1_target[i] = resample_x1(batch["x1"][i], d_i, d_aug, int(batch["t_valid"][i].item()))
                    m = torch.zeros(64)
                    m[:T_aug] = 1.0
                    mask_target[i] = m
                    H_i = model.regulator(h_i, d_aug)
                else:
                    H_i = model.regulator(h_i, d_i)
                C_list.append(model.condition_frames(H_i))
            C = torch.stack(C_list)  # [B, 80, 64]
            x0 = torch.randn_like(x1_target) * mask_target[:, None, :]
            x_t = (1 - t) * x0 + t * x1_target
            v_pred = model.velocity(x_t, torch.tensor([t]), C, mask_target)
            v_target = x1_target - x0
            valid = (mask_target * 80).sum(dim=1, keepdim=True)  # [B,1] elements per sample
            cfm_loss = (((v_pred - v_target) ** 2) * mask_target[:, None, :]).sum() / valid.sum()

            # ---- duration loss (reuse h_all) ----
            duration_losses = []
            for i, h_i in enumerate(h_list):
                n_i = h_i.shape[0]
                predicted_log = model.duration(h_i).squeeze(-1)
                target_log = torch.log1p(batch["durations"][i, :n_i].float())
                duration_losses.append(((predicted_log - target_log) ** 2).mean())
            dur_loss = torch.stack(duration_losses).mean()

            loss = float(cfg["loss"]["cfm_weight"]) * cfm_loss + float(cfg["loss"]["duration_weight"]) * dur_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
            optimizer.step()
            epoch_losses.append(loss.item())
        mean_epoch = float(np.mean(epoch_losses))
        history.append({"epoch": epoch + 1, "loss": round(mean_epoch, 5),
                        "cfm_loss": round(float(cfm_loss), 5), "dur_loss": round(float(dur_loss), 5)})
        if mean_epoch < best_loss:
            best_loss = mean_epoch
            torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                        "config": cfg, "loss": best_loss}, out_dir / "checkpoints" / "best.pt")
        print(f"epoch {epoch + 1}/{epochs}: loss={mean_epoch:.5f} cfm={cfm_loss:.5f} dur={dur_loss:.5f} ({time.time()-t0:.0f}s)")

    torch.save({"epoch": epochs, "model_state": model.state_dict(), "config": cfg,
                "loss": float(np.mean(epoch_losses))}, out_dir / "checkpoints" / "last.pt")
    dump_json(out_dir / "metrics" / "training_history.json", {"history": history, "n_params": n_params,
                                                  "train_units": len(cache), "best_loss": best_loss,
                                                  "duration_augmentation": cfg.get("duration_augmentation", {})})
    print(f"training done in {time.time()-t0:.0f}s; checkpoints -> {out_dir / 'checkpoints'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
