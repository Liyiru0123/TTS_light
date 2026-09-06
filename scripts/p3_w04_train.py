"""P3-W04 TinyT2V I005 training: motion predictor + trajectory-conditioned denoiser.

- MotionPredictor: text (pooled) -> per-frame trajectory [B,8,2], MSE vs GT
- Denoiser: warm-started from P2-W04, full fine-tune; noise MSE + GT-trajectory
  position conditioning
- total = noise_loss + position_loss_weight * trajectory_loss
- Checkpoints save both model and predictor

Run: .venv/Scripts/python.exe scripts/p3_w04_train.py   (~15-25 min on CPU)
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.data import CharacterTokenizer  # noqa: E402
from mini_multimodal.ldm.schedule import DiffusionSchedule  # noqa: E402
from mini_multimodal.video_gen.data import VideoLatentDataset, collate_video  # noqa: E402
from mini_multimodal.video_gen.model import MotionPredictor, TinyVideoDenoiser  # noqa: E402


def load_json(path: Path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> int:
    cfg = load_json(ROOT / "configs" / "p3_w04_tiny_t2v.json")
    out_dir = ROOT / cfg["output_results"]
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]) % (2**32))
    tr_cfg = cfg["training"]
    t0 = time.time()

    package = torch.load(ROOT / cfg["image_ldm_package"], map_location="cpu", weights_only=False)
    base_tokenizer = CharacterTokenizer(package["vocabulary"])
    with io.open(ROOT / cfg["output_data"] / "vocabulary.json", encoding="utf-8") as f:
        tokenizer = CharacterTokenizer(json.load(f))
    schedule = DiffusionSchedule(**package["schedule_config"])
    mc = package["model_config"]

    model = TinyVideoDenoiser(vocab_size=len(tokenizer), base_vocab_size=len(base_tokenizer),
                              **mc, temporal_cfg=cfg["temporal_module"])
    model.load_frozen_spatial(package["model_state"], freeze_spatial=False)
    predictor = MotionPredictor(text_dim=int(mc["text_dim"]))
    model.train()
    predictor.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) \
        + sum(p.numel() for p in predictor.parameters())
    print(f"trainable params: {n_trainable}")

    params = list(model.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=float(tr_cfg["learning_rate"]),
                                 weight_decay=float(tr_cfg["weight_decay"]))
    dataset = VideoLatentDataset(ROOT, cfg["output_data"], split="train")
    loader = DataLoader(dataset, batch_size=int(tr_cfg["batch_size"]), shuffle=True,
                        collate_fn=collate_video, num_workers=0)
    epochs = int(tr_cfg["epochs"])
    position_weight = float(cfg.get("position_loss_weight", 1.0))
    history = []
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_losses = []
        noise_losses = []
        traj_losses = []
        for batch in loader:
            t = (torch.rand(1) * 0.999 + 0.001).item()
            x0 = torch.randn_like(batch["latent"])
            timestep = torch.full((batch["latent"].shape[0],), t * 49, dtype=torch.long)
            x_t = schedule.q_sample(batch["latent"], timestep, x0)
            pred = model(x_t, timestep, batch["text_ids"], batch["text_mask"],
                         trajectory=batch["position_gt"])
            noise_loss = ((pred.predicted_noise - x0) ** 2).mean()
            traj_pred = predictor(model.pool_text(batch["text_ids"], batch["text_mask"]))
            traj_loss = ((traj_pred - batch["position_gt"]) ** 2).mean()
            loss = noise_loss + position_weight * traj_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(tr_cfg["gradient_clip_norm"]))
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            noise_losses.append(float(noise_loss.item()))
            traj_losses.append(float(traj_loss.item()))
        mean_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch + 1, "loss": round(mean_loss, 5),
                        "noise_loss": round(float(np.mean(noise_losses)), 5),
                        "trajectory_loss": round(float(np.mean(traj_losses)), 5)})
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                        "predictor_state": predictor.state_dict(), "config": cfg,
                        "loss": best_loss, "trainable_params": n_trainable}, ckpt_dir / "best.pt")
        print(f"epoch {epoch + 1}/{epochs}: loss={mean_loss:.5f} noise={noise_losses[-1]:.5f} "
              f"traj={traj_losses[-1]:.5f} ({time.time()-t0:.0f}s)")

    torch.save({"epoch": epochs, "model_state": model.state_dict(),
                "predictor_state": predictor.state_dict(), "config": cfg,
                "loss": float(np.mean(epoch_losses)), "trainable_params": n_trainable},
               ckpt_dir / "last.pt")
    dump_json(out_dir / "training_history.json", {
        "history": history, "n_trainable_params": n_trainable,
        "train_videos": len(dataset), "best_loss": best_loss,
        "wall_seconds": round(time.time() - t0, 1)})
    print(f"training done in {time.time()-t0:.0f}s; checkpoints -> {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
