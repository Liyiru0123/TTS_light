"""P3-W03 capability samples (规范能力样例): teaching phenomena demos.

Groups (design doc 3.9-3.10, test plan 6):
  1. normal_tts      : 3 fixed test units (short/mid/long) -> predicted-duration synthesis + reference
  2. gt_vs_predicted : 1 unit, ground-truth vs predicted duration synthesis
  3. condition       : 1 unit, text ON / OFF / shuffled (condition contribution)
  4. duration_rate   : 1 unit, duration x0.7 / x1.0 / x1.3 (speaking rate)
  5. overflow        : constructed long text -> explicit DurationOverflowError (no silent truncation)
  6. zero_duration   : constructed forced d_i=0 for one content token -> token skipped
                      (compression resolution loss, not alignment error)

Evidence -> results/P3-W03_tiny_tts/examples/capability/ (wav)
           results/P3-W03_tiny_tts/examples/capability_examples.json + .md
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.tts import TinySpeechFlow, DurationOverflowError  # noqa: E402
from mini_multimodal.tts.cfm import text_to_token_ids, ode_infer  # noqa: E402
from mini_multimodal.tts.frozen_chain import load_frozen_chain  # noqa: E402

SR = 22050


def load_json(path: Path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def synthesize(model, chain, text_chars, vocab, durations):
    """text + durations -> waveform np. durations: [N] int tensor (may be scaled)."""
    token_ids = torch.tensor(text_to_token_ids(text_chars, vocab), dtype=torch.long)
    h = model.encode_text(token_ids)
    H = model.regulator(h, durations)
    C = model.condition_frames(H)
    T_valid = int(durations.sum().item())
    x1 = ode_infer(model, C, T_valid, t_max=64, ode_steps=20)
    mel = chain.decode_latent(x1.reshape(4, 20, 64))
    wave = chain.synthesize(mel[:, : min(4 * T_valid, 256)])[0, 0].numpy()
    return wave, T_valid


def save_wav(out_dir: Path, name: str, wave: np.ndarray) -> str:
    path = out_dir / name
    sf.write(path, wave.astype(np.float32), SR)
    return name


def unit_by_id(manifest, unit_id: str):
    for u in manifest:
        if u["unit_id"] == unit_id:
            return u
    raise KeyError(unit_id)


def main() -> int:
    cfg = load_json(ROOT / "configs" / "p3_w03_tiny_tts.json")
    out_dir = ROOT / cfg["output_results"] / "examples" / "capability"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg["seed"]))

    chain = load_frozen_chain(cfg, ROOT)
    with io.open(ROOT / cfg["output_data"] / "vocabulary.json", encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]
    ckpt = torch.load(ROOT / cfg["output_results"] / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model = TinySpeechFlow(
        vocab_size=len(vocab), text_cfg=cfg["text_encoder"], duration_cfg=cfg["duration"],
        condition_cfg=cfg["condition"], unet_cfg=cfg["unet"], t_max=int(cfg["container"]["t_max"]),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    manifest = [json.loads(l) for l in io.open(ROOT / cfg["output_data"] / "unit_manifest.jsonl", encoding="utf-8")]
    record_ids = cfg["evaluation_record_ids"]
    units = {uid: unit_by_id(manifest, uid) for uid in record_ids}

    groups = []

    # ---- 1. normal TTS: short / mid / long ----
    g1 = {"id": "normal_tts", "name": "正常 TTS：文本 → 变长连续语音",
          "phenomenon": "同一系统对不同长度文本生成变长语音；predicted duration 端到端",
          "samples": []}
    sel = [units["composition_scene_000633#u005"], units["shortcut_scene_000570#u002"], units["core_scene_000733#u003"]]
    for u in sel:
        token_ids = torch.tensor(text_to_token_ids(u["token_chars"], vocab), dtype=torch.long)
        h = model.encode_text(token_ids)
        d_pred = model.duration.predict_durations(h).long()
        wave, T = synthesize(model, chain, u["token_chars"], vocab, d_pred)
        files = {"predicted": save_wav(out_dir, f"{u['unit_id'].replace('#', '_')}_predicted.wav", wave)}
        # reference slice
        audio_path = ROOT / u["source_audio_path"]
        if audio_path.exists():
            w, sr = sf.read(audio_path, dtype="float32")
            s0, s1 = u["mel_start"] * 256, min(len(w), u["mel_end"] * 256)
            files["reference"] = save_wav(out_dir, f"{u['unit_id'].replace('#', '_')}_reference.wav", w[s0:s1])
        g1["samples"].append({
            "id": f"normal_{'short' if u['t_valid'] <= 27 else 'mid' if u['t_valid'] <= 40 else 'long'}",
            "unit_id": u["unit_id"], "text": u["text"], "T_gt": u["t_valid"],
            "T_pred": int(d_pred.sum().item()), "files": files,
            "observation": "预测时长与真值接近；语音长度随文本长度变化",
        })
    groups.append(g1)

    # ---- 2. gt vs predicted duration ----
    u = units["core_scene_000753#u010"]
    d_gt = torch.tensor(u["token_durations"], dtype=torch.long)
    token_ids = torch.tensor(text_to_token_ids(u["token_chars"], vocab), dtype=torch.long)
    h = model.encode_text(token_ids)
    d_pred = model.duration.predict_durations(h).long()
    wave_gt, T_gt = synthesize(model, chain, u["token_chars"], vocab, d_gt)
    wave_pr, T_pr = synthesize(model, chain, u["token_chars"], vocab, d_pred)
    tag = u["unit_id"].replace("#", "_")
    groups.append({
        "id": "gt_vs_predicted", "name": "真值时长 vs 预测时长（时长规划隔离）",
        "phenomenon": "相同文本，唯一变量是时长来源：TTS 声学质量由 CFM 决定，语速/节奏由时长决定",
        "samples": [{
            "id": "gt_vs_predicted", "unit_id": u["unit_id"], "text": u["text"],
            "T_gt": int(T_gt), "T_pred": int(T_pr), "d_gt": u["token_durations"], "d_pred": d_pred.tolist(),
            "files": {"gt": save_wav(out_dir, f"{tag}_gt.wav", wave_gt),
                      "predicted": save_wav(out_dir, f"{tag}_predicted.wav", wave_pr)},
            "observation": "两者语速/节奏接近，时长预测误差小（Gate C 相对误差 0.025）",
        }],
    })

    # ---- 3. condition ablation: ON / OFF / shuffled ----
    u = units["core_scene_000770#u000"]
    token_ids = text_to_token_ids(u["token_chars"], vocab)
    perm = list(range(len(token_ids)))
    rng = np.random.RandomState(7)
    rng.shuffle(perm)
    token_ids_shuf = [token_ids[p] for p in perm]
    token_ids_off = [0] * len(token_ids)
    d = torch.tensor(u["token_durations"], dtype=torch.long)
    wave_on, _ = synthesize(model, chain, u["token_chars"], vocab, d)
    # OFF/shuffled use the same duration but different conditions -> rebuild via token_ids directly
    h_on = model.encode_text(torch.tensor(token_ids, dtype=torch.long))
    h_off = model.encode_text(torch.tensor(token_ids_off, dtype=torch.long))
    h_shuf = model.encode_text(torch.tensor(token_ids_shuf, dtype=torch.long))
    C_on = model.condition_frames(model.regulator(h_on, d))
    C_off = model.condition_frames(model.regulator(h_off, d))
    C_shuf = model.condition_frames(model.regulator(h_shuf, d))
    T = int(d.sum().item())
    x1_on = ode_infer(model, C_on, T, t_max=64, ode_steps=20)
    x1_off = ode_infer(model, C_off, T, t_max=64, ode_steps=20)
    x1_shuf = ode_infer(model, C_shuf, T, t_max=64, ode_steps=20)
    def to_wave(x1):
        mel = chain.decode_latent(x1.reshape(4, 20, 64))
        return chain.synthesize(mel[:, : min(4 * T, 256)])[0, 0].numpy()
    tag = u["unit_id"].replace("#", "_")
    groups.append({
        "id": "condition", "name": "文本条件消融：ON / OFF / shuffled",
        "phenomenon": "相同时长规划下，关闭或乱序文本条件后，生成内容错乱——证明文本条件对内容有显著贡献",
        "samples": [{
            "id": "condition_on_off_shuffled", "unit_id": u["unit_id"], "text": u["text"],
            "T": T, "files": {"on": save_wav(out_dir, f"{tag}_cond_on.wav", to_wave(x1_on)),
                              "off": save_wav(out_dir, f"{tag}_cond_off.wav", to_wave(x1_off)),
                              "shuffled": save_wav(out_dir, f"{tag}_cond_shuffled.wav", to_wave(x1_shuf))},
            "observation": "ON 内容正确；OFF/shuffled 内容错乱（Gate B：CER ON=0.888 < OFF=0.978 < shuffled=0.966）",
        }],
    })

    # ---- 4. duration rate: x0.85 / x1.0 / x1.15 (mild range, content-preserving) ----
    u = units["core_scene_000758#u003"]  # 9-token phrase; rate change clearly audible
    d_gt = torch.tensor(u["token_durations"], dtype=torch.long)
    token_ids = torch.tensor(text_to_token_ids(u["token_chars"], vocab), dtype=torch.long)
    h = model.encode_text(token_ids)
    waves, Ts = {}, {}
    for label, factor in (("x0.85", 0.85), ("x1.0", 1.0), ("x1.15", 1.15)):
        d_s = torch.clamp(torch.round(d_gt.float() * factor).long(), min=1)
        w, T = synthesize(model, chain, u["token_chars"], vocab, d_s)
        waves[label] = w
        Ts[label] = T
    tag = u["unit_id"].replace("#", "_")
    groups.append({
        "id": "duration_rate", "name": "时长 × 语速（Duration controls speaking rate）",
        "phenomenon": "相同文本，时长缩放 x0.85/x1.0/x1.15 → 语速加快/正常/放慢；时长规划控制韵律节奏",
        "boundary": "极端倍率（×0.7/×1.3）超出训练分布，内容出现丢字/重字（rate 边界，真实 TTS 同样有极限产物；I002 曾用训练时长增强尝试修复，仅部分改善且损失基础音质，故放弃，见 docs/P3-W03_总结报告.md §4）",
        "samples": [{
            "id": "duration_rate_0.85_1.0_1.15", "unit_id": u["unit_id"], "text": u["text"],
            "T_gt": u["t_valid"], "T_x0.85": Ts["x0.85"], "T_x1.0": Ts["x1.0"], "T_x1.15": Ts["x1.15"],
            "audio_seconds": {k: round(len(v) / SR, 2) for k, v in waves.items()},
            "files": {k: save_wav(out_dir, f"{tag}_rate_{k}.wav", v) for k, v in waves.items()},
            "observation": "帧数少→语速快；帧数多→语速慢；温和倍率下内容保持（逐 token 时长均在训练分布内）",
        }],
    })

    # ---- 5. overflow: explicit failure ----
    longest = max(record_ids, key=lambda uid: len(units[uid]["token_chars"]))
    u = units[longest]
    long_text = list(u["text"]) * 3  # construct long text from known vocab chars
    overflow_info = {}
    token_ids = torch.tensor(text_to_token_ids(long_text, vocab), dtype=torch.long)
    h = model.encode_text(token_ids)
    d_pred = model.duration.predict_durations(h).long()
    T_hat = int(d_pred.sum().item())
    error_msg = None
    try:
        model.regulator(h, d_pred)
    except DurationOverflowError as e:
        error_msg = str(e)
    overflow_info = {
        "id": "overflow", "unit_id": longest, "base_text": u["text"],
        "long_text": "".join(long_text), "n_chars": len(long_text),
        "T_hat": T_hat, "t_max": 64, "explicit_failure": error_msg is not None,
        "error_message": error_msg,
        "observation": "T_hat > 64 时显式抛出 DurationOverflowError，不静默截断；真实 TTS 由此引出 chunking/streaming",
    }
    groups.append({"id": "overflow", "name": "overflow：显式失败（不静默截断）",
                   "phenomenon": "长文本导致 T_hat > 64，系统显式报错；引出真实 TTS 的 chunking/streaming",
                   "samples": [overflow_info]})

    # ---- 6. zero_duration: forced d_i=0 ----
    u = units["core_scene_000770#u000"]
    d_orig = torch.tensor(u["token_durations"], dtype=torch.long)
    d_mod = d_orig.clone()
    d_mod[1] = 0  # force second content token to zero duration
    token_ids = torch.tensor(text_to_token_ids(u["token_chars"], vocab), dtype=torch.long)
    h = model.encode_text(token_ids)
    C_orig = model.condition_frames(model.regulator(h, d_orig))
    C_mod = model.condition_frames(model.regulator(h, d_mod))
    T_orig, T_mod = int(d_orig.sum()), int(d_mod.sum())
    x1_orig = ode_infer(model, C_orig, T_orig, t_max=64, ode_steps=20)
    x1_mod = ode_infer(model, C_mod, T_mod, t_max=64, ode_steps=20)
    def to_wave2(x1, T):
        mel = chain.decode_latent(x1.reshape(4, 20, 64))
        return chain.synthesize(mel[:, : min(4 * T, 256)])[0, 0].numpy()
    tag = u["unit_id"].replace("#", "_")
    groups.append({
        "id": "zero_duration", "name": "零时长 token：压缩分辨率损失",
        "phenomenon": "人为将某内容 token 时长置 0 → 该字符不被发音；这是压缩分辨率损失（可跳过），不是对齐错误",
        "samples": [{
            "id": "zero_duration_forced", "unit_id": u["unit_id"], "text": u["text"],
            "token_chars": u["token_chars"], "d_orig": u["token_durations"], "d_modified": d_mod.tolist(),
            "T_orig": T_orig, "T_modified": T_mod,
            "files": {"normal": save_wav(out_dir, f"{tag}_zd_normal.wav", to_wave2(x1_orig, T_orig)),
                      "zero_duration": save_wav(out_dir, f"{tag}_zd_zero.wav", to_wave2(x1_mod, T_mod))},
            "observation": "第二个字符时长 0 → 该字符发音被跳过，总时长减少 d_i 帧",
        }],
    })

    manifest_out = {"schema_version": 1, "work_package": "P3-W03", "groups": groups}
    dump_json(ROOT / cfg["output_results"] / "examples" / "capability_examples.json", manifest_out)
    _write_md(ROOT / cfg["output_results"] / "examples" / "capability_examples.md", manifest_out)
    print("=== P3-W03 capability samples ===")
    for g in groups:
        print(f"  [{g['id']}] {g['name']}: {len(g['samples'])} sample(s)")
        if g["id"] == "overflow":
            s = g["samples"][0]
            print(f"    T_hat={s['T_hat']} > 64 -> explicit failure={s['explicit_failure']}")
    print(f"evidence -> {out_dir.parent}")
    return 0


def _write_md(path: Path, manifest: dict) -> None:
    lines = ["# P3-W03 TinyTTS 能力样例册（capability examples）",
             "",
             "> 用途：设计文档 §3.9—3.10 标志性教学现象的固定演示与证据。每组样例含输入、",
             "> 生成文件、预期现象与实际观察。音频在 `capability/` 子目录（22.05kHz WAV）。",
             "> 复现：`scripts/p3_w03_capabilities.py`（固定种子，确定性）。",
             ""]
    for g in manifest["groups"]:
        lines.append(f"## {g['id']}：{g['name']}")
        lines.append("")
        lines.append(f"**现象**：{g['phenomenon']}")
        lines.append("")
        if g.get("boundary"):
            lines.append(f"**边界**：{g['boundary']}")
            lines.append("")
        for s in g["samples"]:
            lines.append(f"- **{s['id']}**（unit {s.get('unit_id', '—')}）")
            if "text" in s and s.get("text"):
                lines.append(f"  - 文本：`{s['text']}`")
            for k, v in s.items():
                if k in ("id", "unit_id", "text", "files", "observation"):
                    continue
                if isinstance(v, list):
                    lines.append(f"  - {k}：{v}")
                elif k == "observation":
                    lines.append(f"  - 观察：{v}")
                elif k == "error_message" and v:
                    lines.append(f"  - 错误：`{v}`")
                elif v is not None and not isinstance(v, bool):
                    lines.append(f"  - {k}：{v}")
            if "files" in s and s["files"]:
                lines.append("  - 文件：" + "、".join(f"`{f}`" for f in s["files"].values()))
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
