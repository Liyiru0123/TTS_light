"""Run one read-only P3-W05 model probe and emit a machine-readable result."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUTPUT = ROOT / "results/P3-W05_model_integration/runtime_outputs"


def emit(payload: dict) -> None:
    print("P3W05_RESULT=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))


def probe_vlm() -> dict:
    from mini_multimodal.vlm import (
        first_cached_latent,
        load_tiny_vlm_inference_bundle,
        predict_from_latent,
    )

    loaded = load_tiny_vlm_inference_bundle(
        ROOT / "results/P2-W03_tiny_vlm/checkpoints/tiny_vlm_inference.pt"
    )
    latent = first_cached_latent(
        ROOT / "data/latents/p2_w02_v1/posterior_means.npz",
        scene_id="core_scene_000700",
    )
    result = predict_from_latent(loaded, latent, "图中有几个可见对象？")
    return {
        "input": {"scene_id": "core_scene_000700", "question": "图中有几个可见对象？"},
        "expected": "1个",
        "actual": {
            "classification_answer": result["classification_answer"],
            "generation_answer": result["generated_answer"],
        },
        "strict_load": True,
        "success": result["classification_answer"] == "1个" and result["generated_answer"] == "1个",
    }


def probe_image_ldm() -> dict:
    from mini_multimodal.ldm.geometry_evaluation import load_attempt3_checkpoint
    from mini_multimodal.ldm.revision_training import load_config
    from p3_w04_pf011_generated_anchor_motion import generated_anchor_components, select_cases

    config = load_config(ROOT)
    loaded = load_attempt3_checkpoint(ROOT, config, branch="control")
    scaling = float(
        json.loads((ROOT / config["output_data"] / "manifest.json").read_text(encoding="utf-8"))[
            "latent_scaling_factor"
        ]
    )
    rows = [
        json.loads(line)
        for line in (ROOT / "data/video/p3_w04_v1/records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    row = select_cases(rows, 6, split="test")[0]
    with torch.inference_mode():
        _, _, _, anchor_rgb = generated_anchor_components(loaded, config, row, scaling, 20265000)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image_path = OUTPUT / "tiny_image_ldm_anchor.png"
    display_rgb = np.clip(anchor_rgb * 255.0, 0, 255).round().astype(np.uint8)
    Image.fromarray(display_rgb).save(image_path)
    return {
        "input": {"record_id": row["record_id"], "prompt": row["prompt"], "seed": 20265000},
        "actual": {"image": image_path.relative_to(ROOT).as_posix(), "shape": list(anchor_rgb.shape)},
        "checkpoint": loaded["checkpoint"].relative_to(ROOT).as_posix(),
        "strict_load": True,
        "success": tuple(anchor_rgb.shape) == (64, 64, 3),
    }


def probe_asr() -> dict:
    from mini_multimodal.asr import (
        ASRDataset,
        CharacterCTCTokenizer,
        TinyASR,
        collate_asr,
        ctc_greedy_decode,
    )

    path = ROOT / "results/P2-W05_tiny_asr/checkpoints/inference_package.pt"
    package = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = CharacterCTCTokenizer(package["vocabulary"])
    model = TinyASR(vocabulary_size=len(tokenizer), **package["model_config"])
    model.load_state_dict(package["model_state"], strict=True)
    model.eval()
    record_id = "composition_scene_000540:speech:f0"
    dataset = ASRDataset(ROOT, data_dir="data/asr/p2_w05_v1", record_ids=[record_id])
    batch = collate_asr([dataset[0]])
    with torch.inference_mode():
        output = model(batch["logmels"], batch["input_lengths"])
    prediction = ctc_greedy_decode(output.logits, output.output_lengths, tokenizer)[0]
    reference = batch["transcripts"][0]
    return {
        "input": {"record_id": record_id},
        "expected": reference,
        "actual": prediction,
        "strict_load": True,
        "success": prediction == reference,
    }


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    values = np.clip(waveform, -1.0, 1.0)
    pcm = (values * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def probe_tts() -> dict:
    from mini_multimodal.tts import TinySpeechFlow
    from mini_multimodal.tts.cfm import ode_infer, text_to_token_ids
    from mini_multimodal.tts.frozen_chain import load_frozen_chain

    cfg = json.loads((ROOT / "configs/p3_w03_i004_supported_rate.json").read_text(encoding="utf-8"))
    vocab = json.loads((ROOT / cfg["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))["vocab"]
    model = TinySpeechFlow(
        vocab_size=len(vocab),
        text_cfg=cfg["text_encoder"],
        duration_cfg=cfg["duration"],
        condition_cfg=cfg["condition"],
        unet_cfg=cfg["unet"],
        t_max=int(cfg["container"]["t_max"]),
    )
    checkpoint = torch.load(ROOT / cfg["output_results"] / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    chain = load_frozen_chain(cfg, ROOT)
    text = "红色三角形"
    ids = torch.tensor(text_to_token_ids(list(text), vocab), dtype=torch.long)
    torch.manual_seed(20260823)
    with torch.inference_mode():
        encoded = model.encode_text(ids)
        durations = model.duration.predict_durations(encoded).long()
        frames = model.regulator(encoded, durations)
        condition = model.condition_frames(frames)
        valid_latent_frames = int(durations.sum().item())
        latent = ode_infer(model, condition, valid_latent_frames, t_max=64, ode_steps=20)
        mel = chain.decode_latent(latent.reshape(4, 20, 64))
        valid_mel_frames = min(4 * valid_latent_frames, 256)
        waveform = chain.synthesize(mel[:, :valid_mel_frames])[0, 0].cpu().numpy()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    wave_path = OUTPUT / "tiny_speech_flow_i004.wav"
    _write_wav(wave_path, waveform, int(cfg["feature"]["sample_rate"]))
    return {
        "input": {"text": text, "rate": 1.0, "seed": 20260823},
        "actual": {
            "duration_frames": valid_latent_frames,
            "wave_samples": int(waveform.size),
            "waveform": wave_path.relative_to(ROOT).as_posix(),
        },
        "strict_load": True,
        "success": bool(0 < valid_latent_frames <= 64 and waveform.size > 0 and np.isfinite(waveform).all()),
    }


def probe_video_mllm() -> dict:
    from mini_multimodal.video_mllm import TinyVideoMLLMI003, VideoQADataset, collate_video_qa

    path = ROOT / (
        "results/P3-W01_tiny_video_mllm/"
        "checkpoints/inference_package.pt"
    )
    package = torch.load(path, map_location="cpu", weights_only=False)
    dataset = VideoQADataset(ROOT, data_dir="data/video_qa/p3_w01_v1", splits="test")
    tokenizer = dataset.tokenizer
    model = TinyVideoMLLMI003(
        vocabulary_size=len(tokenizer),
        answer_class_count=len(package["answer_classes"]),
        pad_id=tokenizer.id("<pad>"),
        bos_id=tokenizer.id("<bos>"),
        eos_id=tokenizer.id("<eos>"),
        mask_id=tokenizer.id("<mask>"),
        **package["model_config"],
    )
    model.load_state_dict(package["model_state"], strict=True)
    model.eval()
    batch = collate_video_qa([dataset[0]])
    with torch.inference_mode():
        output = model(batch["video"], batch["question_ids"], object_query_masks=batch["object_query_masks"])
        generated_ids, _ = model.generate(
            batch["video"], batch["question_ids"], object_query_masks=batch["object_query_masks"]
        )
    class_id = int(output.classification_logits.argmax(dim=-1)[0])
    classification = package["answer_classes"][class_id]
    generation = tokenizer.decode_answer(generated_ids[0].tolist())
    expected = batch["metadata"][0]["answer_zh"]
    return {
        "input": {
            "record_id": batch["metadata"][0]["record_id"],
            "question": batch["metadata"][0]["question_zh"],
        },
        "expected": expected,
        "actual": {"classification_answer": classification, "generation_answer": generation},
        "strict_load": True,
        "success": classification == expected and generation == expected,
    }


def probe_t2v() -> dict:
    output_dir = "results/P3-W05_model_integration/runtime_outputs/tiny_t2v"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/p3_w04_i008_infer_example.py"),
            "--output-dir",
            output_dir,
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    result = json.loads((ROOT / output_dir / "inference_result.json").read_text(encoding="utf-8"))
    motion = result["metrics"]["motion_on"]
    return {
        "input": {"record_id": result["record_id"], "prompt": result["prompt"]},
        "expected": result["expected_segment_zero_based"],
        "actual": {
            "gif": result["output_gif"],
            "direction_accuracy": motion["direction_accuracy"],
            "start_stop_accuracy": motion["timing_accuracy"],
            "trajectory_mae": motion["trajectory_mae"],
        },
        "subprocess_stdout_tail": (completed.stdout or "")[-1000:],
        "strict_load": True,
        "success": motion["direction_accuracy"] == 1.0 and motion["timing_accuracy"] == 1.0,
    }


PROBES = {
    "tiny_vlm": probe_vlm,
    "tiny_image_ldm": probe_image_ldm,
    "tiny_asr": probe_asr,
    "tiny_speech_flow": probe_tts,
    "tiny_video_mllm": probe_video_mllm,
    "tiny_t2v": probe_t2v,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(PROBES), required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = PROBES[args.model]()
    result.update({"schema_version": 1, "model_id": args.model, "python_executable": sys.executable})
    emit(result)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
