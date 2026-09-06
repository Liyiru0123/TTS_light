"""ASR transcription helper for TinyTTS closed-loop evaluation.

Reuses the frozen P2-W05 TinyASR and the project's 64-mel log_mel feature.
Pipeline: waveform -> 64-mel -> per-utterance per-channel mean/std -> CTC greedy -> text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mini_multimodal.asr import CharacterCTCTokenizer, TinyASR, ctc_greedy_decode
from mini_multimodal.speech.features import log_mel


def load_asr(package_path: Path) -> tuple[TinyASR, CharacterCTCTokenizer]:
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    tokenizer = CharacterCTCTokenizer(package["vocabulary"])
    model = TinyASR(vocabulary_size=len(tokenizer), **package["model_config"])
    model.load_state_dict(package["model_state"], strict=True)
    model.eval().requires_grad_(False)
    return model, tokenizer


def current_logmel(waveform: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
    return log_mel(waveform, sample_rate, n_fft=512, hop_length=256, n_mels=64, fmin=40.0, fmax=8000.0)


def transcribe(waveform: np.ndarray, model: TinyASR, tokenizer: CharacterCTCTokenizer) -> str:
    feature = current_logmel(waveform).T
    feature = (feature - feature.mean(axis=0, keepdims=True)) / np.maximum(feature.std(axis=0, keepdims=True), 1e-5)
    tensor = torch.from_numpy(feature.astype(np.float32))[None]
    with torch.inference_mode():
        output = model(tensor, torch.tensor([tensor.shape[1]]))
    return ctc_greedy_decode(output.logits, output.output_lengths, tokenizer)[0]
