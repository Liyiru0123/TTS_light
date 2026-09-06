"""Self-describing inference helpers for the P3-W01 video-language model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import CharacterTokenizer
from .model import TinyVideoMLLM


@dataclass(slots=True)
class LoadedTinyVideoMLLM:
    model: TinyVideoMLLM
    tokenizer: CharacterTokenizer
    answer_classes: tuple[str, ...]
    metadata: dict[str, Any]


def load_tiny_video_mllm(path: str | Path) -> LoadedTinyVideoMLLM:
    bundle = torch.load(Path(path), map_location="cpu", weights_only=True)
    if bundle.get("schema_version") != 1 or bundle.get("artifact_type") != "p3_w01_tiny_video_mllm_inference":
        raise ValueError("unsupported P3-W01 inference package")
    tokenizer = CharacterTokenizer(bundle["vocabulary"])
    special = bundle["special_tokens"]
    answer_classes = tuple(str(value) for value in bundle["answer_classes"])
    model = TinyVideoMLLM(
        vocabulary_size=len(tokenizer),
        answer_class_count=len(answer_classes),
        pad_id=special["<pad>"],
        bos_id=special["<bos>"],
        eos_id=special["<eos>"],
        mask_id=special["<mask>"],
        **bundle["model_config"],
    )
    model.load_state_dict(bundle["model_state"], strict=True)
    model.eval()
    return LoadedTinyVideoMLLM(
        model,
        tokenizer,
        answer_classes,
        {"validated_metrics": bundle["validated_metrics"], "source_checkpoint_sha256": bundle["source_checkpoint_sha256"]},
    )


def predict_video(loaded: LoadedTinyVideoMLLM, frames: np.ndarray | torch.Tensor, question: str) -> dict[str, Any]:
    video = torch.as_tensor(frames)
    if tuple(video.shape) == (8, 64, 64, 3):
        video = video.permute(0, 3, 1, 2)
    if tuple(video.shape) != (8, 3, 64, 64):
        raise ValueError("frames must have shape [8,64,64,3] or [8,3,64,64]")
    video = video.float()
    if float(video.max()) > 1.0:
        video = video / 255.0
    question_ids = torch.tensor([loaded.tokenizer.encode_question(question)], dtype=torch.long)
    with torch.inference_mode():
        output = loaded.model(video[None], question_ids)
        generated, probabilities = loaded.model.generate(video[None], question_ids)
    class_index = int(output.classification_logits.argmax(dim=-1).item())
    return {
        "question": question,
        "classification_answer": loaded.answer_classes[class_index],
        "classification_probabilities": output.classification_logits.softmax(dim=-1)[0].tolist(),
        "generated_answer": loaded.tokenizer.decode_answer(generated[0].tolist()),
        "generated_token_ids": generated[0].tolist(),
        "generated_token_probabilities": probabilities[0].tolist(),
        "spatial_attention_shape": list(output.spatial_attention.shape),
        "temporal_attention": output.temporal_attention[0].mean(dim=0).tolist(),
    }
