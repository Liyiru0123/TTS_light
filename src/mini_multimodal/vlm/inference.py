"""Small, documented inference helpers for the P2-W03 teaching model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from mini_multimodal.vae import TinyImageVAE

from .data import CharacterTokenizer
from .model import TinyVLM


@dataclass(slots=True)
class LoadedTinyVLM:
    model: TinyVLM
    tokenizer: CharacterTokenizer
    answer_classes: tuple[str, ...]
    metadata: dict[str, Any]


def load_tiny_vlm_inference_bundle(
    path: str | Path,
    *,
    device: str = "cpu",
) -> LoadedTinyVLM:
    """Load the self-describing, weights-only P2-W03 inference artifact."""

    if device != "cpu":
        raise ValueError("the validated P2-W03 teaching runtime is CPU-only")
    bundle = torch.load(Path(path), map_location=device, weights_only=True)
    if bundle.get("schema_version") != 1 or bundle.get("artifact_type") != "p2_w03_tiny_vlm_inference":
        raise ValueError("unsupported P2-W03 inference bundle")
    tokenizer = CharacterTokenizer(bundle["vocabulary"])
    special = bundle["special_tokens"]
    answers = tuple(str(value) for value in bundle["answer_classes"])
    model = TinyVLM(
        vocabulary_size=len(tokenizer),
        answer_class_count=len(answers),
        pad_id=int(special["<pad>"]),
        bos_id=int(special["<bos>"]),
        eos_id=int(special["<eos>"]),
        mask_id=int(special["<mask>"]),
        target_marker_id=int(special["甲"]),
        reference_marker_id=int(special["乙"]),
        **bundle["model_config"],
    ).to(device)
    model.load_state_dict(bundle["model_state"], strict=True)
    model.eval()
    metadata = {
        "validated_metrics": bundle["validated_metrics"],
        "source_training_checkpoint": bundle["source_training_checkpoint"],
    }
    return LoadedTinyVLM(model, tokenizer, answers, metadata)


def predict_from_latent(
    loaded: LoadedTinyVLM,
    latent: torch.Tensor | np.ndarray,
    question: str,
) -> dict[str, Any]:
    """Run closed-set classification and causal generation on one 4x16x16 latent."""

    visual = torch.as_tensor(latent, dtype=torch.float32)
    if tuple(visual.shape) != (4, 16, 16):
        raise ValueError("latent must have shape [4,16,16]")
    question_ids = torch.tensor(
        [loaded.tokenizer.encode_question(question)], dtype=torch.long
    )
    visual = visual.unsqueeze(0)
    with torch.inference_mode():
        output = loaded.model(visual, question_ids)
        class_index = int(output.classification_logits.argmax(dim=-1).item())
        generated, probabilities = loaded.model.generate(visual, question_ids)
    return {
        "question": question,
        "classification_answer": loaded.answer_classes[class_index],
        "generated_answer": loaded.tokenizer.decode_answer(generated[0].tolist()),
        "generated_token_ids": generated[0].tolist(),
        "generated_token_probabilities": probabilities[0].tolist(),
        "target_coordinates": output.target_coordinates[0].tolist(),
        "reference_coordinates": output.reference_coordinates[0].tolist(),
    }


def encode_raw_image_with_vae(
    image_path: str | Path,
    *,
    vae_checkpoint: str | Path,
    vae_config: str | Path,
) -> torch.Tensor:
    """Encode one RGB 64x64 PNG as the posterior mean expected by TinyVLM."""

    config = json.loads(Path(vae_config).read_text(encoding="utf-8"))
    model_config = {
        key: config["model"][key]
        for key in ("base_channels", "latent_channels", "logvar_min", "logvar_max")
    }
    vae = TinyImageVAE(**model_config)
    state = torch.load(Path(vae_checkpoint), map_location="cpu", weights_only=False)
    vae.load_state_dict(state["model_state"], strict=True)
    vae.eval()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        if image.size != (64, 64):
            raise ValueError("raw image must be exactly 64x64 pixels")
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    image_tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        mean, _ = vae.encode(image_tensor)
    return mean[0]


def first_cached_latent(
    cache_path: str | Path,
    *,
    scene_id: str,
) -> torch.Tensor:
    """Read one named scene from the fixed posterior-mean cache."""

    with np.load(cache_path, allow_pickle=False) as cache:
        scene_ids: Sequence[str] = [str(value) for value in cache["scene_ids"].tolist()]
        try:
            index = scene_ids.index(scene_id)
        except ValueError as error:
            raise KeyError(f"scene not found in latent cache: {scene_id}") from error
        return torch.from_numpy(cache["posterior_mean"][index].astype(np.float32))
