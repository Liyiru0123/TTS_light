"""Tiny visual-language model data, modules, and experiment interfaces."""

from .data import (
    CharacterTokenizer,
    VLMQADataset,
    build_p2_w03_data_spec,
    collate_vlm_qa,
    export_p2_w03_data,
    verify_p2_w03_data,
)
from .baselines import ImageOnlyVQABaseline, TextOnlyVQABaseline
from .model import TinyVLM, TinyVLMOutput
from .inference import (
    LoadedTinyVLM,
    encode_raw_image_with_vae,
    first_cached_latent,
    load_tiny_vlm_inference_bundle,
    predict_from_latent,
)
from .task import VLMTask

__all__ = [
    "CharacterTokenizer",
    "VLMQADataset",
    "build_p2_w03_data_spec",
    "collate_vlm_qa",
    "export_p2_w03_data",
    "verify_p2_w03_data",
    "ImageOnlyVQABaseline",
    "TextOnlyVQABaseline",
    "TinyVLM",
    "TinyVLMOutput",
    "LoadedTinyVLM",
    "encode_raw_image_with_vae",
    "first_cached_latent",
    "load_tiny_vlm_inference_bundle",
    "predict_from_latent",
    "VLMTask",
]
