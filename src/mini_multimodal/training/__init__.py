"""Reusable training infrastructure and diagnostic baselines."""

from .baseline import (
    ClassificationTask,
    ManifestImageDataset,
    TinyConvClassifier,
    cue_only_accuracy,
    learn_cue_mapping,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .core import StepOutput, Trainer, TrainerConfig
from .data import build_p2_w01_data_spec, export_p2_w01_data, verify_p2_w01_data
from .randomness import seeded_generator, set_deterministic_seed

__all__ = [
    "ClassificationTask",
    "ManifestImageDataset",
    "StepOutput",
    "TinyConvClassifier",
    "Trainer",
    "TrainerConfig",
    "build_p2_w01_data_spec",
    "cue_only_accuracy",
    "export_p2_w01_data",
    "learn_cue_mapping",
    "load_checkpoint",
    "save_checkpoint",
    "seeded_generator",
    "set_deterministic_seed",
    "verify_p2_w01_data",
]
