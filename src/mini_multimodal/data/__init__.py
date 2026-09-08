"""Deterministic dataset configurations and split diagnostics."""

from .diagnostics import (
    compositional_report,
    dataset_statistics,
    leakage_report,
    shortcut_diagnostics,
    validate_dataset_bundle,
)
from .generation import build_dataset_bundle
from .schema import (
    DataSplit,
    DatasetBuildConfig,
    DatasetBundle,
    DatasetRecord,
    DatasetVariant,
)

__all__ = [
    "DataSplit",
    "DatasetBuildConfig",
    "DatasetBundle",
    "DatasetRecord",
    "DatasetVariant",
    "build_dataset_bundle",
    "compositional_report",
    "dataset_statistics",
    "leakage_report",
    "shortcut_diagnostics",
    "validate_dataset_bundle",
]
