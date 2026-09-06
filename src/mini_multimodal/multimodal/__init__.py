"""Unified four-modality data access, batching, caching, and metrics."""

from .artifacts import (
    build_integration_index,
    export_unified_artifacts,
    verify_unified_artifacts,
)
from .cache import ByteLRUCache, CacheInfo
from .collate import collate_multimodal
from .dataset import UnifiedMultimodalDataset
from .metrics import ExactMatchMetric, MaskedRegressionMetric, normalize_exact_match
from .routing import TaskRoute, route_question_kind, route_table
from .visualization import make_sample_overview, save_sample_overview

__all__ = [
    "ByteLRUCache",
    "CacheInfo",
    "ExactMatchMetric",
    "MaskedRegressionMetric",
    "UnifiedMultimodalDataset",
    "TaskRoute",
    "build_integration_index",
    "collate_multimodal",
    "export_unified_artifacts",
    "make_sample_overview",
    "normalize_exact_match",
    "route_question_kind",
    "route_table",
    "save_sample_overview",
    "verify_unified_artifacts",
]
