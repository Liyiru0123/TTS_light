"""TinyT2V package (P3-W04): video generation with motion predictor + trajectory conditioning."""

from mini_multimodal.video_gen.data import VideoLatentDataset, collate_video
from mini_multimodal.video_gen.model import (
    MotionPredictor,
    SinusoidalCoordinateEmbedding,
    TemporalTransformer,
    TinyVideoDenoiser,
)
from mini_multimodal.video_gen.sampling import sample_video_latents

__all__ = [
    "TemporalTransformer",
    "TinyVideoDenoiser",
    "MotionPredictor",
    "SinusoidalCoordinateEmbedding",
    "VideoLatentDataset",
    "collate_video",
    "sample_video_latents",
]
