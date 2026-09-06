"""TinyTTS (TinySpeechFlow) package.

Text -> Duration/Alignment -> Frame-wise Fusion -> Conditional Flow -> Speech Latent -> Frozen Decoder.
"""

from mini_multimodal.tts.alignment import (
    latent_center,
    valid_latent_frames,
    token_latent_durations,
    alignment_projection,
)
from mini_multimodal.tts.model import (
    DurationOverflowError,
    TinyTextEncoder,
    DurationPredictor,
    LengthRegulator,
    FrameWiseCondition,
    TemporalUNet,
    TinySpeechFlow,
)

__all__ = [
    "DurationOverflowError",
    "latent_center",
    "valid_latent_frames",
    "token_latent_durations",
    "alignment_projection",
    "TinyTextEncoder",
    "DurationPredictor",
    "LengthRegulator",
    "FrameWiseCondition",
    "TemporalUNet",
    "TinySpeechFlow",
]
