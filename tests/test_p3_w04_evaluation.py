from __future__ import annotations

import numpy as np

from mini_multimodal.video_gen.evaluation import background_flicker, movement_signature


def test_motion_signature_recovers_direction_and_exact_boundaries() -> None:
    trajectory = np.asarray([
        [0.2, 0.4], [0.2, 0.4], [0.2, 0.4], [0.3, 0.4],
        [0.4, 0.4], [0.5, 0.4], [0.5, 0.4], [0.5, 0.4],
    ])
    result = movement_signature(trajectory)
    assert result["direction"] == "right"
    assert result["start"] == 2
    assert result["end"] == 5


def test_background_flicker_excludes_correct_object_motion() -> None:
    frames = np.ones((8, 64, 64, 3), dtype=np.float32)
    for frame in range(8):
        frames[frame, 20:28, 4 + frame * 4:12 + frame * 4] = np.asarray([1.0, 0.0, 0.0])
    assert background_flicker(frames) == 0.0


def test_background_flicker_detects_background_change() -> None:
    frames = np.ones((8, 64, 64, 3), dtype=np.float32)
    frames[4:, :, :, :] = 0.8
    assert background_flicker(frames) > 0.0

