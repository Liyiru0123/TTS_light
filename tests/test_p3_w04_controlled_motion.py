import torch

from mini_multimodal.video_gen.controlled_motion import (
    ControlledTemporalAttention,
    motion_controls,
    propagate_aligned_video,
    propagate_generated_residual,
)


def test_motion_controls_preserve_delayed_start_and_early_stop() -> None:
    trajectory = torch.tensor([[[0.2, 0.5], [0.2, 0.5], [0.2, 0.5], [0.3, 0.5],
                                [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5]]])
    velocity, moving = motion_controls(trajectory)
    assert velocity.shape == (1, 8, 2)
    assert moving[0, :, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_zero_initialized_temporal_attention_is_identity() -> None:
    module = ControlledTemporalAttention(channels=4, heads=1)
    video = torch.randn(2, 4, 8, 6, 5)
    trajectory = torch.rand(2, 8, 2)
    output = module(video, trajectory)
    assert output.shape == video.shape
    assert torch.equal(output, video)


def test_explicit_flow_moves_only_during_requested_interval() -> None:
    anchor = torch.zeros(1, 4, 5, 5)
    background = torch.zeros_like(anchor)
    anchor[:, :, 2, 1] = 1.0
    mask = torch.zeros(1, 1, 5, 5)
    mask[:, :, 2, 1] = 1.0
    trajectory = torch.tensor([[[0.25, 0.5], [0.25, 0.5], [0.25, 0.5], [0.5, 0.5],
                                [0.75, 0.5], [0.75, 0.5], [0.75, 0.5], [0.75, 0.5]]])
    video = propagate_generated_residual(anchor, background, mask, trajectory)
    centers = video[0, 0].reshape(8, -1).argmax(dim=1)
    assert centers.tolist() == [11, 11, 11, 12, 13, 13, 13, 13]


def test_aligned_video_can_carry_distinct_generated_frame_residuals() -> None:
    aligned = torch.zeros(1, 4, 2, 3, 3)
    aligned[:, :, 0, 1, 1] = 1.0
    aligned[:, :, 1, 1, 1] = 2.0
    background = torch.zeros(1, 4, 3, 3)
    mask = torch.zeros(1, 1, 3, 3)
    mask[:, :, 1, 1] = 1.0
    trajectory = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    output = propagate_aligned_video(aligned, background, mask, trajectory)
    assert output[0, 0, :, 1, 1].tolist() == [1.0, 2.0]
