import torch

from mini_multimodal.video_gen.model import TinyVideoDenoiser


def test_multiscale_trajectory_control_preserves_video_shape():
    model = TinyVideoDenoiser(vocab_size=20, base_vocab_size=10, base_channels=16,
                              text_dim=32, time_dim=32, attention_heads=4,
                              temporal_cfg={"heads": 4, "text_dim": 32, "multi_scale_trajectory": True})
    video = torch.randn(1, 4, 8, 16, 16)
    ids = torch.tensor([[1, 11, 2]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    trajectory = torch.linspace(0.2, 0.8, 8).view(1, 8, 1).repeat(1, 1, 2)
    output = model(video, torch.tensor([5]), ids, mask, trajectory=trajectory)
    assert output.predicted_noise.shape == video.shape
    assert model.trajectory16.weight.requires_grad
    assert model.trajectory8.weight.requires_grad
