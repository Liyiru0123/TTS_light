import torch

from mini_multimodal.video_gen.anchored_refiner import AnchoredTemporalRefiner, perturb_independent_frames


def test_refiner_and_perturbation_contracts():
    model = AnchoredTemporalRefiner(channels=16, heads=4)
    video = torch.full((1, 3, 8, 64, 64), 0.5)
    perturbed = perturb_independent_frames(video, generator=torch.Generator().manual_seed(3))
    output = model(perturbed)
    assert output.shape == video.shape
    assert not torch.equal(perturbed, video)
    assert torch.all((output >= 0) & (output <= 1))
