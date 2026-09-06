import torch

from mini_multimodal.tts.cfm import ode_infer, scale_durations
from mini_multimodal.tts.model import TinySpeechFlow


def _model():
    return TinySpeechFlow(vocab_size=10,
                          text_cfg={"d_text": 16, "num_layers": 1, "num_heads": 4, "ff_dim": 32, "dropout": 0.0, "max_tokens": 12},
                          duration_cfg={"d_text": 16, "hidden": 16},
                          condition_cfg={"d_text": 16, "latent_channels": 80},
                          unet_cfg={"in_channels": 80, "base_channels": 16, "down_channels": 24, "time_embed_dim": 16})


def test_duration_scaling_preserves_zeros_and_exact_total():
    source = torch.tensor([0, 2, 3, 0, 5])
    fast = scale_durations(source, 0.7)
    slow = scale_durations(source, 1.3)
    assert fast.tolist()[0::3] == [0, 0]
    assert slow.tolist()[0::3] == [0, 0]
    assert int(fast.sum()) == 7
    assert int(slow.sum()) == 13
    assert torch.all(fast[source > 0] >= 1)


def test_ode_padding_is_exactly_zero_and_seed_stable():
    model = _model().eval()
    condition = torch.randn(80, 64)
    first = ode_infer(model, condition, 11, ode_steps=2, seed=4)
    second = ode_infer(model, condition, 11, ode_steps=2, seed=4)
    assert torch.equal(first, second)
    assert torch.count_nonzero(first[:, 11:]) == 0
