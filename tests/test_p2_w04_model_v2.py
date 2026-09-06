from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.data import CharacterTokenizer
from mini_multimodal.ldm.model_v2 import (
    TinyImageDenoiserV2,
    coordinate_grid,
    resolve_spatial_token_coordinates,
)


def _model() -> tuple[TinyImageDenoiserV2, CharacterTokenizer]:
    vocabulary = json.loads((ROOT / "data/ldm/p2_w04_v2/vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocabulary)
    model = TinyImageDenoiserV2(
        vocab_size=len(tokenizer),
        base_channels=16,
        text_dim=32,
        time_dim=32,
        spatial_token_coordinates=resolve_spatial_token_coordinates(tokenizer.token_to_id),
    )
    return model, tokenizer


def _batch(tokenizer: CharacterTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    values = [
        tokenizer.encode("第1帧中有1个可见对象。对象1是绿色的中等大小圆形，位于左侧区域。", 128),
        tokenizer.encode("第1帧中有1个可见对象。对象1是绿色的中等大小圆形，位于右侧区域。", 128),
    ]
    width = max(map(len, values))
    ids = torch.zeros(2, width, dtype=torch.long)
    mask = torch.zeros(2, width, dtype=torch.bool)
    for index, row in enumerate(values):
        ids[index, : len(row)] = torch.tensor(row)
        mask[index, : len(row)] = True
    return ids, mask


def test_coordinate_grid_has_expected_axes_and_range() -> None:
    grid = coordinate_grid(8, 16, device=torch.device("cpu"), dtype=torch.float32)
    assert grid.shape == (1, 2, 8, 16)
    assert torch.isclose(grid[0, 0, 0, 0], torch.tensor(-1.0))
    assert torch.isclose(grid[0, 0, 0, -1], torch.tensor(1.0))
    assert torch.isclose(grid[0, 1, 0, 0], torch.tensor(-1.0))
    assert torch.isclose(grid[0, 1, -1, 0], torch.tensor(1.0))


def test_multiscale_attention_shapes_sums_and_gradients() -> None:
    torch.manual_seed(7)
    model, tokenizer = _model()
    ids, mask = _batch(tokenizer)
    output = model(torch.randn(2, 4, 16, 16), torch.tensor([3, 17]), ids, mask)
    assert output.predicted_noise.shape == (2, 4, 16, 16)
    assert output.cross_attention_16.shape[:3] == (2, 4, 256)
    assert output.cross_attention_8.shape[:3] == (2, 4, 64)
    torch.testing.assert_close(output.cross_attention_16.sum(-1), torch.ones_like(output.cross_attention_16.sum(-1)), atol=1e-5, rtol=0)
    torch.testing.assert_close(output.cross_attention_8.sum(-1), torch.ones_like(output.cross_attention_8.sum(-1)), atol=1e-5, rtol=0)
    output.predicted_noise.square().mean().backward()
    assert float(model.cross_attention_16.attention.q_proj_weight.grad.abs().sum()) > 0
    assert float(model.cross_attention_8.attention.in_proj_weight.grad.abs().sum()) > 0
    assert float(model.input_coordinate.weight.grad.abs().sum()) > 0
    assert float(model.text_coordinate_projection.weight.grad.abs().sum()) > 0


def test_each_coordinate_and_attention_path_is_effective_and_disableable() -> None:
    torch.manual_seed(11)
    model, tokenizer = _model()
    model.eval()
    ids, mask = _batch(tokenizer)
    latent = torch.randn(2, 4, 16, 16)
    step = torch.tensor([9, 9])
    with torch.no_grad():
        full = model(latent, step, ids, mask).predicted_noise
        no_coordinates = model(latent, step, ids, mask, disable_coordinates=True).predicted_noise
        no_high = model(latent, step, ids, mask, disable_cross_attention_16=True).predicted_noise
        no_low = model(latent, step, ids, mask, disable_cross_attention_8=True).predicted_noise
        no_spatial_bias = model(latent, step, ids, mask, disable_spatial_attention_bias=True).predicted_noise
        no_spatial_map = model(latent, step, ids, mask, disable_text_spatial_map=True).predicted_noise
        no_text = model(latent, step, ids, mask, disable_cross_attention=True).predicted_noise
    for changed in (no_coordinates, no_high, no_low, no_spatial_bias, no_spatial_map, no_text):
        assert not torch.equal(full, changed)


def test_text_region_bias_localizes_the_region_token_attention() -> None:
    torch.manual_seed(19)
    model, tokenizer = _model()
    model.eval()
    ids, mask = _batch(tokenizer)
    with torch.no_grad():
        output = model(torch.randn(2, 4, 16, 16), torch.tensor([20, 20]), ids, mask)
    left_token = tokenizer.token_to_id["左侧区域"]
    left_index = int((ids[0] == left_token).nonzero(as_tuple=False)[0, 0])
    left_query = 8 * 16 + 2
    right_query = 8 * 16 + 13
    left_mass = float(output.cross_attention_16[0, :, left_query, left_index].mean())
    right_mass = float(output.cross_attention_16[0, :, right_query, left_index].mean())
    assert left_mass > right_mass


def test_region_words_have_distinct_fixed_coordinates() -> None:
    model, tokenizer = _model()
    mapping = resolve_spatial_token_coordinates(tokenizer.token_to_id)
    left = model.spatial_token_coordinates[tokenizer.token_to_id["左侧区域"]]
    right = model.spatial_token_coordinates[tokenizer.token_to_id["右侧区域"]]
    assert len(mapping) == 9
    assert float(left[0]) < 0 < float(right[0])
    assert float(left[1]) == float(right[1]) == 0.0


def test_text_spatial_map_moves_with_the_region_word() -> None:
    model, tokenizer = _model()
    ids, _ = _batch(tokenizer)
    fields = model.text_spatial_map(ids, 16, 16)
    left_peak = int(fields[0, 0].argmax()) % 16
    right_peak = int(fields[1, 0].argmax()) % 16
    assert left_peak < 8 < right_peak
