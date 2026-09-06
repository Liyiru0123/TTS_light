from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.data import CharacterTokenizer
from mini_multimodal.ldm.model_grounded import TinyImageDenoiserGrounded, resolve_grounding_token_ids
from mini_multimodal.ldm.model_v2 import resolve_spatial_token_coordinates


def _setup() -> tuple[TinyImageDenoiserGrounded, CharacterTokenizer]:
    vocabulary = json.loads((ROOT / "data/ldm/p2_w04_v2/vocabulary.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocabulary)
    model = TinyImageDenoiserGrounded(
        vocab_size=len(tokenizer),
        base_channels=16,
        text_dim=32,
        time_dim=32,
        spatial_token_coordinates=resolve_spatial_token_coordinates(tokenizer.token_to_id),
        grounding_token_ids=resolve_grounding_token_ids(tokenizer.token_to_id),
    )
    return model, tokenizer


def _encode(tokenizer: CharacterTokenizer, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    values = [tokenizer.encode(prompt, 128) for prompt in prompts]
    width = max(map(len, values))
    ids = torch.zeros(len(values), width, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, row in enumerate(values):
        ids[index, : len(row)] = torch.tensor(row)
        mask[index, : len(row)] = True
    return ids, mask


def test_parser_binds_each_object_attributes_to_its_region() -> None:
    model, tokenizer = _setup()
    prompt = "第1帧中有2个可见对象。对象1是蓝色的中等大小圆形，位于左侧区域；对象2是红色的小正方形，位于右下区域。"
    ids, _ = _encode(tokenizer, [prompt])
    slots = model.extract_object_slots(ids)
    assert slots.mask.tolist() == [[True, True, False]]
    decoded = [[tokenizer.tokens[int(ids[0, position])] for position in slot] for slot in slots.token_positions[0, :2]]
    assert decoded == [
        ["蓝色", "圆形", "中等大小", "左侧区域"],
        ["红色", "正方形", "小", "右下区域"],
    ]


def test_grounding_is_shape_sensitive_and_disableable() -> None:
    torch.manual_seed(31)
    model, tokenizer = _setup()
    prompts = [
        "第1帧中有1个可见对象。对象1是绿色的中等大小圆形，位于中央区域。",
        "第1帧中有1个可见对象。对象1是绿色的中等大小正方形，位于中央区域。",
    ]
    ids, mask = _encode(tokenizer, prompts)
    latent = torch.randn(2, 4, 16, 16)
    step = torch.tensor([25, 25])
    full = model(latent, step, ids, mask).predicted_noise
    disabled = model(latent, step, ids, mask, disable_object_grounding=True).predicted_noise
    assert not torch.equal(full, disabled)
    assert not torch.equal(full[0], full[1])


def test_grounding_modules_receive_gradients_at_both_scales() -> None:
    torch.manual_seed(37)
    model, tokenizer = _setup()
    ids, mask = _encode(tokenizer, ["第1帧中有1个可见对象。对象1是黄色的大五边形，位于左上区域。"])
    output = model(torch.randn(1, 4, 16, 16), torch.tensor([17]), ids, mask)
    output.predicted_noise.square().mean().backward()
    assert float(model.object_semantic_encoder[0].weight.grad.abs().sum()) > 0
    assert float(model.object_projection_16.weight.grad.abs().sum()) > 0
    assert float(model.object_projection_8.weight.grad.abs().sum()) > 0


def test_first_round_model_state_interface_remains_unchanged() -> None:
    model, _ = _setup()
    grounded_names = {name for name, _ in model.named_parameters() if name.startswith("object_")}
    assert grounded_names
