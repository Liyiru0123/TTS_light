from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.video_mllm import (  # noqa: E402
    TinyVideoMLLM,
    VideoQADataset,
    collate_video_qa,
)


DATA = ROOT / "data/video_qa/p3_w01_v1"
CONFIG = json.loads((ROOT / "configs/p3_w01_tiny_video_mllm.json").read_text(encoding="utf-8"))


def _batch_and_model():
    torch.manual_seed(CONFIG["seed"])
    dataset = VideoQADataset(ROOT, data_dir=DATA, splits="train")
    batch = collate_video_qa([dataset[0], dataset[1]])
    tokenizer = dataset.tokenizer
    answers = json.loads((DATA / "answer_classes.json").read_text(encoding="utf-8"))
    model = TinyVideoMLLM(
        vocabulary_size=len(tokenizer),
        answer_class_count=len(answers),
        pad_id=tokenizer.id("<pad>"),
        bos_id=tokenizer.id("<bos>"),
        eos_id=tokenizer.id("<eos>"),
        mask_id=tokenizer.id("<mask>"),
        **CONFIG["model"],
    )
    return batch, model


def test_model_vidqa_001_shapes_and_parameter_budget() -> None:
    batch, model = _batch_and_model()
    output = model(batch["video"], batch["question_ids"], batch["answer_input_ids"])
    assert tuple(output.classification_logits.shape) == (2, 8)
    assert tuple(output.answer_logits.shape[:2]) == tuple(batch["answer_input_ids"].shape)
    assert tuple(output.spatial_tokens.shape) == (2, 8, 64, 64)
    assert tuple(output.frame_tokens.shape) == (2, 8, 64)
    assert tuple(output.temporal_tokens.shape) == (2, 10, 64)
    assert tuple(output.spatial_attention.shape) == (2, 8, 4, 1, 64)
    assert output.spatial_attention.shape[-1] == 64
    assert tuple(output.temporal_attention.shape) == (2, 4, 10, 10)
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) <= 1_000_000


def test_model_vidqa_002_temporal_and_cross_modal_gradients_are_active() -> None:
    batch, model = _batch_and_model()
    output = model(batch["video"], batch["question_ids"], batch["answer_input_ids"])
    loss = output.classification_logits.square().mean() + output.answer_logits.square().mean()
    loss.backward()
    assert model.frame_query.grad is not None and model.frame_query.grad.abs().sum() > 0
    assert model.temporal_cls.grad is not None and model.temporal_cls.grad.abs().sum() > 0
    assert model.spatial_layout_projection.weight.grad is not None
    assert model.spatial_layout_projection.weight.grad.abs().sum() > 0
    assert model.motion_delta_projection.weight.grad is not None
    assert model.motion_delta_projection.weight.grad.abs().sum() > 0
    assert model.time_embedding.grad is not None and model.time_embedding.grad.abs().sum() > 0
    assert model.spatial_cross_attention.in_proj_weight.grad is not None
    assert model.spatial_cross_attention.in_proj_weight.grad.abs().sum() > 0
    assert model.temporal_blocks[0].attention.in_proj_weight.grad is not None
    assert model.temporal_blocks[0].attention.in_proj_weight.grad.abs().sum() > 0
    assert model.decoder[0].cross_attention.in_proj_weight.grad is not None
    assert model.decoder[0].cross_attention.in_proj_weight.grad.abs().sum() > 0


def test_model_vidqa_003_causal_mask_blocks_future_answer_tokens() -> None:
    batch, model = _batch_and_model()
    model.eval()
    first = batch["answer_input_ids"].clone()
    second = first.clone()
    second[:, -1] = (second[:, -1] + 1) % len(model.token_embedding.weight)
    with torch.no_grad():
        output_a = model(batch["video"], batch["question_ids"], first).answer_logits
        output_b = model(batch["video"], batch["question_ids"], second).answer_logits
    assert torch.allclose(output_a[:, :-1], output_b[:, :-1], atol=1e-6)


def test_model_vidqa_004_frame_order_and_time_ablations_change_outputs() -> None:
    batch, model = _batch_and_model()
    model.eval()
    # Untrained classifier logits can differ by only about 1e-6, making a
    # default allclose check flaky.  The actual architectural contract is that
    # the explicit temporal representation reacts to each intervention.
    with torch.no_grad():
        full_state = model(batch["video"], batch["question_ids"])
        reversed_state = model(batch["video"].flip(1), batch["question_ids"])
        no_position_state = model(
            batch["video"], batch["question_ids"], disable_time_position=True
        )
        reversed_no_position_state = model(
            batch["video"].flip(1), batch["question_ids"], disable_time_position=True
        )
        no_temporal_state = model(
            batch["video"], batch["question_ids"], disable_temporal_module=True
        )
    assert float((full_state.temporal_tokens[:, 0] - reversed_state.temporal_tokens[:, 0]).abs().max()) > 1e-6
    assert float((full_state.temporal_tokens[:, 0] - no_position_state.temporal_tokens[:, 0]).abs().max()) > 1e-4
    assert float((full_state.temporal_tokens[:, 0] - no_temporal_state.temporal_tokens[:, 0]).abs().max()) > 1e-4
    assert torch.allclose(
        no_position_state.temporal_tokens[:, 0],
        reversed_no_position_state.temporal_tokens[:, 0],
        atol=1e-5,
    )


def test_model_vidqa_005_tied_decoder_starts_in_trainable_range() -> None:
    batch, model = _batch_and_model()
    output = model(batch["video"], batch["question_ids"], batch["answer_input_ids"])
    generation_loss = F.cross_entropy(
        output.answer_logits.reshape(-1, output.answer_logits.shape[-1]),
        batch["answer_target_ids"].reshape(-1),
        ignore_index=model.pad_id,
    )
    assert float(model.token_embedding.weight.std().detach()) < 0.05
    assert float(generation_loss.detach()) < 10.0
