from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.video_mllm import (  # noqa: E402
    TinyVideoMLLMI003,
    VideoQADataset,
    collate_video_qa,
)

CONFIG = json.loads((ROOT / "configs/p3_w01_i003.json").read_text(encoding="utf-8"))
DATA = ROOT / CONFIG["data"]["output_dir"]


def batch_and_model():
    torch.manual_seed(CONFIG["seed"])
    ids = json.loads(
        (ROOT / CONFIG["data"]["move_start_overfit_ids_path"]).read_text(encoding="utf-8")
    )
    dataset = VideoQADataset(ROOT, data_dir=DATA, record_ids=ids[:2])
    batch = collate_video_qa([dataset[0], dataset[1]])
    tokenizer = dataset.tokenizer
    answers = json.loads((DATA / "answer_classes.json").read_text(encoding="utf-8"))
    model = TinyVideoMLLMI003(
        vocabulary_size=len(tokenizer),
        answer_class_count=len(answers),
        pad_id=tokenizer.id("<pad>"),
        bos_id=tokenizer.id("<bos>"),
        eos_id=tokenizer.id("<eos>"),
        mask_id=tokenizer.id("<mask>"),
        **CONFIG["model"],
    )
    return batch, model


def test_unit_vidqa_i003_001_object_masks_and_targets_are_grounded():
    batch, _ = batch_and_model()
    assert tuple(batch["object_query_masks"].shape[:2]) == (2, 2)
    assert batch["object_query_masks"].sum(dim=-1).min() > 0
    assert not (batch["object_query_masks"][:, 0] & batch["object_query_masks"][:, 1]).any()
    assert batch["move_start_targets"].tolist() == [[0, 3], [3, 0]]


def test_unit_vidqa_i003_002_shapes_and_parameter_budget():
    batch, model = batch_and_model()
    output = model(
        batch["video"],
        batch["question_ids"],
        batch["answer_input_ids"],
        object_query_masks=batch["object_query_masks"],
    )
    assert tuple(output.classification_logits.shape) == (2, 8)
    assert tuple(output.answer_logits.shape[:2]) == tuple(batch["answer_input_ids"].shape)
    assert tuple(output.onset_logits.shape) == (2, 2, 8)
    assert tuple(output.spatiotemporal_tokens.shape) == (2, 8, 64, 64)
    assert tuple(output.global_temporal_tokens.shape) == (2, 9, 64)
    assert tuple(output.object_tracks.shape) == (2, 2, 8, 64)
    assert tuple(output.relation_token.shape) == (2, 1, 64)
    assert tuple(output.temporal_attention.shape) == (2, 64, 4, 8, 8)
    assert tuple(output.spatial_attention.shape) == (2, 8, 4, 64, 64)
    assert tuple(output.global_temporal_attention.shape) == (2, 4, 9, 9)
    assert tuple(output.object_query_attention.shape) == (2, 2, 8, 4, 64)
    assert tuple(output.track_attention.shape) == (2, 2, 4, 8, 8)
    assert tuple(output.relation_attention.shape) == (2, 4, 1, 27)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) <= 1_000_000


def test_unit_vidqa_i003_003_all_new_mechanisms_receive_gradients():
    batch, model = batch_and_model()
    output = model(
        batch["video"],
        batch["question_ids"],
        batch["answer_input_ids"],
        object_query_masks=batch["object_query_masks"],
    )
    onset_loss = F.cross_entropy(
        output.onset_logits.reshape(-1, 8), batch["move_start_targets"].reshape(-1)
    )
    loss = (
        output.classification_logits.square().mean()
        + output.answer_logits.square().mean()
        + onset_loss
    )
    loss.backward()
    parameters = (
        model.timesformer_blocks[0].temporal_attention.in_proj_weight,
        model.timesformer_blocks[0].spatial_attention.in_proj_weight,
        model.global_layout_projection.weight,
        model.global_temporal_block.attention.in_proj_weight,
        model.object_cross_attention.in_proj_weight,
        model.track_temporal_attention.in_proj_weight,
        model.relation_cross_attention.in_proj_weight,
        model.onset_head[-1].weight,
        model.decoder[0].cross_attention.in_proj_weight,
    )
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in parameters)


def test_unit_vidqa_i003_004_ablations_change_internal_evidence():
    batch, model = batch_and_model()
    model.eval()
    with torch.no_grad():
        full = model(
            batch["video"],
            batch["question_ids"],
            object_query_masks=batch["object_query_masks"],
        )
        no_time = model(
            batch["video"],
            batch["question_ids"],
            object_query_masks=batch["object_query_masks"],
            disable_timesformer_temporal=True,
        )
        global_query = model(
            batch["video"],
            batch["question_ids"],
            object_query_masks=batch["object_query_masks"],
            global_query_ablation=True,
        )
    assert float((full.spatiotemporal_tokens - no_time.spatiotemporal_tokens).abs().max()) > 1e-5
    assert float((full.object_tracks - global_query.object_tracks).abs().max()) > 1e-5


def test_unit_vidqa_i003_005_decoder_is_causal():
    batch, model = batch_and_model()
    model.eval()
    first = batch["answer_input_ids"].clone()
    second = first.clone()
    second[:, -1] = (second[:, -1] + 1) % len(model.token_embedding.weight)
    with torch.no_grad():
        output_a = model(
            batch["video"],
            batch["question_ids"],
            first,
            object_query_masks=batch["object_query_masks"],
        ).answer_logits
        output_b = model(
            batch["video"],
            batch["question_ids"],
            second,
            object_query_masks=batch["object_query_masks"],
        ).answer_logits
    assert torch.allclose(output_a[:, :-1], output_b[:, :-1], atol=1e-6)


def test_art_vidqa_i003_002_capability_media_links_resolve_from_markdown():
    markdown_path = (
        ROOT
        / "results/P3-W01_tiny_video_mllm/iterations/I003/branches/"
          "spacetime_queries/examples/capability_examples.md"
    )
    text = markdown_path.read_text(encoding="utf-8")
    relative_links = sorted(set(re.findall(r"\((media/[^)]+\.(?:gif|png))\)", text)))
    assert len([value for value in relative_links if value.endswith(".gif")]) == 8
    assert len([value for value in relative_links if value.endswith(".png")]) == 8
    assert all((markdown_path.parent / value).is_file() for value in relative_links)


def test_art_vidqa_i003_003_capability_examples_are_stratified():
    path = (
        ROOT
        / "results/P3-W01_tiny_video_mllm/iterations/I003/branches/"
          "spacetime_queries/examples/capability_examples.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload["examples"]
    assert len(examples) == 8
    assert {item["question_kind"] for item in examples} == {
        "motion_direction", "appear_order", "disappear_order", "move_start_order"
    }
    move_start_answers = {
        item["expected_output"] for item in examples if item["question_kind"] == "move_start_order"
    }
    assert move_start_answers == {"前者更早", "后者更早", "同时发生"}
    assert sum(item["expected_output"] == "同时发生" for item in examples) == 1
    assert sum(not item["is_fully_correct"] for item in examples) >= 1
    assert payload["error_or_boundary_count"] >= 3


def test_doc_vidqa_i003_001_summary_contains_executable_reproduction_levels():
    report = (ROOT / "docs/P3-W01_总结报告.md").read_text(encoding="utf-8")
    for required in (
        "### 8.1 先核对文件与迭代状态",
        "### 8.2 复现一条完整推理",
        "### 8.3 在隔离目录复现完整训练",
        "p3_w01_i003_run.py",
        "source_snapshot.zip",
    ):
        assert required in report
