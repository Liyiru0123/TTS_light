from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.vlm import (  # noqa: E402
    CharacterTokenizer,
    TinyVLM,
    VLMQADataset,
    VLMTask,
    collate_vlm_qa,
)


CONFIG = json.loads((ROOT / "configs" / "p2_w03_tiny_vlm.json").read_text(encoding="utf-8"))
VOCABULARY = json.loads((ROOT / CONFIG["output_data"] / "vocabulary.json").read_text(encoding="utf-8"))
ANSWER_CLASSES = json.loads((ROOT / CONFIG["output_data"] / "answer_classes.json").read_text(encoding="utf-8"))


def make_model() -> TinyVLM:
    tokenizer = CharacterTokenizer(VOCABULARY)
    return TinyVLM(
        vocabulary_size=len(tokenizer),
        answer_class_count=len(ANSWER_CLASSES),
        pad_id=tokenizer.id("<pad>"),
        bos_id=tokenizer.id("<bos>"),
        eos_id=tokenizer.id("<eos>"),
        mask_id=tokenizer.id("<mask>"),
        target_marker_id=tokenizer.id("甲"),
        reference_marker_id=tokenizer.id("乙"),
        **CONFIG["model"],
    )


class TinyVLMUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        overfit_ids = json.loads((ROOT / CONFIG["output_data"] / "overfit_ids.json").read_text(encoding="utf-8"))
        dataset = VLMQADataset(
            ROOT,
            data_dir=CONFIG["output_data"],
            source_latents=CONFIG["source_latents"],
            splits="train",
            record_ids=overfit_ids[:3],
        )
        cls.batch = collate_vlm_qa([dataset[0], dataset[1], dataset[2]])

    def test_unit_vlm_001_tensor_shapes_attention_and_probabilities(self) -> None:
        model = make_model().eval()
        with torch.no_grad():
            output = model(
                self.batch["visual"],
                self.batch["question_ids"],
                self.batch["answer_input_ids"],
            )
        batch_size, question_length = self.batch["question_ids"].shape
        answer_length = self.batch["answer_input_ids"].shape[1]
        self.assertEqual(tuple(output.visual_tokens.shape), (batch_size, 256, 64))
        self.assertEqual(tuple(output.text_tokens.shape), (batch_size, question_length, 64))
        self.assertEqual(tuple(output.classification_logits.shape), (batch_size, 21))
        self.assertEqual(tuple(output.answer_logits.shape), (batch_size, answer_length, len(VOCABULARY["tokens"])))
        self.assertEqual(tuple(output.fusion_attention.shape), (batch_size, 4, question_length, 256))
        self.assertEqual(tuple(output.target_coordinates.shape), (batch_size, 2))
        self.assertEqual(tuple(output.target_attention.shape), (batch_size, 256))
        self.assertEqual(output.decoder_attention.shape[-1], 256 + question_length + 1)
        self.assertTrue(torch.allclose(output.fusion_attention.sum(dim=-1), torch.ones_like(output.fusion_attention.sum(dim=-1)), atol=1e-5))

    def test_unit_vlm_002_causal_mask_blocks_future_tokens(self) -> None:
        model = make_model().eval()
        with torch.no_grad():
            _, _, _, memory, memory_padding = model.encode_multimodal(
                self.batch["visual"], self.batch["question_ids"]
            )
            first = self.batch["answer_input_ids"].clone()
            second = first.clone()
            if second.shape[1] < 4:
                second = torch.nn.functional.pad(second, (0, 4 - second.shape[1]))
                first = torch.nn.functional.pad(first, (0, 4 - first.shape[1]))
            second[:, 2:] = torch.where(second[:, 2:].eq(0), second[:, 2:], torch.full_like(second[:, 2:], 7))
            first_logits, _ = model.decode_teacher_forced(first, memory, memory_padding)
            second_logits, _ = model.decode_teacher_forced(second, memory, memory_padding)
            conditioned_logits, _ = model.decode_teacher_forced(
                first, memory, memory_padding, torch.ones(first.shape[0], 64)
            )
        self.assertTrue(torch.equal(first_logits[:, :2], second_logits[:, :2]))
        self.assertFalse(torch.equal(first_logits, conditioned_logits))

    def test_unit_vlm_003_joint_loss_reaches_all_core_modules(self) -> None:
        model = make_model()
        task = VLMTask(pad_id=0, classification_weight=0.5, generation_weight=1.0)
        output = task.step(model, self.batch, training=True)
        output.loss.backward()
        groups = {
            "visual_connector": model.visual_connector,
            "coordinate_projection": model.coordinate_projection,
            "token_embedding": model.token_embedding,
            "text_encoder": model.text_encoder,
            "fusion": model.fusion,
            "fusion_reasoner": model.fusion_reasoner,
            "classifier": model.classifier,
            "decoder": model.decoder,
        }
        for name, module in groups.items():
            with self.subTest(module=name):
                gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_unit_vlm_004_ablation_switches_preserve_shape_and_change_flow(self) -> None:
        model = make_model().eval()
        with torch.no_grad():
            normal = model(self.batch["visual"], self.batch["question_ids"])
            image_masked = model(self.batch["visual"], self.batch["question_ids"], mask_image=True)
            text_masked = model(self.batch["visual"], self.batch["question_ids"], mask_text=True)
            no_position = model(self.batch["visual"], self.batch["question_ids"], disable_visual_position=True)
            no_fusion = model(self.batch["visual"], self.batch["question_ids"], disable_fusion=True)
        for altered in (image_masked, text_masked, no_position, no_fusion):
            self.assertEqual(altered.classification_logits.shape, normal.classification_logits.shape)
        self.assertFalse(torch.equal(normal.visual_tokens, image_masked.visual_tokens))
        self.assertFalse(torch.equal(normal.text_tokens, text_masked.text_tokens))
        self.assertFalse(torch.equal(normal.visual_tokens, no_position.visual_tokens))
        self.assertTrue(torch.count_nonzero(no_position.target_coordinates) == 0)
        self.assertTrue(torch.count_nonzero(no_position.reference_coordinates) == 0)
        self.assertTrue(torch.count_nonzero(no_fusion.fusion_attention) == 0)

    def test_unit_vlm_004b_coordinates_preserve_axis_order(self) -> None:
        model = make_model()
        coordinates = model.spatial_coordinates.reshape(16, 16, 2)
        self.assertLess(float(coordinates[0, 0, 0]), float(coordinates[0, -1, 0]))
        self.assertLess(float(coordinates[0, 0, 1]), float(coordinates[-1, 0, 1]))

    def test_unit_vlm_005_greedy_generation_is_bounded(self) -> None:
        model = make_model().eval()
        generated, probabilities = model.generate(
            self.batch["visual"], self.batch["question_ids"], max_new_tokens=5
        )
        self.assertEqual(generated.shape, probabilities.shape)
        self.assertEqual(generated.shape[0], 3)
        self.assertLessEqual(generated.shape[1], 5)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())


if __name__ == "__main__":
    unittest.main()
