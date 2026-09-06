"""Training objective for TinyVideoMLLM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_multimodal.training import StepOutput


class TinyVideoMLLMTask:
    def __init__(
        self,
        *,
        pad_id: int,
        classification_weight: float = 1.0,
        generation_weight: float = 1.0,
        mask_video: bool = False,
        single_frame_index: int | None = None,
    ) -> None:
        self.pad_id = int(pad_id)
        self.classification_weight = float(classification_weight)
        self.generation_weight = float(generation_weight)
        self.mask_video = bool(mask_video)
        self.single_frame_index = single_frame_index

    def step(self, model: torch.nn.Module, batch: dict, *, training: bool) -> StepOutput:
        video = batch["video"]
        if self.single_frame_index is not None:
            frame = video[:, self.single_frame_index : self.single_frame_index + 1]
            video = frame.expand(-1, video.shape[1], -1, -1, -1)
        answer_inputs = batch["answer_input_ids"] if self.generation_weight > 0 else None
        output = model(
            video,
            batch["question_ids"],
            answer_inputs,
            mask_video=self.mask_video,
        )
        classification_loss = F.cross_entropy(output.classification_logits, batch["answer_class"])
        if self.generation_weight > 0:
            if output.answer_logits is None:
                raise AssertionError("teacher-forced answer logits are absent")
            generation_loss = F.cross_entropy(
                output.answer_logits.reshape(-1, output.answer_logits.shape[-1]),
                batch["answer_target_ids"].reshape(-1),
                ignore_index=self.pad_id,
            )
        else:
            generation_loss = classification_loss.new_zeros(())
        loss = self.classification_weight * classification_loss + self.generation_weight * generation_loss
        classification_accuracy = output.classification_logits.argmax(dim=-1).eq(batch["answer_class"]).float().mean()
        if output.answer_logits is not None:
            valid = batch["answer_target_ids"].ne(self.pad_id)
            token_accuracy = (
                output.answer_logits.argmax(dim=-1).eq(batch["answer_target_ids"])[valid].float().mean()
            )
        else:
            token_accuracy = classification_accuracy
        joint_score = 0.5 * (classification_accuracy + token_accuracy)
        return StepOutput(
            loss,
            {
                "classification_loss": float(classification_loss.detach()),
                "generation_loss": float(generation_loss.detach()),
                "classification_accuracy": float(classification_accuracy.detach()),
                "token_accuracy": float(token_accuracy.detach()),
                "joint_score": float(joint_score.detach()),
            },
            int(video.shape[0]),
        )
