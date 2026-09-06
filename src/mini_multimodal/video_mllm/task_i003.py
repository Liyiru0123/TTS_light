"""Training objective for the P3-W01 I003 model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_multimodal.training import StepOutput


class TinyVideoMLLMI003Task:
    def __init__(
        self,
        *,
        pad_id: int,
        classification_weight: float = 1.0,
        generation_weight: float = 1.0,
        onset_weight: float = 0.5,
        mask_video: bool = False,
        single_frame_index: int | None = None,
    ) -> None:
        self.pad_id = int(pad_id)
        self.classification_weight = float(classification_weight)
        self.generation_weight = float(generation_weight)
        self.onset_weight = float(onset_weight)
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
            object_query_masks=batch["object_query_masks"],
            mask_video=self.mask_video,
        )
        classification_loss = F.cross_entropy(
            output.classification_logits, batch["answer_class"]
        )
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
        valid_onset = batch["move_start_targets"].ge(0)
        if bool(valid_onset.any()):
            onset_loss = F.cross_entropy(
                output.onset_logits[valid_onset],
                batch["move_start_targets"][valid_onset],
            )
            predicted = output.onset_logits.argmax(dim=-1)
            onset_accuracy = predicted[valid_onset].eq(
                batch["move_start_targets"][valid_onset]
            ).float().mean()
            predicted_relation = torch.sign(predicted[:, 1] - predicted[:, 0])
            target_relation = torch.sign(
                batch["move_start_targets"][:, 1]
                - batch["move_start_targets"][:, 0]
            )
            valid_rows = valid_onset.all(dim=1)
            onset_relation_accuracy = predicted_relation[valid_rows].eq(
                target_relation[valid_rows]
            ).float().mean()
        else:
            onset_loss = classification_loss.new_zeros(())
            onset_accuracy = classification_loss.new_zeros(())
            onset_relation_accuracy = classification_loss.new_zeros(())
        loss = (
            self.classification_weight * classification_loss
            + self.generation_weight * generation_loss
            + self.onset_weight * onset_loss
        )
        classification_accuracy = output.classification_logits.argmax(dim=-1).eq(
            batch["answer_class"]
        ).float().mean()
        if output.answer_logits is not None:
            valid_tokens = batch["answer_target_ids"].ne(self.pad_id)
            token_accuracy = output.answer_logits.argmax(dim=-1).eq(
                batch["answer_target_ids"]
            )[valid_tokens].float().mean()
        else:
            token_accuracy = classification_accuracy
        joint_score = 0.5 * (classification_accuracy + token_accuracy)
        return StepOutput(
            loss,
            {
                "classification_loss": float(classification_loss.detach()),
                "generation_loss": float(generation_loss.detach()),
                "onset_loss": float(onset_loss.detach()),
                "classification_accuracy": float(classification_accuracy.detach()),
                "token_accuracy": float(token_accuracy.detach()),
                "onset_accuracy": float(onset_accuracy.detach()),
                "onset_relation_accuracy": float(onset_relation_accuracy.detach()),
                "joint_score": float(joint_score.detach()),
            },
            int(video.shape[0]),
        )
