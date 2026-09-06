"""Training task adapters for classification and causal VQA stages."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from mini_multimodal.training import StepOutput


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


class VLMTask:
    def __init__(
        self,
        *,
        pad_id: int,
        classification_weight: float,
        generation_weight: float,
        grounding_weight: float = 0.0,
    ) -> None:
        if classification_weight < 0 or generation_weight < 0 or grounding_weight < 0 or classification_weight + generation_weight + grounding_weight <= 0:
            raise ValueError("VLM task weights must be non-negative and non-zero")
        self.pad_id = int(pad_id)
        self.classification_weight = float(classification_weight)
        self.generation_weight = float(generation_weight)
        self.grounding_weight = float(grounding_weight)

    def step(
        self, model: nn.Module, batch: Mapping[str, Any], *, training: bool
    ) -> StepOutput:
        use_generation = self.generation_weight > 0
        output = model(
            batch["visual"],
            batch["question_ids"],
            batch["answer_input_ids"] if use_generation else None,
        )
        weights = batch["sample_weight"]
        class_per_sample = F.cross_entropy(
            output.classification_logits, batch["answer_class"], reduction="none"
        )
        class_loss = _weighted_mean(class_per_sample, weights)
        class_accuracy = (
            output.classification_logits.argmax(dim=-1) == batch["answer_class"]
        ).float().mean()
        generation_loss = torch.zeros((), device=class_loss.device)
        grounding_loss = torch.zeros((), device=class_loss.device)
        grounding_mae = torch.zeros((), device=class_loss.device)
        token_accuracy = sequence_accuracy = torch.zeros((), device=class_loss.device)
        if use_generation:
            assert output.answer_logits is not None
            targets = batch["answer_target_ids"]
            token_losses = F.cross_entropy(
                output.answer_logits.transpose(1, 2),
                targets,
                reduction="none",
                ignore_index=self.pad_id,
            )
            valid = targets.ne(self.pad_id)
            per_sample = (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            generation_loss = _weighted_mean(per_sample, weights)
            predictions = output.answer_logits.argmax(dim=-1)
            token_accuracy = ((predictions == targets) & valid).sum() / valid.sum().clamp_min(1)
            sequence_accuracy = (((predictions == targets) | ~valid).all(dim=1)).float().mean()
        if self.grounding_weight > 0:
            predicted_positions = torch.cat(
                (output.target_coordinates, output.reference_coordinates), dim=0
            )
            target_positions = torch.cat(
                (batch["target_position"], batch["reference_position"]), dim=0
            )
            position_weights = torch.cat((weights, weights), dim=0)
            position_valid = torch.isfinite(target_positions).all(dim=1)
            if bool(position_valid.any()):
                coordinate_error = F.smooth_l1_loss(
                    predicted_positions[position_valid],
                    target_positions[position_valid],
                    reduction="none",
                ).mean(dim=1)
                grounding_loss = _weighted_mean(
                    coordinate_error, position_weights[position_valid]
                )
                absolute_error = (
                    predicted_positions[position_valid] - target_positions[position_valid]
                ).abs().mean(dim=1)
                grounding_mae = _weighted_mean(
                    absolute_error, position_weights[position_valid]
                )
        loss = (
            self.classification_weight * class_loss
            + self.generation_weight * generation_loss
            + self.grounding_weight * grounding_loss
        )
        return StepOutput(
            loss=loss,
            metrics={
                "classification_loss": float(class_loss.detach()),
                "generation_loss": float(generation_loss.detach()),
                "grounding_loss": float(grounding_loss.detach()),
                "grounding_mae": float(grounding_mae.detach()),
                "classification_accuracy": float(class_accuracy.detach()),
                "token_accuracy": float(token_accuracy.detach()),
                "sequence_accuracy": float(sequence_accuracy.detach()),
            },
            weight=len(batch["visual"]),
        )
