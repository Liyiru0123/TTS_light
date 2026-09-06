from __future__ import annotations

import inspect
import json
from pathlib import Path

from mini_multimodal.ldm.i005_experiment import (
    BRANCH_IDS,
    _capability_examples,
    _target_margin,
)
from mini_multimodal.ldm.refiner_inference import sample_latents_refined
from mini_multimodal.ldm.revision_evaluation import sample_latents_v2


ROOT = Path(__file__).resolve().parents[1]


def test_exp_i005_001_equal_budget_and_no_overfit_weight_reuse_are_frozen() -> None:
    spec = json.loads((ROOT / "configs/p2_w04_i005.json").read_text(encoding="utf-8"))
    assert tuple(spec["formal_branches"]) == BRANCH_IDS
    assert spec["formal_sampling"] == {
        "sample_steps": 25,
        "guidance_scale": 2.5,
        "basis": "the only sampler setting shortlisted by the frozen preflight; applied equally to both formal branches",
    }
    assert spec["formal_control"]["same_initial_noise_across_branches"] is True
    assert spec["formal_control"]["overfit_weights_reused_in_formal_training"] is False
    assert spec["formal_control"]["automatic_candidate_selection"] is False
    assert spec["clean_refiner"]["finetune_epochs"] == 16


def test_model_ref_005_target_shape_margin_uses_target_against_best_alternative() -> None:
    probe = {
        "objects": [{"shape_scores": {"circle": 0.7, "square": 0.6, "triangle": 0.2}}]
    }
    assert abs(_target_margin(probe, "circle") - 0.1) < 1e-9
    assert abs(_target_margin(probe, "square") + 0.1) < 1e-9
    assert _target_margin({"objects": []}, "circle") == -1.0


def test_del_i005_003_capability_examples_contain_correct_and_error_cases(tmp_path: Path) -> None:
    branch = tmp_path / "branch"
    rows = []
    for index in range(10):
        correct = index < 5
        rows.append({
            "record_id": f"r{index}",
            "prompt": f"prompt {index}",
            "target_object_count": 1,
            "predicted_object_count": 1 if correct else 2,
            "is_fully_correct": correct,
            "generated_image": f"images/{index}.png",
            "pairs": [{"target": {"shape": "circle"}, "prediction": {"shape": "circle"} if correct else {"shape": "square"}}],
        })
    payload = _capability_examples(tmp_path, branch, rows)
    assert len(payload["examples"]) == 8
    assert {item["category"] for item in payload["examples"]} == {"正确", "错误"}
    assert (branch / "examples/capability_examples.json").is_file()
    assert (branch / "examples/capability_examples.md").is_file()


def test_del_i005_005_sampling_interfaces_cannot_accept_target_geometry() -> None:
    forbidden = {"target", "mask", "box", "class_id", "image"}
    for function in (sample_latents_v2, sample_latents_refined):
        parameters = set(inspect.signature(function).parameters)
        assert not (parameters & forbidden)


def test_all_iteration_selection_registers_i004_control_without_false_release() -> None:
    result = ROOT / "results/P2-W04_tiny_image_ldm"
    candidate = json.loads((result / "current_candidate.json").read_text(encoding="utf-8"))
    official = json.loads((result / "official_release.json").read_text(encoding="utf-8"))
    decision = json.loads(
        (result / "candidate_selection/selection_after_i005.json").read_text(encoding="utf-8")
    )
    assert candidate["current_candidate"]["iteration_id"] == "I004"
    assert candidate["current_candidate"]["branch_id"] == "control"
    assert decision["selected_current_final_profile"]["recommended_sampling"] == {
        "sample_steps": 25,
        "guidance_scale": 2.5,
        "basis": "full 420-sample no-training sampler confirmation on the selected checkpoint",
    }
    assert decision["selected_current_final_profile"]["unseen_single_verified"] is True
    assert decision["selected_current_final_profile"]["seen_single_verified"] is False
    assert official["official_release"] is None
