import torch

from mini_multimodal.video_gen.evaluation import movement_signature
from mini_multimodal.video_gen.motion_plan import parse_motion_plan, plan_prompt_batch


def test_parse_delayed_motion_has_exact_semantic_interval():
    plan = parse_motion_plan("对象1是蓝色的大圆形，初始位于左侧区域，从第3帧到第6帧向上移动，第1帧至第8帧可见。")
    signature = movement_signature(plan.trajectory)
    assert plan.initial_region == "左侧"
    assert signature["direction"] == "up"
    assert signature["start"] == 2
    assert signature["end"] == 5


def test_static_plan_stays_at_region_center():
    plan = parse_motion_plan("对象1是红色的小正方形，初始位于右下区域，保持静止，第1帧至第8帧可见。")
    assert torch.allclose(plan.trajectory, torch.tensor([[0.65, 0.65]] * 8))
    assert movement_signature(plan.trajectory)["direction"] == "static"


def test_turn_plan_and_batch_contract():
    prompt = "对象1是绿色的大圆形，初始位于左下区域，从第3帧到第7帧向右移动，从第7帧到第8帧向左移动，第1帧至第8帧可见。"
    plan = parse_motion_plan(prompt)
    batch = plan_prompt_batch([prompt, prompt])
    assert [segment.direction for segment in plan.segments] == ["right", "left"]
    assert batch.shape == (2, 8, 2)
    assert torch.all((batch >= 0) & (batch <= 1))
