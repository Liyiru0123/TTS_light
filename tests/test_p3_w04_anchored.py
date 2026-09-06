from mini_multimodal.video_gen.anchored import parse_object_appearance, render_planned_video


def test_anchored_prompt_contract_and_video_shape():
    prompt = "对象1是紫色的中等大小五边形，初始位于左侧区域，从第3帧到第7帧向右移动，第1帧至第8帧可见。"
    appearance = parse_object_appearance(prompt)
    frames, target, plan = render_planned_video(prompt)
    assert (appearance.color, appearance.size, appearance.shape) == ("purple", "medium", "pentagon")
    assert frames.shape == (8, 64, 64, 3)
    assert target["trajectory"] == plan.trajectory.tolist()
