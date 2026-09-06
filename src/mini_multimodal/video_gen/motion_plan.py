"""Deterministic semantic motion planning for the bounded P3-W04 language.

The prompt specifies a coarse initial region and motion intervals/directions,
but not an exact coordinate, speed, or easing profile.  This adapter therefore
turns only observable text semantics into an explicit control trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch


_REGION_CENTERS = {
    # Centers remain inside the named coarse region while reserving enough
    # room for seven visible 0.03 movements without boundary clipping.
    "中央": (0.50, 0.50), "左侧": (0.35, 0.50), "右侧": (0.65, 0.50),
    "上方": (0.50, 0.35), "下方": (0.50, 0.65),
    "左上": (0.35, 0.35), "右上": (0.65, 0.35),
    "左下": (0.35, 0.65), "右下": (0.65, 0.65),
}
_DIRECTION = {"左": "left", "右": "right", "上": "up", "下": "down"}
_DELTA = {
    "left": (-1.0, 0.0), "right": (1.0, 0.0),
    "up": (0.0, -1.0), "down": (0.0, 1.0),
}
_SEGMENT = re.compile(r"从第(\d+)帧到第(\d+)帧向(左|右|上|下)移动")


@dataclass(frozen=True)
class MotionSegment:
    direction: str
    start: int
    end: int


@dataclass(frozen=True)
class MotionPlan:
    initial_region: str
    initial_xy: tuple[float, float]
    segments: tuple[MotionSegment, ...]
    trajectory: torch.Tensor


def parse_motion_plan(prompt: str, *, frames: int = 8, step: float | None = None,
                      segment_displacement: float = 0.20) -> MotionPlan:
    """Parse object 1 and create a bounded, constant-speed control trajectory."""
    if frames < 2:
        raise ValueError("frames must be at least 2")
    object_text = prompt.split("；", 1)[0]
    region_match = re.search(r"初始位于(中央|左侧|右侧|上方|下方|左上|右上|左下|右下)区域", object_text)
    if region_match is None:
        raise ValueError(f"unsupported or missing initial region: {object_text}")
    region = region_match.group(1)
    initial = _REGION_CENTERS[region]
    segments = tuple(
        MotionSegment(_DIRECTION[direction], max(0, int(start) - 1), min(frames - 1, int(end) - 1))
        for start, end, direction in _SEGMENT.findall(object_text)
    )
    if "保持静止" not in object_text and not segments:
        raise ValueError(f"unsupported or missing motion clause: {object_text}")
    trajectory = torch.tensor([initial] * frames, dtype=torch.float32)
    current = trajectory[0].clone()
    for frame in range(frames):
        trajectory[frame] = current
        for segment in segments:
            # A segment [s,e] changes coordinates on transitions s->s+1 ... e-1->e.
            if segment.start <= frame < segment.end:
                segment_steps = max(segment.end - segment.start, 1)
                frame_step = float(step) if step is not None else float(segment_displacement) / segment_steps
                current = current + torch.tensor(_DELTA[segment.direction]) * frame_step
                current = current.clamp(0.12, 0.88)
                break
    return MotionPlan(region, initial, segments, trajectory)


def plan_prompt_batch(prompts: list[str], *, frames: int = 8, step: float | None = None,
                      segment_displacement: float = 0.20) -> torch.Tensor:
    if not prompts:
        raise ValueError("prompts must not be empty")
    return torch.stack([parse_motion_plan(prompt, frames=frames, step=step,
                                          segment_displacement=segment_displacement).trajectory
                        for prompt in prompts])
