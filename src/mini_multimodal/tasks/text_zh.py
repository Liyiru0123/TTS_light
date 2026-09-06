"""Deterministic Simplified-Chinese rendering for structured tasks."""

from __future__ import annotations

from mini_multimodal.world.schema import Color, MotionDirection, Scene, SceneObject, Shape, Size

from .reasoning import object_by_id, position_region, resolve_event, visible_objects
from .schema import (
    AnswerSpec,
    EventRef,
    EventType,
    GenerationCondition,
    HorizontalRelation,
    PositionRegion,
    QuestionKind,
    QuestionSpec,
    TargetModality,
    VerticalRelation,
)


COLOR_ZH = {
    Color.RED.value: "红色",
    Color.BLUE.value: "蓝色",
    Color.GREEN.value: "绿色",
    Color.YELLOW.value: "黄色",
    Color.PURPLE.value: "紫色",
    Color.ORANGE.value: "橙色",
}
SHAPE_ZH = {
    Shape.CIRCLE.value: "圆形",
    Shape.SQUARE.value: "正方形",
    Shape.TRIANGLE.value: "三角形",
    Shape.RECTANGLE.value: "矩形",
    Shape.PENTAGON.value: "五边形",
}
SIZE_ZH = {
    Size.SMALL.value: "小",
    Size.MEDIUM.value: "中等大小",
    Size.LARGE.value: "大",
}
MOTION_ZH = {
    MotionDirection.STATIC.value: "保持静止",
    MotionDirection.LEFT.value: "向左移动",
    MotionDirection.RIGHT.value: "向右移动",
    MotionDirection.UP.value: "向上移动",
    MotionDirection.DOWN.value: "向下移动",
}
HORIZONTAL_ZH = {
    HorizontalRelation.LEFT_OF.value: "在左边",
    HorizontalRelation.RIGHT_OF.value: "在右边",
    HorizontalRelation.ALIGNED.value: "水平对齐",
}
VERTICAL_ZH = {
    VerticalRelation.ABOVE.value: "在上方",
    VerticalRelation.BELOW.value: "在下方",
    VerticalRelation.ALIGNED.value: "垂直对齐",
}
REGION_ZH = {
    PositionRegion.UPPER_LEFT.value: "左上区域",
    PositionRegion.UPPER_CENTER.value: "上方区域",
    PositionRegion.UPPER_RIGHT.value: "右上区域",
    PositionRegion.CENTER_LEFT.value: "左侧区域",
    PositionRegion.CENTER.value: "中央区域",
    PositionRegion.CENTER_RIGHT.value: "右侧区域",
    PositionRegion.LOWER_LEFT.value: "左下区域",
    PositionRegion.LOWER_CENTER.value: "下方区域",
    PositionRegion.LOWER_RIGHT.value: "右下区域",
}
EVENT_ZH = {
    EventType.APPEAR: "出现",
    EventType.DISAPPEAR: "消失",
    EventType.MOVE_START: "开始移动",
    EventType.MOVE_END: "结束移动",
}


class ChineseTextAdapter:
    """Render stable Chinese text without changing structured semantics."""

    language = "zh-CN"
    template_version = 1

    def object_number(self, scene: Scene, object_id: str) -> int:
        for index, obj in enumerate(scene.objects, start=1):
            if obj.object_id == object_id:
                return index
        raise ValueError(f"Unknown object id {object_id!r} in scene {scene.scene_id!r}")

    def object_ref(self, scene: Scene, object_id: str, *, with_attributes: bool = False) -> str:
        obj = object_by_id(scene, object_id)
        label = f"对象{self.object_number(scene, object_id)}"
        if not with_attributes:
            return label
        attributes = f"{COLOR_ZH[obj.color.value]}{SIZE_ZH[obj.size.value]}{SHAPE_ZH[obj.shape.value]}"
        return f"{label}（{attributes}）"

    def object_fact(self, scene: Scene, obj: SceneObject, frame: int) -> str:
        label = self.object_ref(scene, obj.object_id)
        region = REGION_ZH[position_region(obj.position_at(frame)).value]
        return (
            f"{label}是{COLOR_ZH[obj.color.value]}的{SIZE_ZH[obj.size.value]}"
            f"{SHAPE_ZH[obj.shape.value]}，位于{region}"
        )

    def describe_scene(self, scene: Scene, frame: int = 0) -> str:
        objects = visible_objects(scene, frame)
        prefix = f"第{frame + 1}帧中有{len(objects)}个可见对象。"
        if not objects:
            return prefix
        facts = "；".join(self.object_fact(scene, obj, frame) for obj in objects)
        return f"{prefix}{facts}。"

    def describe_video(self, scene: Scene) -> str:
        facts: list[str] = []
        for obj in scene.objects:
            base = self.object_fact(scene, obj, 0).replace("位于", "初始位于")
            motion = MOTION_ZH[obj.motion.direction.value]
            visibility = f"第{obj.visible_start + 1}帧至第{obj.visible_end + 1}帧可见"
            if obj.motion.direction is MotionDirection.STATIC:
                facts.append(f"{base}，{motion}，{visibility}")
            else:
                facts.append(
                    f"{base}，从第{obj.motion.start_frame + 1}帧到第{obj.motion.end_frame + 1}帧"
                    f"{motion}，{visibility}"
                )
        return "；".join(facts) + "。"

    def event_text(self, scene: Scene, event_ref: EventRef) -> str:
        resolve_event(scene, event_ref)
        return f"{self.object_ref(scene, event_ref.object_id)}{EVENT_ZH[event_ref.event_type]}"

    def render_question(self, scene: Scene, question: QuestionSpec) -> str:
        if question.scene_id != scene.scene_id:
            raise ValueError("Question and scene ids do not match")
        kind = question.kind
        if kind is QuestionKind.OBJECT_COUNT:
            assert question.frame is not None
            return f"第{question.frame + 1}帧中有几个可见对象？"
        if kind is QuestionKind.EVENT_ORDER:
            assert question.first_event is not None and question.second_event is not None
            first = self.event_text(scene, question.first_event)
            second = self.event_text(scene, question.second_event)
            return f"{first}和{second}，哪一个更早发生？"

        assert question.target_object_id is not None
        target = self.object_ref(scene, question.target_object_id, with_attributes=False)
        if kind is QuestionKind.OBJECT_COLOR:
            return f"{target}是什么颜色？"
        if kind is QuestionKind.OBJECT_SHAPE:
            return f"{target}是什么形状？"
        if kind is QuestionKind.OBJECT_SIZE:
            return f"{target}有多大？"
        if kind is QuestionKind.MOTION_DIRECTION:
            return f"{target}如何运动？"

        assert question.reference_object_id is not None and question.frame is not None
        reference = self.object_ref(scene, question.reference_object_id, with_attributes=False)
        if kind is QuestionKind.HORIZONTAL_RELATION:
            return f"第{question.frame + 1}帧中，{target}与{reference}在水平方向是什么关系？"
        if kind is QuestionKind.VERTICAL_RELATION:
            return f"第{question.frame + 1}帧中，{target}与{reference}在垂直方向是什么关系？"
        raise ValueError(f"Unsupported question kind: {kind}")

    def render_answer(self, question: QuestionSpec, answer: AnswerSpec) -> str:
        value = answer.value
        if question.kind is QuestionKind.OBJECT_COUNT:
            return f"{value}个"
        if question.kind is QuestionKind.OBJECT_COLOR:
            return COLOR_ZH[str(value)]
        if question.kind is QuestionKind.OBJECT_SHAPE:
            return SHAPE_ZH[str(value)]
        if question.kind is QuestionKind.OBJECT_SIZE:
            return SIZE_ZH[str(value)]
        if question.kind is QuestionKind.MOTION_DIRECTION:
            return MOTION_ZH[str(value)]
        if question.kind is QuestionKind.HORIZONTAL_RELATION:
            return HORIZONTAL_ZH[str(value)]
        if question.kind is QuestionKind.VERTICAL_RELATION:
            return VERTICAL_ZH[str(value)]
        if question.kind is QuestionKind.EVENT_ORDER:
            return {
                "first_event": "前者更早",
                "second_event": "后者更早",
                "same_time": "二者同时",
            }[str(value)]
        raise ValueError(f"Unsupported question kind: {question.kind}")

    def render_condition(self, scene: Scene, condition: GenerationCondition) -> str:
        if condition.scene_id != scene.scene_id:
            raise ValueError("Condition and scene ids do not match")
        known_ids = {obj.object_id for obj in scene.objects}
        if any(object_id not in known_ids for object_id in condition.object_ids):
            raise ValueError("Condition references an unknown scene object")
        if condition.target_modality in {TargetModality.IMAGE, TargetModality.SPEECH}:
            assert condition.frame is not None
            return self.describe_scene(scene, condition.frame)
        return self.describe_video(scene)
