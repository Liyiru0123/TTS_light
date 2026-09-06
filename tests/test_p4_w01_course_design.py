from __future__ import annotations

import re
import unittest
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
COURSE_ROOT = RESEARCH_ROOT.parent
AUTHORITY = COURSE_ROOT / "课程顶层设计" / "32学时知识依赖与教学单元设计.md"
PLAN = COURSE_ROOT / "课程顶层设计" / "32学时总体规划.md"


class P4W01CourseDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authority = AUTHORITY.read_text(encoding="utf-8")
        cls.plan = PLAN.read_text(encoding="utf-8")

    def test_ped_p4d_001_n01_n32_audit_rows_are_complete(self) -> None:
        rows = re.findall(r"^\| N(0[1-9]|[12][0-9]|3[0-2]) \|", self.authority, re.M)
        self.assertEqual(rows, [f"{index:02d}" for index in range(1, 33)])

    def test_ped_p4d_002_u01_u16_headings_are_complete(self) -> None:
        units = re.findall(r"^### U(0[1-9]|1[0-6])（N\d{2}—N\d{2}）", self.authority, re.M)
        self.assertEqual(units, [f"{index:02d}" for index in range(1, 17)])

    def test_ped_p4d_003_each_unit_contains_boppps(self) -> None:
        starts = list(re.finditer(r"^### U(?:0[1-9]|1[0-6])（.*$", self.authority, re.M))
        self.assertEqual(len(starts), 16)
        for index, start in enumerate(starts):
            end = starts[index + 1].start() if index + 1 < len(starts) else self.authority.index("\n## 7.", start.start())
            block = self.authority[start.start():end]
            for label in ("导入", "目标", "前测", "参与式学习", "后测", "总结"):
                with self.subTest(unit=index + 1, label=label):
                    self.assertIn(label, block)

    def test_ped_p4d_003_u01_includes_course_content_and_assessment_goal(self) -> None:
        start = self.authority.index("### U01")
        end = self.authority.index("### U02")
        unit = self.authority[start:end]
        self.assertIn("了解课程教学内容、32学时学习路线和考核方式", unit)
        self.assertIn("教学内容—学习活动—考核证据", unit)

    def test_ped_p4d_004_to_008_dependency_and_zero_basis_contracts_exist(self) -> None:
        for phrase in (
            "不要求预先学习机器学习、神经网络",
            "首次建构",
            "激活复用",
            "条件性基础支持",
            "每个核心知识点只有一个明确的首次建构责任位置",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.authority)

    def test_ped_p4d_009_to_010_course_spine_and_generation_boundaries_exist(self) -> None:
        for phrase in (
            "Representation（表示及其分辨率）",
            "Alignment/Correspondence（谁与谁对应）",
            "Fusion/Interaction（对应后怎样交换信息）",
            "图像阶段完整学习一次扩散",
            "Flow是另一种概率运输视角",
            "视频新增的是动力学和时间一致性",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.authority)

    def test_ped_p4d_011_plan_has_32_hours_and_three_experiments(self) -> None:
        hour_rows = re.findall(r"^\| (\d{1,2}) \| (?:讲授|实践|讲授/研讨) \|", self.plan, re.M)
        self.assertEqual([int(item) for item in hour_rows], list(range(1, 33)))
        self.assertEqual(sum(1 for item in hour_rows if item in {"7", "8", "15", "16", "23", "24"}), 6)
        self.assertIn("**32**", self.plan)
        self.assertIn("**26**", self.plan)
        self.assertIn("**6**", self.plan)

    def test_ped_p4d_012_to_015_p4_governance_contracts_exist(self) -> None:
        targets = {
            RESEARCH_ROOT / "最小系统研发规格与约束.md": ("N01—N32知识依赖审计", "不开发独立试讲版本"),
            RESEARCH_ROOT / "docs" / "最小系统开发路线图.md": ("U01—U16", "每单元教师审查门"),
            RESEARCH_ROOT / "docs" / "项目记忆与工作包检查机制.md": ("单元许可", "不等于`approved_release`"),
            COURSE_ROOT / "课程顶层设计" / "课程建设目录与教学材料版本机制.md": ("允许进入下一单元", "`working_draft`"),
        }
        for path, phrases in targets.items():
            text = path.read_text(encoding="utf-8")
            for phrase in phrases:
                with self.subTest(path=path.name, phrase=phrase):
                    self.assertIn(phrase, text)

    def test_reg_p4d_001_to_002_active_documents_are_consistent(self) -> None:
        route = (RESEARCH_ROOT / "docs" / "最小系统开发路线图.md").read_text(encoding="utf-8")
        mapping = (COURSE_ROOT / "课程顶层设计" / "最小系统技术路线与教学知识点映射.md").read_text(encoding="utf-8")
        self.assertIn("P4-W01` | 已验收", route)
        self.assertNotIn("候选单元", mapping)
        self.assertNotIn("N13—N14还是N10—N11", self.plan)


if __name__ == "__main__":
    unittest.main()
