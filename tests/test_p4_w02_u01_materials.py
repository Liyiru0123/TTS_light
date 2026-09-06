"""P4-W02 U01 教学材料的结构、执行证据与版本门测试。"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "教学材料" / "P4单元材料" / "U01_N01-N02"

ASSETS = {
    "ppt": UNIT / "PPT" / "U01_P01-P02_课程简介与张量基础.pptx",
    "teacher_notebook": UNIT
    / "Notebook"
    / "教师演示"
    / "U01_D01-D02_四模态与张量演示.ipynb",
    "student_notebook": UNIT
    / "Notebook"
    / "学生实验"
    / "U01_L01-L02_张量观察练习.ipynb",
    "notes": UNIT / "连续讲义" / "U01_N01-N02_课程简介与张量基础.md",
    "quiz": UNIT / "互动与测试" / "U01_Q01-Q02_入口诊断与后测.md",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ped_u01_001_five_assets_exist_and_are_nonempty() -> None:
    for label, path in ASSETS.items():
        assert path.is_file(), f"missing {label}: {path}"
        assert path.stat().st_size > 500, f"unexpectedly small {label}: {path}"


def test_ped_u01_002_boppps_and_90_minute_script_are_explicit() -> None:
    teacher = read_text(UNIT / "教师参考" / "U01_教师实施说明与答案.md")
    for term in ("导入", "学习目标", "前测", "参与式学习", "后测", "总结"):
        assert term in teacher
    assert "时间总计90分钟" in teacher
    ranges = re.findall(r"\| (\d+)—(\d+)分钟 \|", teacher)
    assert ranges[0] == ("0", "2")
    assert ranges[-1] == ("85", "90")
    assert all(int(end) > int(start) for start, end in ranges)


def test_ped_u01_003_to_005_goals_and_assessment_contract() -> None:
    combined = "\n".join(
        read_text(path)
        for path in (
            ASSETS["notes"],
            ASSETS["quiz"],
            UNIT / "U01_目标活动评价完备性矩阵.md",
        )
    )
    for phrase in (
        "课程主要内容",
        "考核方式",
        "分析",
        "生成",
        "shape",
        "轴语义",
        "| 知识点测验 | 25% |",
        "| 程序测试 | 25% |",
        "| 调研报告 | 10% |",
        "期末考试",
        "40%",
        "具体次数、门槛和期末实施细则",
    ):
        assert phrase in combined


def test_ped_u01_006_real_scene_and_shapes_are_consistent() -> None:
    teacher_nb = json.loads(read_text(ASSETS["teacher_notebook"]))
    serialized = json.dumps(teacher_nb, ensure_ascii=False)
    for value in (
        "core_scene_000000",
        "(64, 64, 3)",
        "(296126,)",
        "(64, 1155)",
        "(8, 64, 64, 3)",
        "tiny_image_ldm_anchor.png",
        "core_scene_000700_u000_normal_1p0.wav",
        "第1帧中有1个",
        "comparison.gif",
    ):
        assert value in serialized


def test_ped_u01_007_notebooks_have_saved_outputs_and_no_errors() -> None:
    for path in (ASSETS["teacher_notebook"], ASSETS["student_notebook"]):
        notebook = json.loads(read_text(path))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells
        assert all(cell.get("execution_count") is not None for cell in code_cells)
        assert all(
            output.get("output_type") != "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        )
        assert any(cell.get("outputs") for cell in code_cells)


def test_ped_u01_015_notebook_plot_environment_and_sample_are_current() -> None:
    for path in (ASSETS["teacher_notebook"], ASSETS["student_notebook"]):
        serialized = json.dumps(json.loads(read_text(path)), ensure_ascii=False)
        assert "%matplotlib inline" in serialized
        assert "Microsoft YaHei" in serialized
        assert "axes.unicode_minus" in serialized
    teacher = json.dumps(
        json.loads(read_text(ASSETS["teacher_notebook"])), ensure_ascii=False
    )
    assert "tiny_speech_flow_i004.wav" not in teacher
    assert "core_scene_000700_u000_normal_1p0.wav" in teacher
    assert "第1帧中有1个" in teacher


def test_ped_u01_008_to_014_learning_cycle_support_links_and_offline_path() -> None:
    notes = read_text(ASSETS["notes"])
    quiz = read_text(ASSETS["quiz"])
    teacher = read_text(UNIT / "教师参考" / "U01_教师实施说明与答案.md")
    matrix = read_text(UNIT / "U01_目标活动评价完备性矩阵.md")
    for phrase in ("问题", "定义", "代码", "输出", "常见错误", "课程模型映射"):
        assert phrase in notes
    for phrase in ("入口诊断", "讨论入口选择", "参与式学习任务", "后测", "出口问题"):
        assert phrase in quiz
    for phrase in ("Notebook启动失败", "音频无法播放", "视频无法播放", "已执行Notebook"):
        assert phrase in teacher
    assert matrix.count("| N0") >= 6
    combined = "\n".join((notes, quiz, teacher, matrix))
    assert "补桥" not in combined
    for phrase in ("PPT 1—2", "D01-2", "L01-1", "讲义§"):
        assert phrase in combined


def test_reg_u01_001_version_is_working_draft_pending_review() -> None:
    registry = json.loads(
        read_text(ROOT / "教学材料" / "教学材料版本登记.json")
    )
    package = registry["materials"]["p4_unit_packages"]
    unit = next(item for item in package["units"] if item["unit_id"] == "U01")
    assert package["state"] == "working_draft"
    assert unit["state"] == "working_draft"
    assert unit["review_gate"] == "pending_teacher_review"
    assert unit["version"] == "v0.4"
    assert len(package["units"]) == 1, "U02 must not be started before teacher approval"


def test_reg_u01_002_readme_paths_resolve() -> None:
    readme = read_text(UNIT / "README.md")
    for path in ASSETS.values():
        relative = path.relative_to(UNIT).as_posix()
        assert f"`{relative}`" in readme


def test_reg_u01_003_ppt_has_36_slides_notes_and_no_placeholders() -> None:
    ppt = ASSETS["ppt"]
    with zipfile.ZipFile(ppt) as archive:
        names = archive.namelist()
        slides = [
            name
            for name in names
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        ]
        notes = [
            name
            for name in names
            if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)
        ]
        xml_text = "".join(
            archive.read(name).decode("utf-8", errors="ignore") for name in slides
        )
        slide2 = archive.read("ppt/slides/slide2.xml").decode(
            "utf-8", errors="ignore"
        )
    assert len(slides) == 36
    assert len(notes) == 36
    assert not re.search(r"\b(?:TODO|TBD)\b|待补|占位符", xml_text, re.IGNORECASE)
    for phrase in (
        "本次课提纲",
        "现在了解你对多模态的已有认识",
        "代码演示",
        "请打开学生Notebook",
        "考核方式：四部分合计100%",
        "期末考试",
        "相同形状，可以有两种合法解释",
        "数据来源 + 接口约定 + 后续算子",
        "y = x.reshape(6, 4)",
        "y = np.reshape(x, (6, 4))",
    ):
        assert phrase in xml_text
    for forbidden in ("读写shape", "解释错误", "入口前测：shape与轴"):
        assert forbidden not in xml_text
    for phrase in ("张量的三项基本信息", "数据类型（dtype）", "不构成选课门槛"):
        assert phrase in xml_text
    agenda = ("产生兴趣", "认识课程", "了解已有认识", "学习张量基础")
    assert all(phrase in slide2 for phrase in agenda)
    assert [slide2.index(phrase) for phrase in agenda] == sorted(
        slide2.index(phrase) for phrase in agenda
    )


def test_reg_u01_004_native_animations_and_embedded_audio_exist() -> None:
    audio = UNIT / "PPT" / "media" / "U01_生成语音_第1帧中有1个.wav"
    assert audio.is_file()
    assert audio.stat().st_size > 1000
    with zipfile.ZipFile(ASSETS["ppt"]) as archive:
        names = archive.namelist()
        timing_slides = {
            int(re.search(r"slide(\d+)\.xml", name).group(1))
            for name in names
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            and b"<p:timing>" in archive.read(name)
        }
        embedded_audio = [name for name in names if name.lower().endswith(".wav")]
    assert {4, 8, 14, 19, 24, 28, 31}.issubset(timing_slides)
    assert embedded_audio
    assert all(archive_name.startswith("ppt/media/") for archive_name in embedded_audio)
