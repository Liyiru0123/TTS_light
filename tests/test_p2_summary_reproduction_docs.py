"""Guard executable inference and reproduction instructions in P2 summaries."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_p2_summaries_contain_executable_reproduction_instructions():
    expectations = {
        "P2-W01_总结报告.md": ("可执行复现与检查点恢复", "run_p2_w01_experiments"),
        "P2-W02_总结报告.md": ("复现单幅重建", "p2_w02_reconstruction.py", "从零复现完整工作包"),
        "P2-W03_总结报告.md": ("直接执行推理与恢复检查", "p2_w03_inference.py", "从零复现完整工作包"),
        "P2-W04_总结报告.md": ("原版检查点的推理与恢复", "p2_w04_inference.py", "原版完整流程复现"),
        "P2-W05_总结报告.md": ("复现一条ASR推理", "p2_w05_i002_inference.py", "在隔离副本复现I002完整流程"),
    }
    for name, required in expectations.items():
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "```powershell" in text
        assert ".venv\\Scripts\\python.exe" in text
        for value in required:
            assert value in text


def test_p2_runnable_examples_exist():
    for relative in (
        "examples/p2_w02_reconstruction.py",
        "examples/p2_w03_inference.py",
        "examples/p2_w03_resume_checkpoint.py",
        "examples/p2_w04_inference.py",
        "examples/p2_w04_resume_checkpoint.py",
        "examples/p2_w05_i002_inference.py",
        "examples/p2_w05_i002_resume_checkpoint.py",
    ):
        assert (ROOT / relative).is_file()
