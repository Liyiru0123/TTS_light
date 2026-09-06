"""Move newly added iteration-trajectory sections behind the former final section.

This is an idempotent, repository-specific migration.  The first trajectory
audit inserted several new sections immediately after an existing final
heading; this utility restores the old heading/body adjacency and leaves the
trajectory section as the new final section.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# path: (new trajectory heading, first line of the former final-section body)
MOVES = {
    "docs/P2-W01_共享训练框架与判别基线设计.md": ("## 9. 正式模型周期与开发变更说明", "若路由数量或题型"),
    "docs/P2-W01_测试计划.md": ("## 5. 正式模型周期与测试变化说明", "资源超限但正确性通过时"),
    "docs/P2-W02_图像VAE设计.md": ("## 10. 正式模型周期与设计收敛轨迹", "32/64通道、4线程原型"),
    "docs/P2-W02_测试计划.md": ("## 5. 正式模型周期与测试收敛轨迹", "阈值只允许依据重复实验"),
    "docs/P2-W02_图像VAE模型卡.md": ("## 10. 正式模型周期与结构来源", "感知损失与Patch判别器"),
    "docs/P2-W03_TinyVLM设计.md": ("## 9. 正式模型周期与结构收敛轨迹", "Tokenizer、文本Self-Attention"),
    "docs/P2-W03_测试计划.md": ("## 5. 正式模型周期与测试收敛轨迹", "阈值只允许在保留失败证据"),
    "docs/P2-W03_TinyVLM模型卡.md": ("## 10. 正式模型周期与结构来源", "CLIP/SigLIP视觉预训练"),
    "docs/P2-W03_总结报告.md": ("## 11. 技术变更轨迹适用性", "P2-W03于2026年8月8日"),
    "docs/P2-W04_TinyImageLDM设计.md": ("## 9. I001—I005技术设计演化", "- `DATA-LDC-001`未关闭"),
    "docs/P2-W04_测试计划.md": ("## 6. I001—I005评价协议演化", "- 所有比例同时保存"),
    "docs/P2-W04_TinyImageLDM模型卡.md": ("## 6. 模型结构谱系与当前候选来源", "适用于课堂讲解潜空间扩散"),
    "docs/P2-W04_总结报告.md": ("## 10. I001—I005完整技术变更轨迹", "P2-W04保持“进行中”"),
    "docs/P2-W05_TinyASR设计.md": ("## 7. I001—I002技术设计演化", "I001先验证闭环与教学充分性"),
    "docs/P2-W05_测试计划.md": ("## 5. I001—I002测试协议演化", "所有指标使用未舍入原值"),
    "docs/P2-W05_TinyASR模型卡.md": ("## I001—I002模型谱系", "I002保持411,691个参数"),
    "docs/P2-W05_总结报告.md": ("## 13. 完整技术变更轨迹", "| 证据路径 | 用途 |"),
    "docs/P3-W01_TinyVideoMLLM设计.md": ("## 8. I001—I003技术设计演化", "若时间挑战池无法消除答案"),
    "docs/P3-W01_测试计划.md": ("## I001—I003测试协议演化", "阻断性验收项不通过时"),
    "docs/P3-W01_TinyVideoMLLM模型卡.md": ("## 7. I001—I003结构谱系", "- 机器验收："),
    "docs/P3-W01_总结报告.md": ("## 12. I001—I003完整技术变更轨迹", "教师已于2026年8月17日"),
    "docs/P3-W02_语音潜表示与固定解码链设计.md": ("## 7. I001—I002技术设计演化", "若原始Log-Mel经固定解码链"),
    "docs/P3-W02_测试计划.md": ("## 迭代测试协议演化", "> 状态：已完成，待确认"),
    "docs/P3-W02_语音潜表示模型卡.md": ("## I001—I002模型谱系", "沿AudioLDM理解连续声谱VAE"),
    "docs/P3-W02_总结报告.md": ("## 13. 完整技术变更轨迹", "- `data/speech_vae/p3_w02_hifigan_v2/data_audit.json`"),
    "docs/P3-W03_TinyTTS设计.md": ("## 13. I001—I004技术设计演化", "| 编号 | 说明 |"),
    "docs/P3-W03_测试计划.md": ("## 9. I001—I004测试协议演化", "| 交付物 | 路径 | 状态 |"),
    "docs/P3-W03_TinyTTS模型卡.md": ("## 12. 模型结构与训练路线谱系", "复核发现旧语速演示存在三项"),
    "docs/P3-W04_TinyT2V设计.md": ("## 14. I001—I008设计决策总表", "PF011在20条边界运动样例上证明"),
    "docs/P3-W04_测试计划.md": ("## 11. I001—I008测试协议与评价资格演化", "| 验收项编号 | 检查 | 通过条件 | 失败含义 |"),
    "docs/P3-W04_TinyT2V模型卡.md": ("## 9. I001—I008模型结构谱系", "- 教学验收决定："),
    "docs/P3-W04_总结报告.md": ("## 13. 技术轨迹的教学读取方式", "- 全部I008阻塞机器验收项通过"),
}


def normalize(path: Path, heading: str, old_body_start: str) -> str:
    text = path.read_text(encoding="utf-8")
    heading_pos = text.find(heading)
    if heading_pos < 0:
        return "missing_heading"
    body_pos = text.find(old_body_start, heading_pos + len(heading))
    if body_pos < 0:
        # Already moved: the old body now precedes the trajectory heading.
        if text.find(old_body_start) < heading_pos:
            return "already_normalized"
        return "missing_old_body"

    prefix = text[:heading_pos].rstrip()
    trajectory = text[heading_pos:body_pos].strip()
    old_body = text[body_pos:].strip()
    path.write_text(f"{prefix}\n\n{old_body}\n\n{trajectory}\n", encoding="utf-8", newline="\n")
    return "moved"


def main() -> int:
    counts: dict[str, int] = {}
    failures: list[str] = []
    for relative, (heading, old_body_start) in MOVES.items():
        status = normalize(ROOT / relative, heading, old_body_start)
        counts[status] = counts.get(status, 0) + 1
        if status.startswith("missing"):
            failures.append(f"{relative}: {status}")
    print({"counts": counts, "failures": failures})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
