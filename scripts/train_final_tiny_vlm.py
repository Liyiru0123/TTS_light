# -*- coding: utf-8 -*-
"""TinyVLM 单阶段入口：prepare→train→eval 一步（P2-W03 本就是单入口）。

注意：定位包根用 parents[1]（本脚本位于 <包根>/scripts/ 下）；
可选 --root 指定其他工作区（数据与结果所在目录）。
"""
import argparse, sys, time, json, subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    ROOT = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    t = time.time()
    entry = "src/mini_multimodal/tools/run_p2_w03_experiments.py"
    p = subprocess.run([sys.executable, str(ROOT / entry)], cwd=ROOT)
    assert p.returncode == 0, entry
    print(json.dumps({"model": "tiny_vlm", "elapsed_seconds": round(time.time() - t, 1)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
