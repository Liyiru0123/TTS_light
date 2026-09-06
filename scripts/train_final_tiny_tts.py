# -*- coding: utf-8 -*-
"""TinyTTS 单阶段入口：prepare→train 一条命令（需先完成 P3-W02 SpeechVAE，即其编码链可用）。

注意：定位包根用 parents[1]（本脚本位于 <包根>/scripts/ 下）；
可选 --root 指定其他工作区（数据与结果所在目录）。
可选 --config 指定训练配置：
  - 默认 configs/p3_w03_tiny_tts.json（I002 基础版，无父权、可独立训练）；
  - 教学版 configs/p3_w03_i004_supported_rate.json 需要 I003/rate_robust 父权重（随开发链提供，见运行手册）。
"""
import argparse, sys, time, json, subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--config", default="configs/p3_w03_tiny_tts.json")
    args = ap.parse_args()
    ROOT = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    t = time.time()
    for s in ["scripts/p3_w03_prepare_data.py", "scripts/p3_w03_train.py"]:
        cmd = [sys.executable, str(ROOT / s)]
        if s.endswith("p3_w03_train.py"):
            cmd += ["--config", args.config]
        p = subprocess.run(cmd, cwd=ROOT)
        assert p.returncode == 0, s
    print(json.dumps({"model": "tiny_tts", "config": args.config,
                      "elapsed_seconds": round(time.time() - t, 1)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
