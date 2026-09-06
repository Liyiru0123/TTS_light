# -*- coding: utf-8 -*-
"""TinySpeechVAE + HiFi-GAN 单阶段入口：prepare_data→i002_train 一条命令。

注意：定位包根用 parents[1]（本脚本位于 <包根>/scripts/ 下）；
可选 --root 指定其他工作区（数据与结果所在目录）。
说明：i002_train 训练/评估/打包主体完整；末尾 A/B/C/D 对比样例册需要 I001 指标
与 PF001 预检结果（开发链副产物，包内不一定随附），缺失时跳过对应路（B/C），
不影响训练与推理包写出。
"""
import argparse, sys, time, json, subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    ROOT = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    t = time.time()
    for s in ["scripts/p3_w02_i002_prepare_data.py", "scripts/p3_w02_i002_train.py"]:
        p = subprocess.run([sys.executable, str(ROOT / s)], cwd=ROOT)
        assert p.returncode == 0, s
    print(json.dumps({"model": "tiny_speech_vae", "elapsed_seconds": round(time.time() - t, 1)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
