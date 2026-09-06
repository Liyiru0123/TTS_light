# -*- coding: utf-8 -*-
"""TinyT2V 单阶段入口：prepare→i008_train_evaluate 一条命令（需先完成 P2-W04 LDM + P2-W02 图像VAE）。

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
    for s in ["scripts/p3_w04_prepare_data.py", "scripts/p3_w04_i008_train_evaluate.py"]:
        p = subprocess.run([sys.executable, str(ROOT / s)], cwd=ROOT)
        assert p.returncode == 0, s
    print(json.dumps({"model": "tiny_t2v", "elapsed_seconds": round(time.time() - t, 1)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
