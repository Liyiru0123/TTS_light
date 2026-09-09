# -*- coding: utf-8 -*-
"""TinyImageLDM 单阶段入口：prepare→calibrate→freeze→overfit→train 一条命令（跳过 repro 探索阶段）。

注意：overfit 是进入正式训练前必须通过的小集过拟合门（非探索阶段），不能跳过；
定位包根用 parents[1]（本脚本位于 <包根>/scripts/ 下）；可选 --root 指定其他工作区。
"""
import argparse, sys, time, json, subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    ROOT = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    t = time.time()
    steps = [
        "src/mini_multimodal/tools/prepare_p2_w04_data.py",
        "src/mini_multimodal/tools/prepare_p2_w04_v2_data.py",
        "src/mini_multimodal/tools/prepare_p2_w04_eval_v2.py",
        "src/mini_multimodal/tools/calibrate_p2_w04_evaluator_v2.py",
        "src/mini_multimodal/tools/freeze_p2_w04_calibration.py",
    ]
    for s in steps:
        p = subprocess.run([sys.executable, str(ROOT / s)], cwd=ROOT)
        assert p.returncode == 0, s
    p = subprocess.run([sys.executable, str(ROOT / "src/mini_multimodal/tools/run_p2_w04_revision_training.py"), "overfit"], cwd=ROOT)
    assert p.returncode == 0, "overfit"
    p = subprocess.run([sys.executable, str(ROOT / "src/mini_multimodal/tools/run_p2_w04_revision_training.py"), "train"], cwd=ROOT)
    assert p.returncode == 0, "train"
    print(json.dumps({"model": "tiny_image_ldm", "elapsed_seconds": round(time.time() - t, 1)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
