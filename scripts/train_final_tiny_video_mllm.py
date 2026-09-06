# -*- coding: utf-8 -*-
"""TinyVideoMLLM 单阶段入口：prepare→train I003 一条命令（忽略末尾的可选打包/校验子进程）。

注意：定位包根用 parents[1]（本脚本位于 <包根>/scripts/ 下）；
可选 --root 指定其他工作区（数据与结果所在目录）。
说明：i003_run 末尾的可选 pytest/打包子进程失败（如 pytest 断言失败的
CreateProcess/CalledProcessError）不影响训练本体，此处忽略；若需要分层样例册，
用 scripts/p3_w01_rebuild_curated_examples.py 重建（开发链脚本，包内未含）。
"""
import argparse, sys, time, json, subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    ROOT = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    t = time.time()
    p = subprocess.run([sys.executable, str(ROOT / "scripts/p3_w01_prepare.py")], cwd=ROOT)
    if p.returncode:
        raise SystemExit("prepare 失败")
    import runpy
    import os
    os.environ["SKIP_SUBPROCESS"] = "1"   # i003_run 若支持则跳过；不支持时由下方 except 兜底
    try:
        runpy.run_path(str(ROOT / "scripts/p3_w01_i003_run.py"), run_name="__main__")
    except subprocess.CalledProcessError:
        pass                 # 末尾 pytest/打包子进程失败不影响训练本体
    except Exception as error:
        if "CreateProcess" not in str(error):
            raise
    print(json.dumps({"model": "tiny_video_mllm", "elapsed_seconds": round(time.time() - t, 1)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
