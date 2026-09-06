from __future__ import annotations

import argparse
from pathlib import Path

from mini_multimodal.ldm.geometry_evaluation import evaluate_attempt3_fixed
from mini_multimodal.ldm.revision_training import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--branch", choices=("control", "geometry", "geometry_lite"), required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    evaluate_attempt3_fixed(root, load_config(root), branch=args.branch)


if __name__ == "__main__":
    main()
