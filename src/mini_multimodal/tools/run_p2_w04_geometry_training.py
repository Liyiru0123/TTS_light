from __future__ import annotations

import argparse
from pathlib import Path

from mini_multimodal.ldm.geometry_training import run_attempt3_finetuning, run_geometry_overfit
from mini_multimodal.ldm.revision_training import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--stage",
        choices=("overfit", "overfit_lite", "control", "geometry", "geometry_lite"),
        required=True,
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config = load_config(root)
    if args.stage == "overfit":
        run_geometry_overfit(root, config)
    elif args.stage == "overfit_lite":
        run_geometry_overfit(root, config, geometry_variant="geometry_lite")
    else:
        run_attempt3_finetuning(root, config, branch=args.stage)


if __name__ == "__main__":
    main()
