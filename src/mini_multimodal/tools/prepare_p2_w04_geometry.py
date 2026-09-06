from __future__ import annotations

import argparse
from pathlib import Path

from mini_multimodal.ldm.geometry_data import build_geometry_masks, calibrate_soft_foreground
from mini_multimodal.ldm.revision_training import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config = load_config(root)
    build_geometry_masks(root, config)
    report = calibrate_soft_foreground(root, config)
    if not report["verified_for_training"]:
        raise RuntimeError("attempt-3 geometry calibration failed")


if __name__ == "__main__":
    main()
