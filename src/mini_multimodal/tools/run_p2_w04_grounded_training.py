from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.grounded_training import run_grounded_overfit, run_grounded_training
from mini_multimodal.ldm.revision_training import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("overfit", "train"))
    args = parser.parse_args()
    config = load_config(ROOT)
    print(run_grounded_overfit(ROOT, config) if args.phase == "overfit" else run_grounded_training(ROOT, config))
