from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.revision_training import load_config, run_overfit, run_reproducibility_probe, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("overfit", "train", "repro"))
    args = parser.parse_args()
    config = load_config(ROOT)
    if args.phase == "overfit":
        result = run_overfit(ROOT, config)
    elif args.phase == "train":
        result = run_training(ROOT, config)
    else:
        result = run_reproducibility_probe(ROOT, config)
    print(result)


if __name__ == "__main__":
    main()
