from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_multimodal.ldm.i005_experiment import (
    compare_formal_branches,
    evaluate_formal_branch,
    run_clean_refiner_overfit,
    train_formal_branch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled P2-W04 I005 stages")
    parser.add_argument("command", choices=("overfit", "train", "evaluate", "compare"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--branch", choices=("continuation", "clean_refiner"))
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    if args.command == "overfit":
        result = run_clean_refiner_overfit(root)
    elif args.command == "train":
        if not args.branch:
            raise SystemExit("train requires --branch")
        result = train_formal_branch(root, args.branch)
    elif args.command == "evaluate":
        if not args.branch:
            raise SystemExit("evaluate requires --branch")
        result = evaluate_formal_branch(root, args.branch)
    else:
        result = compare_formal_branches(root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
