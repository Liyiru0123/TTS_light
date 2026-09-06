"""Patch is_fully_correct in generation_predictions.jsonl to include position (v2.0).

The main experiment run was started before the is_fully_correct definition was
updated to require joint_position_accuracy. This script recomputes
is_fully_correct from the semantic_match data already present in each row.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    result_dir = root / "results" / "P2-W04_tiny_image_ldm"
    predictions_path = result_dir / "generation_predictions.jsonl"
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    updated = 0
    for row in rows:
        match = row["semantic_match"]
        probe = row["probe"]
        new_correct = bool(
            probe["foreground_present"]
            and match["count_correct"]
            and match.get("joint_position_accuracy", match.get("joint_accuracy", 0.0)) == 1.0
        )
        if row.get("is_fully_correct") != new_correct:
            row["is_fully_correct"] = new_correct
            updated += 1
    predictions_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    print(json.dumps({"updated": updated, "total": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()