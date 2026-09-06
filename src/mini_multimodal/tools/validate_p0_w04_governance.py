"""Validate and persist the P0-W04 project-governance evidence."""

from __future__ import annotations

import json
from pathlib import Path

from mini_multimodal.governance import validate_project_governance


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    result = validate_project_governance(root)
    destination = root / "results" / "P0-W04_governance" / "governance_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "verified": result["verified"],
                "ledger_entry_count": result["ledger_entry_count"],
                "open_ledger_entry_count": result["open_ledger_entry_count"],
            },
            ensure_ascii=False,
        )
    )
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
