from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.revision_evaluation import main_fixed


if __name__ == "__main__":
    print(main_fixed(ROOT))
