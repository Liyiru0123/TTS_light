from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.ldm.grounded_evaluation import main_grounded


if __name__ == "__main__":
    print(main_grounded(ROOT))
