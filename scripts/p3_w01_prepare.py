from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.video_mllm import (  # noqa: E402
    export_temporal_challenge,
    verify_temporal_challenge,
)


def main() -> None:
    config = json.loads((ROOT / "configs/p3_w01_tiny_video_mllm.json").read_text(encoding="utf-8"))
    manifest = export_temporal_challenge(ROOT, config)
    verified = verify_temporal_challenge(ROOT, config["data"]["output_dir"])
    print(json.dumps({"manifest": manifest["statistics"], "verification": verified}, ensure_ascii=False))


if __name__ == "__main__":
    main()
