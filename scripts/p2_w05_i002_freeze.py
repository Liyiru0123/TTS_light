"""Refresh the pre-training I002 source/config snapshot after implementation tests pass."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mini_multimodal.iteration_management import build_source_snapshot, sha256_file


ROOT = Path(__file__).resolve().parents[1]
ITERATION = ROOT / "results/P2-W05_tiny_asr/iterations/I002"


def main() -> None:
    source = build_source_snapshot(ROOT, ITERATION / "source_snapshot.zip", ITERATION / "source_manifest.json")
    manifest_path = ITERATION / "iteration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots = []
    for relative in ("configs/p2_w05_tiny_asr.json", "configs/p2_w05_i002.json"):
        origin = ROOT / relative; destination = ITERATION / "configs" / origin.name; shutil.copy2(origin, destination)
        snapshots.append({"source_path":relative,"snapshot_path":origin.name,"bytes":destination.stat().st_size,"sha256":sha256_file(destination)})
    manifest["config_snapshots"] = snapshots
    manifest["source_reproducibility"] = {"state":"snapshot_available","snapshot_path":"results/P2-W05_tiny_asr/iterations/I002/source_snapshot.zip","manifest_path":"results/P2-W05_tiny_asr/iterations/I002/source_manifest.json","snapshot_sha256":source["snapshot_sha256"]}
    manifest.setdefault("status_history", []).append({"state":"running","updated_at":"2026-08-17T00:20:00+08:00","record":"implementation and preflight tests frozen immediately before training"})
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"snapshot_sha256":source["snapshot_sha256"],"file_count":source["file_count"],"config_snapshots":snapshots},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
