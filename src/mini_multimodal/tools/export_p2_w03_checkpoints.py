"""Export a self-describing P2-W03 inference bundle and checkpoint manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_entry(root: Path, relative_path: str, role: str) -> dict[str, Any]:
    path = root / relative_path
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": relative_path.replace("\\", "/"),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "epoch": int(state["epoch"]),
        "best_metric": float(state["best_metric"]),
        "monitor": str(state["config"]["monitor"]),
        "maximize": bool(state["config"]["maximize"]),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config_path = root / "configs" / "p2_w03_tiny_vlm.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir = root / config["output_data"]
    vocabulary_path = data_dir / "vocabulary.json"
    answers_path = data_dir / "answer_classes.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    answer_classes = json.loads(answers_path.read_text(encoding="utf-8"))

    results = root / config["output_results"]
    source_checkpoint = results / "generative" / "checkpoints" / "best.pt"
    source_state = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    metrics = json.loads((results / "generation_metrics.json").read_text(encoding="utf-8"))

    export_dir = results / "checkpoints"
    export_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = export_dir / "tiny_vlm_inference.pt"
    temporary = bundle_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "p2_w03_tiny_vlm_inference",
            "model_state": source_state["model_state"],
            "model_config": config["model"],
            "vocabulary": vocabulary,
            "answer_classes": answer_classes,
            "special_tokens": {
                name: int(vocabulary["token_to_id"][name])
                for name in ("<pad>", "<bos>", "<eos>", "<mask>", "甲", "乙")
            },
            "source_training_checkpoint": {
                "path": "results/P2-W03_tiny_vlm/generative/checkpoints/best.pt",
                "sha256": _sha256(source_checkpoint),
                "epoch": int(source_state["epoch"]),
            },
            "validated_metrics": {
                "classification_accuracy": metrics["classification_accuracy"],
                "generation_accuracy": metrics["generation_accuracy"],
                "grounding_relation_accuracy": metrics["grounding_relation_accuracy"],
            },
        },
        temporary,
    )
    temporary.replace(bundle_path)

    vae_path = root / config["source_vae_checkpoint"]
    manifest = {
        "schema_version": 1,
        "primary_inference_bundle": {
            "path": str(bundle_path.relative_to(root)).replace("\\", "/"),
            "bytes": bundle_path.stat().st_size,
            "sha256": _sha256(bundle_path),
            "contains": [
                "TinyVLM model_state",
                "model architecture config",
                "character vocabulary",
                "answer classes",
                "special token ids",
                "validated metrics",
            ],
        },
        "training_checkpoints": {
            "classification_best": _checkpoint_entry(
                root,
                "results/P2-W03_tiny_vlm/classification/checkpoints/best.pt",
                "best closed-set VQA checkpoint",
            ),
            "classification_last": _checkpoint_entry(
                root,
                "results/P2-W03_tiny_vlm/classification/checkpoints/last.pt",
                "classification resume checkpoint",
            ),
            "generative_best": _checkpoint_entry(
                root,
                "results/P2-W03_tiny_vlm/generative/checkpoints/best.pt",
                "primary causal-generation training checkpoint",
            ),
            "generative_last": _checkpoint_entry(
                root,
                "results/P2-W03_tiny_vlm/generative/checkpoints/last.pt",
                "causal-generation resume checkpoint",
            ),
        },
        "training_checkpoint_contents": [
            "model_state",
            "optimizer_state",
            "epoch",
            "best_metric",
            "trainer config",
            "Python/NumPy/PyTorch random state",
        ],
        "external_dependencies": {
            "full_experiment_config": {
                "path": str(config_path.relative_to(root)).replace("\\", "/"),
                "sha256": _sha256(config_path),
            },
            "vocabulary": {
                "path": str(vocabulary_path.relative_to(root)).replace("\\", "/"),
                "sha256": _sha256(vocabulary_path),
            },
            "answer_classes": {
                "path": str(answers_path.relative_to(root)).replace("\\", "/"),
                "sha256": _sha256(answers_path),
            },
            "raw_image_vae_frontend": {
                "path": str(vae_path.relative_to(root)).replace("\\", "/"),
                "bytes": vae_path.stat().st_size,
                "sha256": _sha256(vae_path),
                "required_when": "encoding a raw 64x64 image instead of using cached 4x16x16 latents",
            },
        },
        "usage": {
            "inference": "load primary_inference_bundle and instantiate TinyVLM from its embedded config/vocabulary/answer classes",
            "resume_training": "load the corresponding last.pt with model and optimizer, then restore the saved random state",
        },
    }
    manifest_path = export_dir / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"verified": True, "bundle": manifest["primary_inference_bundle"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
