"""Generate persistent data and result artifacts for work package P1-W01."""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path

import torch
import torchaudio
import torchvision

from mini_multimodal.world.artifacts import (
    benchmark_sampler,
    export_scene_fixture,
    verify_scene_fixture,
)
from mini_multimodal.world.sampling import SamplerConfig


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name] = dist.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "fixtures" / "world_core_v1"
    results_dir = project_root / "results" / "P1-W01_world_core"
    results_dir.mkdir(parents=True, exist_ok=True)

    config = SamplerConfig(min_objects=1, max_objects=3)
    manifest = export_scene_fixture(data_dir, count=200, base_seed=2026, config=config)
    verification = verify_scene_fixture(data_dir)
    benchmark = benchmark_sampler(count=5000, base_seed=2026, config=config)

    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    matrix_product = tensor @ tensor.T
    waveform = torch.randn(1, 1600, generator=torch.Generator().manual_seed(2026))
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=8000, n_fft=256, hop_length=80, n_mels=20
    )(waveform)

    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "tensor_smoke_shape": list(matrix_product.shape),
        "mel_smoke_shape": list(mel.shape),
        "packages": _installed_packages(),
    }

    _write_json(results_dir / "fixture_verification.json", verification)
    _write_json(results_dir / "sampling_benchmark.json", benchmark)
    _write_json(results_dir / "environment.json", environment)

    lock_lines = [f"{name}=={version}" for name, version in environment["packages"].items()]
    (project_root / "requirements-lock.txt").write_text(
        "\n".join(lock_lines) + "\n", encoding="utf-8", newline="\n"
    )

    print(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "results_dir": str(results_dir),
                "manifest_sha256": manifest["scenes_sha256"],
                "verified": verification["verified"],
                "scene_count": verification["scene_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
