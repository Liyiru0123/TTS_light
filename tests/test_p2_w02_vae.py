from __future__ import annotations

import json
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.vae import (  # noqa: E402
    TinyImageVAE,
    VAEImageDataset,
    VAELossWeights,
    build_p2_w02_data_spec,
    collate_vae_images,
    compute_vae_loss,
    verify_p2_w02_data,
)


CONFIG = json.loads(
    (ROOT / "configs" / "p2_w02_image_vae.json").read_text(encoding="utf-8")
)


class VAEDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = build_p2_w02_data_spec(
            ROOT,
            source_index=CONFIG["source_index"],
            source_images=CONFIG["source_images"],
            overfit_count=CONFIG["training"]["overfit_count"],
        )

    def test_data_vae_001_split_counts_and_ids_are_fixed(self) -> None:
        ids = self.spec["split_ids"]
        self.assertEqual({name: len(values) for name, values in ids.items()}, {
            "train": 1200,
            "validation": 220,
            "test": 300,
        })
        self.assertFalse(set(ids["train"]) & set(ids["validation"]))
        self.assertFalse(set(ids["train"]) & set(ids["test"]))
        self.assertEqual(len(self.spec["overfit_ids"]), 20)

    def test_data_vae_002_small_target_statistics_are_recorded(self) -> None:
        statistics = self.spec["statistics"]
        self.assertEqual(statistics["scene_count"], 1720)
        self.assertEqual(statistics["object_count"], 2544)
        self.assertEqual(statistics["bbox_width_px"]["median"], 13.0)
        self.assertAlmostEqual(statistics["scene_foreground_ratio"]["median"], 0.041259765625)

    def test_data_vae_003_mask_matches_visible_annotation_area(self) -> None:
        dataset = VAEImageDataset(
            ROOT,
            source_index=CONFIG["source_index"],
            source_images=CONFIG["source_images"],
            scene_ids=self.spec["overfit_ids"],
        )
        for index in range(len(dataset)):
            sample = dataset[index]
            expected = sum(
                obj["visible_area_pixels"] for obj in sample["metadata"]["objects"]
            )
            self.assertLessEqual(int(sample["foreground_mask"].sum()), expected)
            self.assertGreater(int(sample["foreground_mask"].sum()), 0)

    def test_data_vae_004_persisted_manifest_verifies(self) -> None:
        report = verify_p2_w02_data(ROOT, ROOT / CONFIG["output_data"])
        self.assertTrue(report["verified"])


class VAEUnitTests(unittest.TestCase):
    def _model(self) -> TinyImageVAE:
        return TinyImageVAE(**{
            key: CONFIG["model"][key]
            for key in ("base_channels", "latent_channels", "logvar_min", "logvar_max")
        })

    def test_unit_vae_001_tensor_shapes_and_ranges(self) -> None:
        model = self._model()
        image = torch.rand(2, 3, 64, 64)
        output = model(image, sample_posterior=False)
        self.assertEqual(tuple(output.mean.shape), (2, 4, 16, 16))
        self.assertEqual(tuple(output.logvar.shape), (2, 4, 16, 16))
        self.assertEqual(tuple(output.reconstruction.shape), tuple(image.shape))
        self.assertGreaterEqual(float(output.reconstruction.min().detach()), 0.0)
        self.assertLessEqual(float(output.reconstruction.max().detach()), 1.0)

    def test_unit_vae_002_reparameterization_is_reproducible_and_differentiable(self) -> None:
        model = self._model()
        image = torch.rand(2, 3, 64, 64)
        mean, logvar = model.encode(image)
        first_generator = torch.Generator().manual_seed(77)
        second_generator = torch.Generator().manual_seed(77)
        first = model.reparameterize(mean, logvar, generator=first_generator)
        second = model.reparameterize(mean, logvar, generator=second_generator)
        self.assertTrue(torch.equal(first, second))
        first.mean().backward()
        gradients = [parameter.grad for parameter in model.encoder.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_unit_vae_003_loss_components_are_finite_and_separable(self) -> None:
        model = self._model()
        image = torch.rand(2, 3, 64, 64)
        mask = torch.zeros(2, 1, 64, 64, dtype=torch.bool)
        mask[:, :, 20:40, 20:40] = True
        output = model(image)
        result = compute_vae_loss(output, image, mask, VAELossWeights())
        self.assertEqual(
            set(result),
            {"loss", "reconstruction_loss", "global_mae", "global_mse", "foreground_mae", "background_mae", "edge_mae", "kl"},
        )
        self.assertTrue(all(torch.isfinite(value) for value in result.values()))
        self.assertGreaterEqual(float(result["kl"].detach()), 0.0)

    def test_unit_vae_004_collate_preserves_metadata_without_padding(self) -> None:
        spec = build_p2_w02_data_spec(
            ROOT,
            source_index=CONFIG["source_index"],
            source_images=CONFIG["source_images"],
            overfit_count=CONFIG["training"]["overfit_count"],
        )
        dataset = VAEImageDataset(
            ROOT,
            source_index=CONFIG["source_index"],
            source_images=CONFIG["source_images"],
            scene_ids=spec["overfit_ids"][:2],
        )
        batch = collate_vae_images([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["image"].shape), (2, 3, 64, 64))
        self.assertEqual(tuple(batch["foreground_mask"].shape), (2, 1, 64, 64))
        self.assertEqual(len(batch["metadata"]), 2)


class VAEPersistedArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result_dir = ROOT / CONFIG["output_results"]
        self.latent_dir = ROOT / CONFIG["output_latents"]

    def test_artifact_vae_001_all_machine_gates_pass(self) -> None:
        summary = json.loads(
            (self.result_dir / "experiment_summary.json").read_text(encoding="utf-8")
        )
        self.assertTrue(summary["verified"])
        self.assertTrue(all(summary.values()))
        model = json.loads(
            (self.result_dir / "vae_metrics.json").read_text(encoding="utf-8")
        )
        self.assertTrue(model["verified"])
        self.assertTrue(all(model["checks"].values()))

    def test_artifact_vae_002_latent_cache_matches_best_checkpoint(self) -> None:
        report = json.loads(
            (self.result_dir / "latent_cache_report.json").read_text(encoding="utf-8")
        )
        checkpoint = self.result_dir / "checkpoints" / "best.pt"
        self.assertEqual(
            report["source_checkpoint_sha256"],
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
        with np.load(self.latent_dir / "posterior_means.npz", allow_pickle=False) as cache:
            self.assertEqual(tuple(cache["posterior_mean"].shape), (1720, 4, 16, 16))
            self.assertEqual(cache["posterior_mean"].dtype, np.float16)
            self.assertEqual(len(set(cache["scene_ids"].tolist())), 1720)
        self.assertTrue(report["verified"])

    def test_artifact_vae_003_required_checkpoints_reports_and_figures_exist(self) -> None:
        required = [
            self.result_dir / "checkpoints" / "best.pt",
            self.result_dir / "checkpoints" / "last.pt",
            self.result_dir / "training_log.jsonl",
            self.result_dir / "reconstruction_grid.png",
            self.result_dir / "latent_interpolation.png",
            self.result_dir / "training_curve.png",
            self.result_dir / "loss_ablation_report.json",
            self.result_dir / "compression_ablation_report.json",
            self.result_dir / "posterior_report.json",
            self.result_dir / "resource_benchmark.json",
        ]
        for path in required:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
