from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_multimodal.training import (  # noqa: E402
    ClassificationTask,
    ManifestImageDataset,
    TinyConvClassifier,
    Trainer,
    TrainerConfig,
    build_p2_w01_data_spec,
    cue_only_accuracy,
    learn_cue_mapping,
    load_checkpoint,
    seeded_generator,
    set_deterministic_seed,
)


INDEX = ROOT / "data" / "multimodal" / "p1_w07_v1"


class SharedTrainingFrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.baseline = build_p2_w01_data_spec(INDEX)

    def test_unit_step_001_shape_loss_and_gradients_are_finite(self) -> None:
        set_deterministic_seed(101, cpu_threads=2)
        dataset = ManifestImageDataset(ROOT, self.baseline["overfit"][:5])
        batch = next(iter(DataLoader(dataset, batch_size=5)))
        model = TinyConvClassifier()
        output = ClassificationTask().step(model, batch, training=True)
        output.loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertEqual(output.weight, 5)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_unit_ckpt_001_checkpoint_restores_model_optimizer_and_epoch(self) -> None:
        set_deterministic_seed(202, cpu_threads=2)
        dataset = ManifestImageDataset(ROOT, self.baseline["overfit"])
        loader = DataLoader(dataset, batch_size=20, shuffle=False)
        with tempfile.TemporaryDirectory() as temporary:
            model = TinyConvClassifier()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                task=ClassificationTask(),
                config=TrainerConfig(epochs=1, gradient_clip_norm=5.0),
                output_dir=temporary,
            )
            trainer.fit(loader, loader)
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            restored = TinyConvClassifier()
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
            state = load_checkpoint(
                Path(temporary) / "checkpoints" / "last.pt",
                model=restored,
                optimizer=restored_optimizer,
                restore_rng=False,
            )
            self.assertEqual(state["epoch"], 0)
            self.assertEqual(set(optimizer.state_dict()), set(restored_optimizer.state_dict()))
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]))

    def test_int_train_001_resume_appends_log_and_continues_epoch(self) -> None:
        set_deterministic_seed(303, cpu_threads=2)
        dataset = ManifestImageDataset(ROOT, self.baseline["overfit"])
        loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            generator=seeded_generator(303),
        )
        with tempfile.TemporaryDirectory() as temporary:
            model = TinyConvClassifier()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            first = Trainer(
                model=model,
                optimizer=optimizer,
                task=ClassificationTask(),
                config=TrainerConfig(epochs=1, gradient_clip_norm=5.0),
                output_dir=temporary,
            )
            first.fit(loader, loader)
            resumed_model = TinyConvClassifier()
            resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
            second = Trainer(
                model=resumed_model,
                optimizer=resumed_optimizer,
                task=ClassificationTask(),
                config=TrainerConfig(epochs=2, gradient_clip_norm=5.0),
                output_dir=temporary,
            )
            history = second.fit(
                loader,
                loader,
                resume_from=Path(temporary) / "checkpoints" / "last.pt",
            )
            rows = [
                json.loads(line)
                for line in (Path(temporary) / "training_log.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["epoch"] for row in rows], [0, 1])
            self.assertEqual(history[0]["epoch"], 1)

    def test_ped_shortcut_001_cue_baseline_breaks_under_counterfactual(self) -> None:
        groups = self.baseline["groups"]
        mapping = learn_cue_mapping(groups["train"])
        iid = cue_only_accuracy(
            [entry for entry in groups["validation"] if entry["variant"] == "shortcut_iid"],
            mapping,
        )
        counterfactual = cue_only_accuracy(
            [entry for entry in groups["test"] if entry["variant"] == "counterfactual"],
            mapping,
        )
        self.assertGreaterEqual(iid["accuracy"], 0.85)
        self.assertLessEqual(counterfactual["accuracy"], 0.10)

    def test_comp_overfit_001_persisted_small_data_gate_passes(self) -> None:
        report = json.loads(
            (ROOT / "results" / "P2-W01_training_framework" / "overfit_report.json")
            .read_text(encoding="utf-8")
        )
        self.assertTrue(report["verified"])
        self.assertGreaterEqual(report["maximum_accuracy"], report["threshold"])
        self.assertLess(report["final"]["loss"], report["initial"]["loss"])

    def test_model_base_001_persisted_group_metrics_pass(self) -> None:
        result_dir = ROOT / "results" / "P2-W01_training_framework"
        metrics = json.loads((result_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
        summary = json.loads((result_dir / "experiment_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(metrics["verified"])
        self.assertTrue(summary["verified"])
        self.assertTrue((result_dir / "checkpoints" / "best.pt").is_file())
        self.assertTrue((result_dir / "checkpoints" / "last.pt").is_file())
        self.assertTrue((result_dir / "training_curve.png").is_file())

    def test_res_repro_001_persisted_repro_resume_and_resource_gates_pass(self) -> None:
        result_dir = ROOT / "results" / "P2-W01_training_framework"
        for name in (
            "reproducibility_report.json",
            "checkpoint_resume_report.json",
            "resource_benchmark.json",
        ):
            with self.subTest(name=name):
                report = json.loads((result_dir / name).read_text(encoding="utf-8"))
                self.assertTrue(report["verified"])


if __name__ == "__main__":
    unittest.main()
