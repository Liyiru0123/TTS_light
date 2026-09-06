from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch

from mini_multimodal.speech_vae import (
    SpeechWindowDataset, TinySpeechVAE, collate_speech_windows,
    export_window_manifest, logmel_to_waveform, speech_vae_loss, stitch_windows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/p3_w02_speech_vae.json").read_text(encoding="utf-8"))


class P3W02SpeechVAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        export_window_manifest(ROOT, source_dir=CONFIG["source_speech"], output_dir=CONFIG["output_data"], window_frames=256, hop_frames=192, minimum_valid_frames=64)

    def test_data_spr_001_windows_and_source_splits(self):
        audit=json.loads((ROOT/CONFIG["output_data"] / "data_audit.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(audit["window_count"],CONFIG["thresholds"]["minimum_total_windows"])
        self.assertGreaterEqual(audit["split_window_counts"]["train"],CONFIG["thresholds"]["minimum_train_windows"])
        self.assertEqual(audit["source_split_intersections"],{"train_test":0,"train_validation":0,"validation_test":0})
        self.assertEqual(audit["coverage_failure_count"],0)
        self.assertGreaterEqual(audit["minimum_valid_frames"],64); self.assertLessEqual(audit["maximum_valid_frames"],256)

    def test_data_spr_003_dataset_shape_and_mask(self):
        dataset=SpeechWindowDataset(ROOT,data_dir=CONFIG["output_data"],splits="test",config=CONFIG); batch=collate_speech_windows([dataset[0],dataset[-1]])
        self.assertEqual(tuple(batch["window"].shape),(2,1,64,256)); self.assertEqual(tuple(batch["mask"].shape),(2,256)); self.assertTrue(batch["mask"].any())

    def test_unit_spr_001_forward_loss_mask_and_gradients(self):
        torch.manual_seed(1); model=TinySpeechVAE(**CONFIG["model"]); target=torch.randn(2,1,64,256).clamp(-1,1); mask=torch.ones(2,256,dtype=torch.bool); mask[:,-32:]=False
        out=model(target); loss,metrics=speech_vae_loss(out,target,mask,**CONFIG["loss"]); loss.backward()
        self.assertEqual(tuple(out.mean.shape),(2,4,16,64)); self.assertTrue(torch.isfinite(loss)); self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)); self.assertGreater(metrics["kl"],0)

    def test_unit_spr_002_reparameterization_is_seeded(self):
        mean=torch.zeros(1,4,2,3); logvar=torch.zeros_like(mean); a=TinySpeechVAE.reparameterize(mean,logvar,generator=torch.Generator().manual_seed(2)); b=TinySpeechVAE.reparameterize(mean,logvar,generator=torch.Generator().manual_seed(2)); c=TinySpeechVAE.reparameterize(mean,logvar,generator=torch.Generator().manual_seed(3)); self.assertTrue(torch.equal(a,b)); self.assertFalse(torch.equal(a,c))

    def test_unit_spr_003_stitching_is_exact_for_source_windows(self):
        source=np.arange(64*700,dtype=np.float32).reshape(64,700); starts=[0,192,384,444]; valid=[256,256,256,256]; values=[source[:,s:s+v] for s,v in zip(starts,valid)]; reconstructed,weight=stitch_windows(values,starts,valid,700); self.assertLessEqual(float(np.abs(reconstructed-source).max()),1e-6); self.assertGreaterEqual(float(weight.min()),1)

    def test_voc_spr_001_fixed_decoder_is_finite_and_deterministic(self):
        dataset=SpeechWindowDataset(ROOT,data_dir=CONFIG["output_data"],splits="test",config=CONFIG); row=dataset.rows[0]; source=np.load(ROOT/row["source_logmel_path"],allow_pickle=False).astype(np.float32)[:,:96]
        kwargs=dict(sample_rate=22050,n_fft=512,hop_length=256,fmin=40,fmax=8000,n_iter=2,momentum=.99); a=logmel_to_waveform(source,**kwargs); b=logmel_to_waveform(source,**kwargs); self.assertTrue(torch.equal(a,b)); self.assertTrue(torch.isfinite(a).all()); self.assertLessEqual(float(a.abs().max()),1.0)


if __name__ == "__main__": unittest.main()
