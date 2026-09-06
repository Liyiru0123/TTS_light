from __future__ import annotations
import hashlib,json,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));RESULT=ROOT/"results/P2-W05_tiny_asr";BRANCH=RESULT/"iterations/I002/branches/robust_finetune"
from mini_multimodal.asr import CharacterCTCTokenizer,TinyASR  # noqa:E402
class P2W05I002ArtifactTests(unittest.TestCase):
    def test_art_asr_i002_001_acceptance_all_passes(self):
        report=json.loads((BRANCH/"metrics/acceptance_report.json").read_text(encoding="utf-8"));self.assertTrue(report["all_blocking_acceptance_items_passed"]);self.assertFalse(report["failed_items"]);self.assertTrue(all(item["passed"] for item in report["items"].values()))
    def test_art_asr_i002_002_checkpoint_and_package_strict_load(self):
        manifest=json.loads((BRANCH/"checkpoints/checkpoint_manifest.json").read_text(encoding="utf-8"))
        for name,meta in manifest.items():self.assertEqual(hashlib.sha256((BRANCH/"checkpoints"/name).read_bytes()).hexdigest(),meta["sha256"])
        package=torch.load(BRANCH/"inference_package.pt",map_location="cpu",weights_only=False);tokenizer=CharacterCTCTokenizer(package["vocabulary"]);model=TinyASR(vocabulary_size=len(tokenizer),**package["model_config"]);model.load_state_dict(package["model_state"],strict=True)
    def test_ped_asr_i002_001_paired_showcase_has_correct_and_error(self):
        data=json.loads((RESULT/"capability_examples.json").read_text(encoding="utf-8"));examples=data["examples"];self.assertEqual(data["example_count"],12);self.assertTrue(any(x["is_fully_correct"] for x in examples));self.assertTrue(any(not x["is_fully_correct"] for x in examples));self.assertEqual(sum(x["source_record_id"].endswith("::clean") for x in examples),6);self.assertEqual(sum(x["source_record_id"].endswith("::noise15") for x in examples),6)
        for item in examples:
            for path in item["input_artifacts"]+item["display_artifacts"]:self.assertTrue((ROOT/path).is_file())
    def test_reg_asr_i002_001_candidate_and_release_separate(self):
        candidate=json.loads((RESULT/"current_candidate.json").read_text(encoding="utf-8"))["current_candidate"];release=json.loads((RESULT/"official_release.json").read_text(encoding="utf-8"))["official_release"];self.assertEqual((candidate["iteration_id"],candidate["branch_id"]),("I002","robust_finetune"));self.assertIsNone(release)
    def test_reg_asr_i002_002_registry_matches_iteration_manifests(self):
        registry=json.loads((RESULT/"iteration_registry.json").read_text(encoding="utf-8"))
        for row in registry["iterations"]:
            manifest=json.loads((RESULT/row["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(row["execution_state"],manifest["execution_state"])
            self.assertEqual(row["selection_decision"],manifest["selection_decision"])
            self.assertEqual(row["all_blocking_acceptance_items_passed"],manifest["all_blocking_acceptance_items_passed"])
    def test_reg_asr_i002_003_work_package_accepted_without_formal_release(self):
        contract=json.loads((RESULT/"artifact_contract.json").read_text(encoding="utf-8"))
        decision=(ROOT/contract["acceptance_decision"]).read_text(encoding="utf-8")
        release=json.loads((RESULT/"official_release.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["status"],"已验收")
        self.assertIn("状态：已验收",decision)
        self.assertIn("数据增广",decision)
        self.assertIn("真实环境噪声",decision)
        self.assertIsNone(release["official_release"])
    def test_reg_asr_i002_004_canonical_summary_contract_is_current_and_complete(self):
        contract=json.loads((RESULT/"artifact_contract.json").read_text(encoding="utf-8"));summary=(ROOT/contract["summary_document"]).read_text(encoding="utf-8");candidate=json.loads((RESULT/"current_candidate.json").read_text(encoding="utf-8"))["current_candidate"]
        self.assertEqual(contract["schema_version"],3);self.assertEqual(contract["summary_document"],"docs/P2-W05_总结报告.md");self.assertEqual(contract["summary_contract"]["current_candidate"],{"iteration_id":candidate["iteration_id"],"branch_id":candidate["branch_id"]})
        for heading in ("## 7. 直观能力样例","## 9. 与当前代表性系统的差距","## 12. 主要证据路径"):self.assertIn(heading,summary)
        for relative in contract["historical_summary_documents"]+[contract["iteration_summary_document"]]:self.assertTrue((ROOT/relative).is_file())
if __name__=="__main__":unittest.main()
