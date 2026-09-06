from pathlib import Path
import json,re
import numpy as np,torch
from mini_multimodal.speech_vae import TinySpeechVAE
ROOT=Path(__file__).resolve().parents[1];BRANCH=ROOT/"results/P3-W02_speech_vae/iterations/I001/branches/spectral_vae"
def test_art_spr_001_acceptance_and_thresholds():
    a=json.loads((BRANCH/"metrics/acceptance_report.json").read_text(encoding="utf-8"));assert a["all_blocking_acceptance_items_passed"] and not a["failed_items"]
    assert len(a["items"])==16 and all(item["passed"] for item in a["items"].values())
def test_art_spr_002_inference_package_strict_load():
    p=torch.load(BRANCH/"inference_package.pt",map_location="cpu",weights_only=False);m=TinySpeechVAE(**p["model_config"]);m.load_state_dict(p["model_state"],strict=True);assert sum(x.numel() for x in m.parameters())==187177
def test_art_spr_003_latent_cache_shape_and_ids():
    with np.load(ROOT/"data/latents/p3_w02_v1/posterior_means.npz",allow_pickle=False) as c:assert c["posterior_mean"].shape==(8648,4,16,64) and len(c["window_id"])==8648
def test_art_spr_004_capability_examples_and_links():
    p=BRANCH/"examples/capability_examples.json";j=json.loads(p.read_text(encoding="utf-8"));assert j["example_count"]==8;assert sum(x["selection_type"]=="boundary" for x in j["examples"])>=3
    md=(BRANCH/"examples/capability_examples.md").read_text(encoding="utf-8");links=re.findall(r"\((media/[^)]+)\)",md);assert len(links)==24 and all((BRANCH/"examples"/x).is_file() for x in links)
def test_doc_spr_001_summary_contract():
    text=(ROOT/"docs/P3-W02_总结报告.md").read_text(encoding="utf-8");
    for heading in ("工作包结论与状态","任务、数据与评价范围","模型结构与输入输出","核心量化结果","直观能力样例","检查点、推理、资源与交付物","与当前代表性系统的差距","主要证据路径"):assert heading in text
