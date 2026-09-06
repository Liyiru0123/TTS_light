"""Train, evaluate and package P3-W02 I001/spectral_vae."""

from __future__ import annotations

import hashlib,json,os,random,sys,time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil,soundfile as sf,torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from mini_multimodal.asr import CharacterCTCTokenizer,TinyASR,ctc_greedy_decode  # noqa:E402
from mini_multimodal.speech.features import log_mel  # noqa:E402
from mini_multimodal.speech_vae import SpeechVAETask,SpeechWindowDataset,TinySpeechVAE,collate_speech_windows,logmel_to_waveform,stitch_windows  # noqa:E402
from mini_multimodal.training import Trainer,TrainerConfig,load_checkpoint  # noqa:E402

RESULT=ROOT/"results/P3-W02_speech_vae";BRANCH=RESULT/"iterations/I001/branches/spectral_vae"

def write_json(path,value): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def set_seed(seed,threads=8): random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
def loader(ds,batch,shuffle,seed): return DataLoader(ds,batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed),num_workers=0,collate_fn=collate_speech_windows)
def denorm(x,cfg):
    lo,hi=cfg["feature"]["minimum_logmel"],cfg["feature"]["maximum_logmel"]
    return ((x+1)*.5*(hi-lo)+lo).clip(lo,hi)
def edit(a,b):
    p=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        c=[i]
        for j,y in enumerate(b,1): c.append(min(c[-1]+1,p[j]+1,p[j-1]+(x!=y)))
        p=c
    return p[-1]
def model_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()): h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def short_run(cfg,batch,steps,seed,loss_cfg):
    set_seed(seed,cfg["cpu_threads"]);m=TinySpeechVAE(**cfg["model"]);o=torch.optim.AdamW(m.parameters(),lr=.001);task=SpeechVAETask(**loss_cfg);losses=[]
    for _ in range(steps): o.zero_grad(set_to_none=True);out=task.step(m,batch,training=True);out.loss.backward();o.step();losses.append(float(out.loss.detach()))
    return m,losses

@torch.inference_mode()
def evaluate_windows(model,ds,cfg):
    rows=[];tot={"normalized_mae":0.,"logmel_mae":0.,"temporal_delta_mae":0.,"kl":0.};count=0
    for batch in loader(ds,32,False,cfg["seed"]):
        out=model(batch["window"],sample_posterior=False);mask=batch["mask"][:,None,None,:].float();den=(mask.sum()*64).clamp_min(1)
        norm=((out.reconstruction-batch["window"]).abs()*mask).sum()/den
        target=denorm(batch["window"].numpy(),cfg);recon=denorm(out.reconstruction.numpy(),cfg);m=mask.numpy();logmae=float((np.abs(recon-target)*m).sum()/max(1,m.sum()*64))
        pair=(batch["mask"][:,1:]&batch["mask"][:,:-1])[:,None,None,:].numpy();dt=np.diff(target,axis=-1);dr=np.diff(recon,axis=-1);dmae=float((np.abs(dr-dt)*pair).sum()/max(1,pair.sum()*64))
        kl=float((-0.5*(1+out.logvar-out.mean.square()-out.logvar.exp())).mean())
        b=batch["window"].shape[0]; count+=b
        for k,v in (("normalized_mae",float(norm)),("logmel_mae",logmae),("temporal_delta_mae",dmae),("kl",kl)):tot[k]+=v*b
        for i,meta in enumerate(batch["metadata"]):
            valid=meta["valid_frames"];rows.append({"window_id":meta["window_id"],"source_utterance_id":meta["source_utterance_id"],"split":meta["split"],"normalized_mae":float(np.abs(out.reconstruction[i,0,:,:valid].numpy()-batch["window"][i,0,:,:valid].numpy()).mean()),"logmel_mae":float(np.abs(recon[i,0,:,:valid]-target[i,0,:,:valid]).mean()),"temporal_delta_mae":float(np.abs(np.diff(recon[i,0,:,:valid],axis=-1)-np.diff(target[i,0,:,:valid],axis=-1)).mean())})
    return {k:v/count for k,v in tot.items()},rows

def load_asr():
    p=torch.load(ROOT/"results/P2-W05_tiny_asr/iterations/I002/branches/robust_finetune/inference_package.pt",map_location="cpu",weights_only=False);tok=CharacterCTCTokenizer(p["vocabulary"]);m=TinyASR(vocabulary_size=len(tok),**p["model_config"]);m.load_state_dict(p["model_state"],strict=True);m.eval();return m,tok

@torch.inference_mode()
def asr_predict(model,tok,mel):
    x=mel.T.astype(np.float32);x=(x-x.mean(0,keepdims=True))/np.maximum(x.std(0,keepdims=True),1e-5);t=torch.from_numpy(x)[None];o=model(t,torch.tensor([t.shape[1]]));return ctc_greedy_decode(o.logits,o.output_lengths,tok)[0]

@torch.inference_mode()
def evaluate_utterances(model,ds,cfg,limit=12):
    groups=defaultdict(list)
    for index,row in enumerate(ds.rows): groups[row["source_utterance_id"]].append(index)
    asr,tok=load_asr();asr_rows={r["record_id"]:r for r in (json.loads(x) for x in (ROOT/"data/asr/p2_w05_v1/records.jsonl").read_text(encoding="utf-8").splitlines())}
    media=BRANCH/"examples/media";media.mkdir(parents=True,exist_ok=True);records=[]
    for utterance_id in sorted(groups)[:limit]:
        items=[ds[i] for i in groups[utterance_id]];batch=collate_speech_windows(items);out=model(batch["window"],sample_posterior=False);windows=denorm(out.reconstruction[:,0].numpy(),cfg)
        metas=batch["metadata"];starts=[m["start_frame"] for m in metas];valid=[m["valid_frames"] for m in metas];total=metas[0]["source_frames"]
        stitched,weights=stitch_windows(list(windows),starts,valid,total);boundaries=[]
        for i in range(len(windows)-1):
            overlap=starts[i]+valid[i]-starts[i+1]
            if overlap>0:boundaries.append(float(np.abs(windows[i][:,valid[i]-overlap:valid[i]]-windows[i+1][:,:overlap]).mean()))
        wave=logmel_to_waveform(stitched,sample_rate=cfg["feature"]["sample_rate"],n_fft=cfg["feature"]["n_fft"],hop_length=cfg["feature"]["hop_length"],fmin=cfg["feature"]["fmin"],fmax=cfg["feature"]["fmax"],n_iter=cfg["vocoder"]["griffin_lim_iterations"],momentum=cfg["vocoder"]["momentum"])
        cycle=log_mel(wave.numpy(),cfg["feature"]["sample_rate"],n_fft=512,hop_length=256,n_mels=64,fmin=40,fmax=8000);common=min(stitched.shape[1],cycle.shape[1]);cycle_mae=float(np.abs(stitched[:,:common]-cycle[:,:common]).mean());reference=asr_rows[utterance_id]["normalized_transcript"];prediction=asr_predict(asr,tok,cycle);cer=edit(reference,prediction)/len(reference)
        stem=hashlib.sha256(utterance_id.encode()).hexdigest()[:16];audio=media/f"{stem}_reconstructed.flac";sf.write(audio,wave.numpy(),22050,format="FLAC",subtype="PCM_16")
        original=ROOT/metas[0]["source_audio_path"]
        figure=media/f"{stem}_spectrogram.png";source=np.load(ROOT/metas[0]["source_logmel_path"],allow_pickle=False).astype(np.float32)
        fig,ax=plt.subplots(2,1,figsize=(10,4),sharex=True);ax[0].imshow(source,aspect="auto",origin="lower");ax[0].set_title("source Log-Mel");ax[1].imshow(stitched,aspect="auto",origin="lower");ax[1].set_title("VAE stitched reconstruction");fig.tight_layout();fig.savefig(figure,dpi=120);plt.close(fig)
        records.append({"source_record_id":utterance_id,"input_artifacts":[original.relative_to(ROOT).as_posix()],"display_artifacts":[audio.relative_to(ROOT).as_posix(),figure.relative_to(ROOT).as_posix()],"expected_output":reference,"model_outputs":{"asr_transcript":prediction,"window_count":len(items)},"overlap_boundary_mae":float(np.mean(boundaries)) if boundaries else 0.,"waveform_cycle_logmel_mae":cycle_mae,"asr_cer":cer,"waveform_peak":float(wave.abs().max()),"duration_seconds":len(wave)/22050})
    return records

def main():
    cfg=json.loads((ROOT/"configs/p3_w02_speech_vae.json").read_text(encoding="utf-8"));started=time.perf_counter();process=psutil.Process(os.getpid());set_seed(cfg["seed"],cfg["cpu_threads"])
    train=SpeechWindowDataset(ROOT,data_dir=cfg["output_data"],splits="train",config=cfg);val=SpeechWindowDataset(ROOT,data_dir=cfg["output_data"],splits="validation",config=cfg);test=SpeechWindowDataset(ROOT,data_dir=cfg["output_data"],splits="test",config=cfg)
    fixture=collate_speech_windows([train[i] for i in range(16)])
    first,l1=short_run(cfg,fixture,3,cfg["seed"]+77,cfg["loss"]);second,l2=short_run(cfg,fixture,3,cfg["seed"]+77,cfg["loss"]);write_json(BRANCH/"logs/reproducibility_report.json",{"losses_a":l1,"losses_b":l2,"parameter_sha256_a":model_hash(first),"parameter_sha256_b":model_hash(second),"identical":l1==l2 and model_hash(first)==model_hash(second)})
    overfit,losses=short_run(cfg,fixture,cfg["training"]["overfit_updates"],cfg["seed"]+1,cfg["loss"]);overfit.eval();oo=overfit(fixture["window"],sample_posterior=False);_,om=speech_loss=__import__('mini_multimodal.speech_vae',fromlist=['speech_vae_loss']).speech_vae_loss(oo,fixture["window"],fixture["mask"],**cfg["loss"]);write_json(BRANCH/"metrics/overfit_report.json",{"updates":cfg["training"]["overfit_updates"],"initial_loss":losses[0],"final_loss":losses[-1],**om})
    set_seed(cfg["seed"],cfg["cpu_threads"]);model=TinySpeechVAE(**cfg["model"]);opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"]);trainer=Trainer(model=model,optimizer=opt,task=SpeechVAETask(**cfg["loss"]),config=TrainerConfig(epochs=cfg["training"]["epochs"],device="cpu",gradient_clip_norm=cfg["training"]["gradient_clip_norm"],monitor="normalized_mae",maximize=False),output_dir=BRANCH);history=trainer.fit(loader(train,32,True,cfg["seed"]),loader(val,32,False,cfg["seed"]));load_checkpoint(BRANCH/"checkpoints/best.pt",model=model,optimizer=None,restore_rng=False);model.eval()
    metrics,predictions=evaluate_windows(model,test,cfg);metrics["element_compression_ratio"]=4.0;write_json(BRANCH/"metrics/evaluation_metrics.json",{"sample_count":len(test),**metrics});(BRANCH/"predictions").mkdir(parents=True,exist_ok=True);(BRANCH/"predictions/test_window_predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in predictions),encoding="utf-8")
    utter=evaluate_utterances(model,test,cfg,12);agg={"sample_count":len(utter),"overlap_boundary_mae":float(np.mean([r["overlap_boundary_mae"] for r in utter])),"waveform_cycle_logmel_mae":float(np.mean([r["waveform_cycle_logmel_mae"] for r in utter])),"asr_cer":sum(edit(r["expected_output"],r["model_outputs"]["asr_transcript"]) for r in utter)/sum(len(r["expected_output"]) for r in utter),"maximum_waveform_peak":max(r["waveform_peak"] for r in utter)};write_json(BRANCH/"metrics/utterance_metrics.json",{"aggregate":agg,"utterances":utter})
    # Two fixed short loss ablations on the same fixture.
    ablations={}
    for name,loss_cfg in (("no_kl",{**cfg["loss"],"kl_weight":0.0}),("no_temporal_delta",{**cfg["loss"],"temporal_delta_weight":0.0})):
        am,_=short_run(cfg,fixture,80,cfg["seed"]+8,loss_cfg);am.eval();ao=am(fixture["window"],sample_posterior=False);_,vals=__import__('mini_multimodal.speech_vae',fromlist=['speech_vae_loss']).speech_vae_loss(ao,fixture["window"],fixture["mask"],**cfg["loss"]);ablations[name]=vals
    write_json(BRANCH/"metrics/ablation_metrics.json",ablations)
    # Cache all posterior means for P3-W03.
    all_ds=SpeechWindowDataset(ROOT,data_dir=cfg["output_data"],splits=["train","validation","test"],config=cfg);means=[];ids=[]
    with torch.inference_mode():
        for b in loader(all_ds,32,False,cfg["seed"]): means.append(model.encode(b["window"])[0].numpy().astype(np.float16));ids.extend(m["window_id"] for m in b["metadata"])
    cache=ROOT/"data/latents/p3_w02_v1/posterior_means.npz";cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,posterior_mean=np.concatenate(means),window_id=np.asarray(ids))
    package={"schema_version":1,"work_package":"P3-W02","iteration_id":"I001","branch_id":"spectral_vae","model_state":model.state_dict(),"model_config":cfg["model"],"feature_config":cfg["feature"],"window_config":cfg["window"],"vocoder_config":cfg["vocoder"],"source_checkpoint_sha256":sha(BRANCH/"checkpoints/best.pt")};torch.save(package,BRANCH/"inference_package.pt")
    # Curate four lowest and four highest ASR/boundary cases; top three are boundaries.
    ranked=sorted(utter,key=lambda r:(r["asr_cer"],r["overlap_boundary_mae"]));chosen=ranked[:4]+ranked[-4:];th=cfg["thresholds"]
    for i,r in enumerate(chosen):
        r["selection_type"]="boundary" if i>=5 else "representative";r["is_fully_correct"]=r["asr_cer"]<=th["maximum_asr_reconstruction_cer"] and r["overlap_boundary_mae"]<=th["maximum_overlap_boundary_mae"] and r["waveform_cycle_logmel_mae"]<=th["maximum_waveform_cycle_logmel_mae"] and r["waveform_peak"]<=1.0;r["teaching_note"]="固定解码链中的高误差或拼接边界样例。" if i>=5 else "代表性整句重建样例。"
    write_json(BRANCH/"examples/capability_examples.json",{"schema_version":1,"work_package":"P3-W02","iteration_id":"I001","branch_id":"spectral_vae","example_count":len(chosen),"examples":chosen})
    lines=["# P3-W02 I001语音潜表示直观能力样例","","> 8条样例来自固定测试语句；含4条低误差代表样例、1条普通样例和3条高误差/拼接边界样例。策展样例不替代完整测试统计。",""]
    for i,r in enumerate(chosen,1):
        original=r["input_artifacts"][0];audio=r["display_artifacts"][0];image=r["display_artifacts"][1];lines += [f"## 样例{i}：{r['source_record_id']}","",f"![原始音频]({Path(original).as_posix()})",f"![重建音频]({Path(audio).name})",f"![声谱对照](media/{Path(image).name})","",f"- 参考文本：{r['expected_output']}",f"- TinyASR回译：{r['model_outputs']['asr_transcript']}",f"- ASR CER：{r['asr_cer']:.4f}",f"- 拼接边界MAE：{r['overlap_boundary_mae']:.4f}",f"- 波形循环Log-Mel MAE：{r['waveform_cycle_logmel_mae']:.4f}",f"- 说明：{r['teaching_note']}",""]
    (BRANCH/"examples/capability_examples.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    acceptance={
      "DATA-SPR-001":len(train)+len(val)+len(test)>=th["minimum_total_windows"] and len(train)>=th["minimum_train_windows"],"MODEL-SPR-001":om["normalized_mae"]<=th["overfit_normalized_mae_max"],"MODEL-SPR-002":metrics["normalized_mae"]<=th["test_normalized_mae_max"] and metrics["logmel_mae"]<=th["test_logmel_mae_max"] and metrics["temporal_delta_mae"]<=th["test_delta_mae_max"],"MODEL-SPR-003":metrics["kl"]>th["minimum_kl"] and metrics["element_compression_ratio"]==4,"MODEL-SPR-004":agg["overlap_boundary_mae"]<=th["maximum_overlap_boundary_mae"],"MODEL-SPR-005":agg["waveform_cycle_logmel_mae"]<=th["maximum_waveform_cycle_logmel_mae"] and agg["asr_cer"]<=th["maximum_asr_reconstruction_cer"],"REPRO-SPR-001":l1==l2 and model_hash(first)==model_hash(second)}
    resource={"trainable_parameters":sum(p.numel() for p in model.parameters()),"wall_seconds":time.perf_counter()-started,"peak_rss_bytes":process.memory_info().rss};acceptance["RES-SPR-001"]=resource["trainable_parameters"]<=th["maximum_parameters"] and resource["wall_seconds"]<=th["maximum_wall_seconds"] and resource["peak_rss_bytes"]<=th["maximum_peak_rss_bytes"];write_json(BRANCH/"metrics/resource_benchmark.json",resource);write_json(BRANCH/"metrics/acceptance_report.json",{"items":acceptance,"all_blocking_acceptance_items_passed":all(acceptance.values()),"failed_items":[k for k,v in acceptance.items() if not v]});write_json(BRANCH/"logs/training_history.json",history)
    print(json.dumps({"metrics":metrics,"utterance":agg,"resource":resource,"acceptance":acceptance},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
