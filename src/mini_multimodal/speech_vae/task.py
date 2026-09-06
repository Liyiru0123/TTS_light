"""Mask-aware SpeechVAE loss and shared Trainer adapter."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from mini_multimodal.training import StepOutput


def speech_vae_loss(output, target, mask, *, reconstruction_weight=1.0, temporal_delta_weight=0.25, kl_weight=1e-5):
    if mask.shape != (target.shape[0], target.shape[-1]): raise ValueError("mask shape mismatch")
    expanded=mask[:,None,None,:].to(target.dtype); denom=expanded.sum()*target.shape[2]
    reconstruction=((output.reconstruction-target).abs()*expanded).sum()/denom.clamp_min(1)
    pair=(mask[:,1:]&mask[:,:-1])[:,None,None,:].to(target.dtype); pair_denom=pair.sum()*target.shape[2]
    delta_target=target[...,1:]-target[...,:-1]; delta_recon=output.reconstruction[...,1:]-output.reconstruction[...,:-1]
    delta=((delta_recon-delta_target).abs()*pair).sum()/pair_denom.clamp_min(1)
    latent_mask=F.interpolate(mask[:,None,None,:].float(),size=output.mean.shape[-2:],mode="nearest")
    kl_map=-0.5*(1+output.logvar-output.mean.square()-output.logvar.exp())
    kl=(kl_map*latent_mask).sum()/(latent_mask.sum()*output.mean.shape[1]).clamp_min(1)
    loss=reconstruction_weight*reconstruction+temporal_delta_weight*delta+kl_weight*kl
    return loss,{"normalized_mae":float(reconstruction.detach()),"temporal_delta_normalized_mae":float(delta.detach()),"kl":float(kl.detach())}


class SpeechVAETask:
    def __init__(self,**weights): self.weights=weights
    def step(self,model,batch,*,training):
        output=model(batch["window"],sample_posterior=training); loss,metrics=speech_vae_loss(output,batch["window"],batch["mask"],**self.weights)
        return StepOutput(loss=loss,metrics=metrics,weight=int(batch["window"].shape[0]))
