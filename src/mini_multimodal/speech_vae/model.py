"""Compact spatial Gaussian VAE for fixed Log-Mel windows."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__(); groups = min(8, channels)
        self.block = nn.Sequential(nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))
    def forward(self, x): return x + self.block(x)


@dataclass(slots=True)
class SpeechVAEOutput:
    reconstruction: torch.Tensor; mean: torch.Tensor; logvar: torch.Tensor; latent: torch.Tensor


class TinySpeechVAE(nn.Module):
    def __init__(self, *, base_channels=16, latent_channels=4, logvar_min=-8.0, logvar_max=4.0, n_mels=64, window_frames=256):
        super().__init__(); hidden = base_channels * 2; deep = base_channels * 3
        if n_mels % 4 or window_frames % 4:
            raise ValueError("n_mels and window_frames must be divisible by four")
        self.latent_channels=latent_channels; self.logvar_min=logvar_min; self.logvar_max=logvar_max
        self.n_mels=n_mels; self.window_frames=window_frames
        self.encoder=nn.Sequential(nn.Conv2d(1,base_channels,3,padding=1),ResidualBlock(base_channels),nn.Conv2d(base_channels,hidden,4,2,1),ResidualBlock(hidden),nn.Conv2d(hidden,deep,4,2,1),ResidualBlock(deep),nn.GroupNorm(8,deep),nn.SiLU(),nn.Conv2d(deep,latent_channels*2,3,padding=1))
        self.decoder_input=nn.Conv2d(latent_channels,deep,3,padding=1)
        self.decoder=nn.Sequential(ResidualBlock(deep),nn.Upsample(scale_factor=2,mode="nearest"),nn.Conv2d(deep,hidden,3,padding=1),ResidualBlock(hidden),nn.Upsample(scale_factor=2,mode="nearest"),nn.Conv2d(hidden,base_channels,3,padding=1),ResidualBlock(base_channels),nn.GroupNorm(8,base_channels),nn.SiLU(),nn.Conv2d(base_channels,1,3,padding=1),nn.Tanh())
    def encode(self,x):
        expected=(1,self.n_mels,self.window_frames)
        if x.ndim!=4 or x.shape[1:]!=expected: raise ValueError(f"TinySpeechVAE expects [B,{expected[0]},{expected[1]},{expected[2]}]")
        mean,logvar=self.encoder(x).chunk(2,1); return mean,logvar.clamp(self.logvar_min,self.logvar_max)
    @staticmethod
    def reparameterize(mean,logvar,*,generator=None): return mean+torch.exp(0.5*logvar)*torch.randn(mean.shape,device=mean.device,dtype=mean.dtype,generator=generator)
    def decode(self,z):
        if z.ndim!=4 or z.shape[1]!=self.latent_channels: raise ValueError("latent shape mismatch")
        return self.decoder(self.decoder_input(z))
    def forward(self,x,*,sample_posterior=True,generator=None):
        mean,logvar=self.encode(x); z=self.reparameterize(mean,logvar,generator=generator) if sample_posterior else mean
        return SpeechVAEOutput(self.decode(z),mean,logvar,z)
