"""
model/video_unet_blocks.py

Shared spatial building blocks, pulled out of the original video_unet.py
so both TinyUNet (image DDPM) and VideoUNet (temporal DDPM) can reuse the
exact same, already-working spatial architecture. Nothing here changed
from the original TinyUNet implementation — just relocated so it's not
duplicated between the two models.
"""

import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_ch) if cond_emb_dim else None

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.residual_conv = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb, cond_emb=None):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        if cond_emb is not None and self.cond_mlp is not None:
            h = h + self.cond_mlp(cond_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_emb_dim=None):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_emb_dim, cond_emb_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb, cond_emb=None):
        h = self.res(x, t_emb, cond_emb)
        return self.pool(h), h


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim, cond_emb_dim=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, skip_ch, 4, stride=2, padding=1)
        self.res = ResBlock(skip_ch * 2, out_ch, time_emb_dim, cond_emb_dim)

    def forward(self, x, skip, t_emb, cond_emb=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb, cond_emb)