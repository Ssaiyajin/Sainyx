"""
model/video_unet.py

TinyUNet: unchanged from the original — still what the image DDPM pipeline
uses. Kept here for backward compatibility.

VideoUNet: new. Same spatial blocks as TinyUNet (imported from
video_unet_blocks so nothing is duplicated), plus TemporalMix, a
from-scratch temporal block so information can flow between frames in a
clip. Not a reimplementation of a published video-diffusion architecture
(no 3D convs, no factorized spatio-temporal attention) — it pools each
frame's feature map to a per-channel summary, runs a small 1D conv across
the T axis on those summaries so each frame "sees" its neighbours, then
feeds the result back as a per-frame additive bias (FiLM-style) on the
original spatial features.
"""

import torch
import torch.nn as nn

from model.video_unet_blocks import timestep_embedding, ResBlock, Down, Up


class TinyUNet(nn.Module):
    """Original Tier-1 image DDPM U-Net. Unchanged behaviour."""

    def __init__(
        self,
        channels_in=3,
        channels_out=3,
        base_ch=64,
        time_emb_dim=256,
        cond_emb_dim=None,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim

        self.in_conv = nn.Conv2d(channels_in, base_ch, 3, padding=1)

        self.down1 = Down(base_ch, base_ch * 2, time_emb_dim, cond_emb_dim)
        self.down2 = Down(base_ch * 2, base_ch * 4, time_emb_dim, cond_emb_dim)

        self.bottleneck = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim, cond_emb_dim)

        self.up1 = Up(base_ch * 4, base_ch * 4, base_ch * 2, time_emb_dim, cond_emb_dim)
        self.up2 = Up(base_ch * 2, base_ch * 2, base_ch, time_emb_dim, cond_emb_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, channels_out, 3, padding=1)
        )

    def forward(self, x, t, cond_emb=None):
        t_emb = self.time_mlp(timestep_embedding(t, self.time_emb_dim))

        x = self.in_conv(x)
        x, skip1 = self.down1(x, t_emb, cond_emb)
        x, skip2 = self.down2(x, t_emb, cond_emb)

        x = self.bottleneck(x, t_emb, cond_emb)

        x = self.up1(x, skip2, t_emb, cond_emb)
        x = self.up2(x, skip1, t_emb, cond_emb)

        return self.out_conv(x)


class TemporalMix(nn.Module):
    """Custom-designed temporal mixing block. See module docstring."""

    def __init__(self, channels):
        super().__init__()
        self.temporal_conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(channels, channels))

    def forward(self, x, T):
        BT, C, H, W = x.shape
        B = BT // T

        pooled = x.mean(dim=[2, 3])                      # [B*T, C]
        pooled = pooled.view(B, T, C).permute(0, 2, 1)    # [B, C, T]
        mixed = self.temporal_conv(pooled)                # [B, C, T]
        mixed = mixed.permute(0, 2, 1).reshape(B * T, C)  # [B*T, C]

        bias = self.gate(mixed)[:, :, None, None]         # [B*T, C, 1, 1]
        return x + bias


class VideoUNet(nn.Module):
    """
    Same spatial backbone as TinyUNet, with TemporalMix inserted at each
    resolution so frames can influence each other.

    Input/output shape: [B, T, C, H, W].
    """

    def __init__(self, channels_in=3, channels_out=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_emb_dim = time_emb_dim

        self.in_conv = nn.Conv2d(channels_in, base_ch, 3, padding=1)

        self.down1 = Down(base_ch, base_ch * 2, time_emb_dim)
        self.temporal_down1 = TemporalMix(base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4, time_emb_dim)
        self.temporal_down2 = TemporalMix(base_ch * 4)

        self.bottleneck = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.temporal_bottleneck = TemporalMix(base_ch * 4)

        self.up1 = Up(base_ch * 4, base_ch * 4, base_ch * 2, time_emb_dim)
        self.temporal_up1 = TemporalMix(base_ch * 2)
        self.up2 = Up(base_ch * 2, base_ch * 2, base_ch, time_emb_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, channels_out, 3, padding=1)
        )

    def forward(self, x, t):
        """
        x: [B, T, C, H, W]
        t: [B] diffusion timestep, one per clip (broadcast across all T
           frames so the whole clip is noised/denoised together — this is
           what makes it a *video* model and not T independent images).
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        if t.shape[0] == B:
            t = t.repeat_interleave(T)  # [B] -> [B*T]

        t_emb = self.time_mlp(timestep_embedding(t, self.time_emb_dim))

        x = self.in_conv(x)

        x, skip1 = self.down1(x, t_emb)
        x = self.temporal_down1(x, T)
        skip1 = self.temporal_down1(skip1, T)

        x, skip2 = self.down2(x, t_emb)
        x = self.temporal_down2(x, T)
        skip2 = self.temporal_down2(skip2, T)

        x = self.bottleneck(x, t_emb)
        x = self.temporal_bottleneck(x, T)

        x = self.up1(x, skip2, t_emb)
        x = self.temporal_up1(x, T)

        x = self.up2(x, skip1, t_emb)

        x = self.out_conv(x)
        return x.reshape(B, T, C, H, W)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = VideoUNet(base_ch=64)
    print(f"Param count: {count_params(model):,}")
    x = torch.randn(2, 8, 3, 64, 64)   # B=2 clips, T=8 frames
    t = torch.randint(0, 1000, (2,))   # one timestep per clip
    out = model(x, t)
    print(f"Output shape: {out.shape}")  # expect [2, 8, 3, 64, 64]