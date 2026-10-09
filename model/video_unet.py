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
import torch.nn.functional as F

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


class TemporalBlock(nn.Module):
    """Per-pixel temporal conv over the T axis (kernel 3x1x1), zero-initialised
    residual. Unlike TemporalMix it keeps the spatial layout, so a feature at
    (y, x) in frame t can be matched with the same place in frames t-1 / t+1.
    That is what lets the model represent motion at all."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0))
        self.out = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, T):
        BT, C, H, W = x.shape
        B = BT // T
        h = F.silu(self.norm(x))
        h = h.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)      # [B, C, T, H, W]
        h = self.out(F.silu(self.conv(h)))
        h = h.permute(0, 2, 1, 3, 4).reshape(BT, C, H, W)
        return x + h


class TemporalAttention(nn.Module):
    """Self-attention across frames at every spatial position (bottleneck only,
    where H*W is small). Gives every frame a view of the whole clip, which keeps
    long-range content consistent. Frame order comes from a sinusoidal embedding."""

    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.out = nn.Linear(channels, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.channels = channels

    def forward(self, x, T):
        BT, C, H, W = x.shape
        B = BT // T
        seq = x.view(B, T, C, H * W).permute(0, 3, 1, 2).reshape(B * H * W, T, C)
        pos = timestep_embedding(torch.arange(T, device=x.device), C).to(seq.dtype)
        h = self.norm(seq) + pos[None]
        h, _ = self.attn(h, h, h, need_weights=False)
        seq = seq + self.out(h)
        return seq.view(B, H * W, T, C).permute(0, 2, 3, 1).reshape(BT, C, H, W)


class VideoUNet(nn.Module):
    """
    Same spatial backbone as TinyUNet, with TemporalMix inserted at each
    resolution so frames can influence each other.

    Input/output shape: [B, T, C, H, W].
    """

    def __init__(self, channels_in=3, channels_out=3, base_ch=64, time_emb_dim=256,
                 temporal="mix"):
        super().__init__()
        # temporal="mix": original pooled TemporalMix (old checkpoints).
        # temporal="v2":  TemporalBlock at every level + TemporalAttention in the
        #                 bottleneck (needed for real frame-to-frame motion).
        self.temporal = temporal
        Mix = TemporalMix if temporal == "mix" else TemporalBlock
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_emb_dim = time_emb_dim

        self.in_conv = nn.Conv2d(channels_in, base_ch, 3, padding=1)

        self.down1 = Down(base_ch, base_ch * 2, time_emb_dim)
        self.temporal_down1 = Mix(base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4, time_emb_dim)
        self.temporal_down2 = Mix(base_ch * 4)

        self.bottleneck = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.temporal_bottleneck = Mix(base_ch * 4)

        self.temporal_attn = TemporalAttention(base_ch * 4) if temporal == "v2" else None

        self.up1 = Up(base_ch * 4, base_ch * 4, base_ch * 2, time_emb_dim)
        self.temporal_up1 = Mix(base_ch * 2)
        self.up2 = Up(base_ch * 2, base_ch * 2, base_ch, time_emb_dim)
        self.temporal_up2 = TemporalBlock(base_ch) if temporal == "v2" else None

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
        if self.temporal_attn is not None:
            x = self.temporal_attn(x, T)

        x = self.up1(x, skip2, t_emb)
        x = self.temporal_up1(x, T)

        x = self.up2(x, skip1, t_emb)
        if self.temporal == "v2":
            x = self.temporal_up2(x, T)

        x = self.out_conv(x)
        return x.reshape(B, T, C, H, W)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = VideoUNet(base_ch=64, temporal="v2")
    print(f"Param count: {count_params(model):,}")
    x = torch.randn(2, 16, 3, 64, 64)  # B=2 clips, T=16 frames
    t = torch.randint(0, 1000, (2,))   # one timestep per clip
    out = model(x, t)
    print(f"Output shape: {out.shape}")  # expect [2, 8, 3, 64, 64]