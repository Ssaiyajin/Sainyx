"""
model/unet.py
A deliberately small U-Net for the tiny-DDPM milestone.
Target: low tens of millions of params, fits comfortably on a single T4.

Built with a `cond` argument threaded through from day one, even though
Tier 1 training won't use it. That way Tier 2 (image-to-image /
"turn this guy into a Super Saiyan") doesn't need an architecture rewrite -
you just start passing a real conditioning tensor instead of None.
"""

import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    """Sinusoidal embedding for the diffusion timestep, same idea as
    positional embeddings in a transformer - gives the model a sense of
    'how noisy is this input right now'."""
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
    """Conv block that injects the timestep embedding (and optionally a
    conditioning embedding) additively into the feature map."""

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
        return self.pool(h), h  # downsampled, skip connection


class Up(nn.Module):
    """in_ch: channels coming into this block (from bottleneck/previous Up).
    skip_ch: channels of the matching skip connection from the Down path.
    out_ch: channels to produce for the next block up."""

    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim, cond_emb_dim=None):
        super().__init__()
        # Upsample straight to skip_ch so concatenation always lines up.
        self.upsample = nn.ConvTranspose2d(in_ch, skip_ch, 4, stride=2, padding=1)
        self.res = ResBlock(skip_ch * 2, out_ch, time_emb_dim, cond_emb_dim)

    def forward(self, x, skip, t_emb, cond_emb=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb, cond_emb)


class TinyUNet(nn.Module):
    """
    channels_in: 3 for plain RGB (Tier 1). Bump to 6 for Tier 2 if you
    concatenate the source image alongside the noisy target image.
    cond_emb_dim: set to a real value (e.g. 256) once you have a text/label
    embedding to condition on (Tier 2). Leave None for Tier 1.
    """

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

        # skip1 has base_ch*2 channels (down1's res output), skip2 has base_ch*4
        self.down1 = Down(base_ch, base_ch * 2, time_emb_dim, cond_emb_dim)
        self.down2 = Down(base_ch * 2, base_ch * 4, time_emb_dim, cond_emb_dim)

        self.bottleneck = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim, cond_emb_dim)

        # up1: in=bottleneck(base_ch*4), skip=skip2(base_ch*4), out=base_ch*2
        self.up1 = Up(base_ch * 4, base_ch * 4, base_ch * 2, time_emb_dim, cond_emb_dim)
        # up2: in=up1 output(base_ch*2), skip=skip1(base_ch*2), out=base_ch
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Quick sanity check - run this locally to confirm param count before
    # committing to a training run on Kaggle.
    model = TinyUNet(base_ch=64)
    print(f"Param count: {count_params(model):,}")
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    print(f"Output shape: {out.shape}")