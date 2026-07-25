# ══════════════════════════════════════════════════
# Sainyx Diffusion — U-Net Model
# ══════════════════════════════════════════════════
#
# This U-Net predicts the noise added to an image at a given timestep.
# Structure: downsample path -> bottleneck -> upsample path, with skip
# connections (like a regular U-Net), plus timestep embeddings injected
# into every block so the model knows "how noisy" the input currently is.

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Same idea as positional embeddings in a transformer, but for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Conv block with GroupNorm, SiLU activation, and timestep embedding injection."""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.block1(x)
        time_out = self.time_mlp(t_emb)
        h = h + time_out[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    """
    Small U-Net for 64x64 (or 128x128) image diffusion.
    base_channels controls model size — 64 is a good starting point for Kaggle T4s.
    """

    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ── Downsampling path ──
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.down1_pool = Downsample(base_channels)                       # 64 -> 32

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2_pool = Downsample(base_channels * 2)                   # 32 -> 16

        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3_pool = Downsample(base_channels * 4)                   # 16 -> 8

        # ── Bottleneck ──
        self.bottleneck1 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.bottleneck2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # ── Upsampling path (with skip connections) ──
        self.up3_upsample = Upsample(base_channels * 4)                   # 8 -> 16
        self.up3 = ResidualBlock(base_channels * 4 + base_channels * 4, base_channels * 2, time_emb_dim)

        self.up2_upsample = Upsample(base_channels * 2)                   # 16 -> 32
        self.up2 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, time_emb_dim)

        self.up1_upsample = Upsample(base_channels)                       # 32 -> 64
        self.up1 = ResidualBlock(base_channels + base_channels, base_channels, time_emb_dim)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        x0 = self.init_conv(x)                    # [B, base, 64, 64]

        d1 = self.down1(x0, t_emb)                 # [B, base, 64, 64]
        d1_down = self.down1_pool(d1)               # [B, base, 32, 32]

        d2 = self.down2(d1_down, t_emb)             # [B, base*2, 32, 32]
        d2_down = self.down2_pool(d2)                # [B, base*2, 16, 16]

        d3 = self.down3(d2_down, t_emb)              # [B, base*4, 16, 16]
        d3_down = self.down3_pool(d3)                 # [B, base*4, 8, 8]

        b = self.bottleneck1(d3_down, t_emb)
        b = self.bottleneck2(b, t_emb)                # [B, base*4, 8, 8]

        u3 = self.up3_upsample(b)                      # [B, base*4, 16, 16]
        u3 = torch.cat([u3, d3], dim=1)                 # skip connection
        u3 = self.up3(u3, t_emb)                        # [B, base*2, 16, 16]

        u2 = self.up2_upsample(u3)                       # [B, base*2, 32, 32]
        u2 = torch.cat([u2, d2], dim=1)                   # skip connection
        u2 = self.up2(u2, t_emb)                          # [B, base, 32, 32]

        u1 = self.up1_upsample(u2)                         # [B, base, 64, 64]
        u1 = torch.cat([u1, d1], dim=1)                     # skip connection
        u1 = self.up1(u1, t_emb)                            # [B, base, 64, 64]

        return self.final_conv(u1)                           # [B, 3, 64, 64] predicted noise


# ── Quick sanity check ──────────────────────────────
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")