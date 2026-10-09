# ══════════════════════════════════════════════════
# Sainyx Tag-Conditioned U-Net
# ══════════════════════════════════════════════════
#
# Same job as model/image_unet.py (predict the noise in a noisy image) but
# it also receives a multi-hot "tag vector" saying what the image should
# contain (character, hair colour, form, ...). The tag vector is turned into
# an embedding and mixed into every residual block through FiLM
# (scale and shift), so the prompt can steer the whole image.
#
# A zero tag vector means "no condition". Training randomly zeroes the tags,
# which lets sampling use classifier-free guidance.
#
# The old unconditional UNet stays in model/image_unet.py so existing
# checkpoints keep loading.

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.image_unet import SinusoidalTimeEmbedding


def _norm(channels):
    return nn.GroupNorm(32, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)  # FiLM: scale and shift
        self.norm2 = _norm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        assert channels % heads == 0
        self.heads = heads
        self.norm = _norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        # [b, heads, tokens, head_dim]
        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class _EmbSequential(nn.Sequential):
    """Sequential that passes the conditioning embedding to ResBlocks only."""

    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, ResBlock) else layer(x)
        return x


class CondUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, ch_mult=(1, 2, 4, 4),
                 num_res_blocks=2, attn_levels=(2, 3), num_tags=1, dropout=0.1, heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_tags = num_tags
        emb_dim = base_channels * 4

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )
        self.tag_in = nn.Linear(num_tags, emb_dim, bias=False)
        self.tag_out = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.input_blocks = nn.ModuleList([
            _EmbSequential(nn.Conv2d(in_channels, base_channels, 3, padding=1))
        ])
        skip_channels = [base_channels]
        ch = base_channels
        for level, mult in enumerate(ch_mult):
            out = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out, emb_dim, dropout)]
                ch = out
                if level in attn_levels:
                    layers.append(AttentionBlock(ch, heads))
                self.input_blocks.append(_EmbSequential(*layers))
                skip_channels.append(ch)
            if level != len(ch_mult) - 1:
                self.input_blocks.append(_EmbSequential(Downsample(ch)))
                skip_channels.append(ch)

        self.middle = _EmbSequential(
            ResBlock(ch, ch, emb_dim, dropout), AttentionBlock(ch, heads),
            ResBlock(ch, ch, emb_dim, dropout),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            out = base_channels * mult
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(ch + skip_channels.pop(), out, emb_dim, dropout)]
                ch = out
                if level in attn_levels:
                    layers.append(AttentionBlock(ch, heads))
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                self.output_blocks.append(_EmbSequential(*layers))

        self.out = nn.Sequential(_norm(ch), nn.SiLU(), nn.Conv2d(ch, in_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def embed(self, t, cond):
        emb = self.time_embed(t)
        if cond is None:
            cond = torch.zeros(t.shape[0], self.num_tags, device=t.device, dtype=emb.dtype)
        cond = cond.to(emb.dtype)
        scale = cond.sum(dim=1, keepdim=True).clamp(min=1.0).sqrt()
        return emb + self.tag_out(self.tag_in(cond) / scale)

    def forward(self, x, t, cond=None):
        emb = self.embed(t, cond)
        hs = []
        h = x
        for block in self.input_blocks:
            h = block(h, emb)
            hs.append(h)
        h = self.middle(h, emb)
        for block in self.output_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h)