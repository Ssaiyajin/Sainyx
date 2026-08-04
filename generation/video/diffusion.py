"""
generation/video/diffusion.py
Forward noising process + reverse sampling loop for a DDPM.

add_noise() and sample_step() are unchanged from the original — the math
already generalizes to any tensor rank because _extract reshapes purely
based on len(x_shape). Feed it [B,C,H,W] clips and it behaves exactly as
before (Tier 1 image DDPM); feed it [B,T,C,H,W] clips (with t shaped [B],
one timestep per clip) and every frame in a clip gets noised/denoised
together, which is what lets VideoUNet's temporal block do anything useful.

Only sample() is extended, with an optional num_frames argument, so it can
draw a full clip of pure noise instead of a single frame.
"""

import torch
import torch.nn.functional as F


class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, arr, t, x_shape):
        batch_size = t.shape[0]
        out = arr.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x0, t, noise=None):
        """x0 may be [B,C,H,W] (image) or [B,T,C,H,W] (clip). t is always
        one timestep per batch element (per clip, not per frame) so the
        whole clip is noised jointly."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

        noisy = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy, noise

    @torch.no_grad()
    def sample_step(self, model, x, t, t_index, cond=None):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        predicted_noise = model(x, t, cond) if cond is not None else model(x, t)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean

        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(
        self, model, image_size, batch_size=1, channels=3,
        num_frames=None, cond=None, device="cuda",
    ):
        """
        num_frames=None -> Tier 1 behaviour, returns [B, C, H, W].
        num_frames=T    -> draws a full clip, returns [B, T, C, H, W], and
                            calls the model once per timestep on the whole
                            clip (t stays shaped [B], one timestep per clip,
                            same as during training).
        """
        if num_frames is None:
            shape = (batch_size, channels, image_size, image_size)
        else:
            shape = (batch_size, num_frames, channels, image_size, image_size)

        x = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.sample_step(model, x, t, i, cond=cond)

        return x