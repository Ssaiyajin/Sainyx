# ══════════════════════════════════════════════════
# Sainyx Diffusion — Noise Scheduler & Diffusion Process
# ══════════════════════════════════════════════════
#
# This handles two directions:
# 1. Forward process: gradually add noise to a real image over T steps
#    (used during training — we corrupt images and ask the U-Net to
#    predict what noise was added)
# 2. Reverse process: start from pure noise and iteratively denoise
#    using the trained U-Net (used during inference/generation)

import torch
import torch.nn.functional as F


class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device

        # Linear beta schedule — how much noise is added at each step
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute terms used repeatedly
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, values, t, shape):
        """Pull out the right timestep values and reshape for broadcasting over an image batch."""
        batch_size = t.shape[0]
        out = values.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def add_noise(self, x_start, t, noise=None):
        """
        Forward process: take a clean image x_start and noise it to timestep t.
        Returns the noisy image. Used during training.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def sample_step(self, model, x, t, t_index):
        """One reverse diffusion step: denoise x from timestep t to t-1."""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        predicted_noise = model(x, t)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=1, channels=3, device='cpu'):
        """
        Full reverse process: start from pure noise and denoise all the way
        down to timestep 0, returning a generated image batch.
        """
        model.eval()
        x = torch.randn((batch_size, channels, image_size, image_size), device=device)

        for t_index in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), t_index, device=device, dtype=torch.long)
            x = self.sample_step(model, x, t, t_index)

        model.train()
        return x  # values roughly in [-1, 1] — denormalize before saving as image