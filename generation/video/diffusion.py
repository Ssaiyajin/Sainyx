"""
diffusion.py
Forward noising process + reverse sampling loop for a DDPM.
No learned parameters live here - just the math. The U-Net (model/unet.py)
is what actually learns anything.
"""

import torch
import torch.nn.functional as F


class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        # Linear beta schedule (simplest option - cosine schedule is a later upgrade)
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute terms used repeatedly during training/sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, arr, t, x_shape):
        """Pull the right timestep values out of a precomputed tensor and
        reshape them to broadcast against a batch of images."""
        batch_size = t.shape[0]
        out = arr.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x0, t, noise=None):
        """Forward process: q(x_t | x_0). Takes a clean image x0 and a
        timestep t, returns the noised version plus the noise that was added
        (the noise is what the model will learn to predict)."""
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
        """One reverse-diffusion step: predict noise, remove it, add back a
        controlled amount of randomness (unless we're at the last step)."""
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
    def sample(self, model, image_size, batch_size=1, channels=3, cond=None, device="cuda"):
        """Full reverse process: start from pure noise, run T steps, return a
        generated image (or batch of images). This is Tier 1 (unconditional).
        Tier 2 conditional sampling reuses this exact loop, just passes cond."""
        x = torch.randn((batch_size, channels, image_size, image_size), device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.sample_step(model, x, t, i, cond=cond)

        return x