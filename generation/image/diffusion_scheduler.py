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

import math

import torch
import torch.nn.functional as F


def cosine_betas(timesteps, s=0.008, max_beta=0.999):
    """Cosine noise schedule (Nichol & Dhariwal). Destroys information more
    gradually than the linear schedule, which matters at small resolutions."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=max_beta).float()


class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu',
                 schedule='linear'):
        self.timesteps = timesteps
        self.device = device
        self.schedule = schedule

        # How much noise is added at each step. Old checkpoints use 'linear';
        # the tag-conditioned model uses 'cosine'.
        if schedule == 'cosine':
            self.betas = cosine_betas(timesteps).to(device)
        elif schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        else:
            raise ValueError(f"unknown noise schedule: {schedule}")

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
    def sample_step(self, model, x, t, t_index, clip_denoised=True, generator=None):
        """One reverse diffusion step: denoise x from timestep t to t-1."""
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        predicted_noise = model(x, t)

        # Reconstruct predicted x0 and clip it to [-1, 1] — this is the key
        # stabilizer. Without it, small prediction errors compound over
        # 1000 steps and can drive pixel values toward the clamp boundary,
        # collapsing the image toward black/uniform color.
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)

        alphas_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        alphas_cumprod_prev_t = self._extract(self.alphas_cumprod_prev, t, x.shape)
        betas_t = self._extract(self.betas, t, x.shape)
        alphas_t = self._extract(self.alphas, t, x.shape)

        # Posterior mean using the clipped x0 estimate (standard DDPM posterior formula)
        posterior_mean = (
            (torch.sqrt(alphas_cumprod_prev_t) * betas_t / (1.0 - alphas_cumprod_t)) * pred_x0
            + (torch.sqrt(alphas_t) * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)) * x
        )

        if t_index == 0:
            return posterior_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn(
                x.shape, device=x.device, dtype=x.dtype, generator=generator
            )
            return posterior_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=1, channels=3, device='cpu', generator=None):
        """
        Full reverse process: start from pure noise and denoise all the way
        down to timestep 0, returning a generated image batch.
        """
        model.eval()
        x = torch.randn(
            (batch_size, channels, image_size, image_size),
            device=device,
            generator=generator,
        )

        for t_index in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), t_index, device=device, dtype=torch.long)
            x = self.sample_step(model, x, t, t_index, generator=generator)

        model.train()
        return x  # values roughly in [-1, 1] — denormalize before saving as image

    @torch.no_grad()
    def ddim_sample(self, model, image_size, batch_size=1, channels=3, device='cpu',
                    generator=None, cond=None, guidance_scale=1.0, steps=50):
        """
        Deterministic DDIM sampling with optional classifier-free guidance.

        cond is a [batch, num_tags] multi-hot tensor (or None for unconditional).
        With guidance_scale > 1 the model is run with and without the tags and
        the difference is amplified, which is what makes the image actually
        follow the prompt.
        """
        model.eval()
        x = torch.randn((batch_size, channels, image_size, image_size),
                        device=device, generator=generator)
        steps = min(steps, self.timesteps)
        ts = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        use_guidance = cond is not None and guidance_scale != 1.0

        for i, t in enumerate(ts):
            tt = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if use_guidance:
                eps_cond, eps_uncond = model(
                    torch.cat([x, x]), torch.cat([tt, tt]),
                    torch.cat([cond, torch.zeros_like(cond)]),
                ).chunk(2)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            elif cond is not None:
                eps = model(x, tt, cond)
            else:
                eps = model(x, tt)

            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=device)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()  # keep eps consistent with clipped x0
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

        return x