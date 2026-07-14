import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image

from unet_model import UNet
from diffusion_scheduler import DiffusionScheduler


def denormalize(img_tensor):
    return (img_tensor.clamp(-1, 1) + 1) / 2


def upscale_to_1080p(pil_image, device='cpu'):
    """
    Upscales a small generated image to 1080p using Real-ESRGAN.
    Real-ESRGAN is a pretrained super-resolution model — we don't train
    this ourselves, we just call it as a post-processing step.
    """
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    # x4 model — good balance of speed/quality for anime-style art
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=256,          # tile processing avoids OOM on larger outputs
        tile_pad=10,
        pre_pad=0,
        half=(device == 'cuda'),   # fp16 on GPU for speed, fp32 on CPU
        device=device
    )

    img_np = np.array(pil_image)
    output, _ = upsampler.enhance(img_np, outscale=4)
    upscaled = Image.fromarray(output)

    # Final resize to exact 1920x1080 (Real-ESRGAN x4 won't land exactly
    # on 1080p depending on input size, so we snap to it here)
    upscaled = upscaled.resize((1920, 1080), Image.LANCZOS)
    return upscaled


def load_model(checkpoint_path, device='cpu'):
    """
    Loads the consolidated sainyx_diffusion_full.pt file
    (same format saved at the end of train.py).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = UNet(
        in_channels=checkpoint.get('channels', 3),
        base_channels=checkpoint.get('base_channels', 64)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Loaded model — trained {checkpoint.get('epochs_trained', '?')} epochs, "
          f"final loss: {checkpoint.get('final_loss', '?')}")

    image_size = checkpoint.get('image_size', 64)
    timesteps = checkpoint.get('timesteps', 1000)

    return model, image_size, timesteps


def generate_images(model, image_size, timesteps, num_images=4, device='cpu',
                     save_path=None, upscale=False):
    scheduler = DiffusionScheduler(timesteps=timesteps, device=device)

    samples = scheduler.sample(
        model, image_size=image_size, batch_size=num_images,
        channels=3, device=device
    )
    samples = denormalize(samples)

    if save_path:
        nrow = int(num_images ** 0.5) or 1
        save_image(samples, save_path, nrow=nrow)
        print(f"🖼️  Saved to {save_path}")

    if upscale:
        # Upscale each image individually (upscaling only makes sense
        # per-image, not on a multi-image grid)
        upscaled_paths = []
        for i in range(samples.shape[0]):
            single = samples[i]
            single_np = (single.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            pil_img = Image.fromarray(single_np)

            upscaled = upscale_to_1080p(pil_img, device=device)

            if save_path:
                base = save_path.rsplit('.', 1)[0]
                up_path = f"{base}_1080p_{i}.png"
            else:
                up_path = f"generated_1080p_{i}.png"

            upscaled.save(up_path)
            upscaled_paths.append(up_path)
            print(f"🖼️  1080p upscale saved: {up_path}")

        return samples, upscaled_paths

    return samples


# ── Usage (local machine, after downloading sainyx_diffusion_full.pt) ──
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, image_size, timesteps = load_model("sainyx_diffusion_full.pt", device=device)

    generate_images(
        model, image_size=image_size, timesteps=timesteps,
        num_images=4, device=device, save_path="generated_sample.png"
    )