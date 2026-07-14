import torch
from torchvision.utils import save_image

from unet_model import UNet
from diffusion_scheduler import DiffusionScheduler


def denormalize(img_tensor):
    return (img_tensor.clamp(-1, 1) + 1) / 2


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


def generate_images(model, image_size, timesteps, num_images=4, device='cpu', save_path=None):
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

    return samples


# ── Usage (local machine, after downloading sainyx_diffusion_full.pt) ──
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, image_size, timesteps = load_model("sainyx_diffusion_full.pt", device=device)

    generate_images(
        model, image_size=image_size, timesteps=timesteps,
        num_images=4, device=device, save_path="generated_sample.png"
    )