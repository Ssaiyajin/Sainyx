import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from unet_model import UNet
from diffusion_scheduler import DiffusionScheduler
from dataset_loader import ImageDataset


def denormalize(img_tensor):
    """Convert from [-1, 1] back to [0, 1] for saving/viewing."""
    return (img_tensor.clamp(-1, 1) + 1) / 2


def train(cfg, dataset_dir, checkpoint_dir, sample_dir, output_dir="/kaggle/working"):
    """
    cfg: object with IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, TIMESTEPS, CHANNELS, DEVICE
    dataset_dir: path to dataset_raw/ (folder of tag subfolders with images)
    checkpoint_dir: where per-epoch checkpoints get saved
    sample_dir: where sample image grids get saved during training
    output_dir: where the FINAL downloadable .pt file gets saved (Kaggle output root by default)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    dataset = ImageDataset(dataset_dir, image_size=cfg.IMAGE_SIZE)
    dataloader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=2, drop_last=True
    )

    model = UNet(in_channels=cfg.CHANNELS, base_channels=64).to(cfg.DEVICE)
    scheduler = DiffusionScheduler(timesteps=cfg.TIMESTEPS, device=cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    print(f"🔥 Training on {len(dataset)} images | device: {cfg.DEVICE}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    final_loss = None

    for epoch in range(cfg.EPOCHS):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")

        for images in progress:
            images = images.to(cfg.DEVICE)
            batch_size = images.shape[0]

            t = torch.randint(0, cfg.TIMESTEPS, (batch_size,), device=cfg.DEVICE).long()
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, t, noise=noise)
            predicted_noise = model(noisy_images, t)

            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        final_loss = avg_loss
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

        # Per-epoch checkpoint (for resuming) + sample grid every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.EPOCHS:
            checkpoint_path = f"{checkpoint_dir}/sainyx_diffusion_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

            samples = scheduler.sample(
                model, image_size=cfg.IMAGE_SIZE, batch_size=4,
                channels=cfg.CHANNELS, device=cfg.DEVICE
            )
            samples = denormalize(samples)
            sample_path = f"{sample_dir}/epoch{epoch+1}.png"
            save_image(samples, sample_path, nrow=2)
            print(f"🖼️  Saved sample grid: {sample_path}")

    # ── FINAL consolidated download-ready file ──
    # Same convention as sainyx_v2_full.pt — one file with everything needed
    # to reload the model elsewhere (locally, or in the Sainyx Flask app).
    full_model_path = f"{output_dir}/sainyx_diffusion_full.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'image_size': cfg.IMAGE_SIZE,
        'channels': cfg.CHANNELS,
        'timesteps': cfg.TIMESTEPS,
        'base_channels': 64,
        'final_loss': final_loss,
        'epochs_trained': cfg.EPOCHS,
    }, full_model_path)
    print(f"\n🔥 Training complete! Final model saved to: {full_model_path}")
    print(f"   Download this from Kaggle's Output panel (right sidebar → Output → Download).")

    return model, scheduler