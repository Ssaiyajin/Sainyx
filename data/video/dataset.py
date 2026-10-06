"""
data/dataset.py
Loads a folder of images for Tier 1 (unconditional) training.

For Tier 2 (image-to-image, e.g. "turn this guy into a Super Saiyan"),
swap this for PairedImageDataset below, which expects:
    data_dir/
        source/   (e.g. plain photo)
        target/   (e.g. saiyan-ified version)
    with matching filenames in each folder.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageFolderDataset(Dataset):
    """Tier 1: plain unconditional dataset. Point this at a folder of
    DBZ/anime stills - same curation instinct you used for the text model."""

    def __init__(self, root_dir, image_size=64):
        self.root_dir = root_dir
        self.files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class PairedImageDataset(Dataset):
    """Tier 2: source/target pairs for conditional image-to-image training
    (e.g. 'normal guy' -> 'super saiyan guy'). Filenames must match between
    the source/ and target/ subfolders.

    This is the dataset you'll need once you're ready to move past
    unconditional generation - the hard part isn't the code, it's collecting
    or generating the before/after pairs themselves.
    """

    def __init__(self, root_dir, image_size=64):
        self.source_dir = os.path.join(root_dir, "source")
        self.target_dir = os.path.join(root_dir, "target")
        self.files = [
            f for f in os.listdir(self.source_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        source = Image.open(os.path.join(self.source_dir, fname)).convert("RGB")
        target = Image.open(os.path.join(self.target_dir, fname)).convert("RGB")
        return self.transform(source), self.transform(target)