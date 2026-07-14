import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    """Loads all images from dataset_raw/<tag>/ subfolders into one flat dataset."""

    def __init__(self, root_dir, image_size=64):
        self.image_paths = []
        for tag_folder in os.listdir(root_dir):
            tag_path = os.path.join(root_dir, tag_folder)
            if os.path.isdir(tag_path):
                for fname in os.listdir(tag_path):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(tag_path, fname))

        print(f"📦 Found {len(self.image_paths)} images in {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)