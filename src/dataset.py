from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class ForestDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.images = sorted(self.image_dir.glob("*.tiff"))

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        # Corresponding mask has same filename but .png extension
        mask_path = self.mask_dir / (image_path.stem + ".png")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load mask
        mask = Image.open(mask_path)

        # Convert to NumPy
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask)

        # Normalize image from 0-255 to 0-1
        image = image / 255.0

        # Convert image:
        # (H, W, C) → (C, H, W)
        image = torch.tensor(image).permute(2, 0, 1)

        # IMPORTANT:
        # Dataset uses mask values 1 and 2.
        # Convert them to classes 0 and 1.
        mask = (mask == 2).astype(np.int64)

        mask = torch.tensor(mask)

        return image, mask


if __name__ == "__main__":

    dataset = ForestDataset(
        "data/Amazon Forest Dataset/Training/images",
        "data/Amazon Forest Dataset/Training/masks"
    )

    print("=" * 50)
    print("PYTORCH DATASET TEST")
    print("=" * 50)

    print("Number of images:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape :", mask.shape)

    print("Image dtype:", image.dtype)
    print("Mask dtype :", mask.dtype)

    print("Image range:", image.min().item(), "to", image.max().item())

    print("Mask values:", torch.unique(mask).tolist())