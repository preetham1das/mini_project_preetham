import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMAGE_SIZE = 256
TARGET_SIZE = (IMAGE_SIZE, IMAGE_SIZE)
SEED = 42

class TemporalForestDataset(Dataset):
    """
    PyTorch Dataset for Temporal Deforestation Detection.
    Loads matching T1 (A), T2 (B), and ground-truth label pairs.
    """

    def __init__(self, split_dir, augment=False, target_size=TARGET_SIZE):
        self.split_dir = Path(split_dir)
        self.augment = augment
        self.target_size = target_size

        self.a_dir = self.split_dir / "A"
        self.b_dir = self.split_dir / "B"
        self.label_dir = self.split_dir / "label"

        if not self.a_dir.exists() or not self.b_dir.exists() or not self.label_dir.exists():
            raise FileNotFoundError(f"Missing required subdirectories (A, B, label) in {split_dir}")

        self.image_names = sorted([f.name for f in self.a_dir.glob("*.png")])

        if len(self.image_names) == 0:
            raise RuntimeError(f"No PNG images found in {self.a_dir}")

        for name in self.image_names:
            b_path = self.b_dir / name
            label_path = self.label_dir / name
            if not b_path.exists():
                raise FileNotFoundError(f"Missing matching B image for {name}")
            if not label_path.exists():
                raise FileNotFoundError(f"Missing matching label for {name}")

    def __len__(self):
        return len(self.image_names)

    def _apply_augmentation(self, img_a, img_b, mask):
        # Random Horizontal Flip
        if random.random() > 0.5:
            img_a = np.fliplr(img_a)
            img_b = np.fliplr(img_b)
            mask = np.fliplr(mask)

        # Random Vertical Flip
        if random.random() > 0.5:
            img_a = np.flipud(img_a)
            img_b = np.flipud(img_b)
            mask = np.flipud(mask)

        # Random 90-degree Rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            img_a = np.rot90(img_a, k, axes=(0, 1))
            img_b = np.rot90(img_b, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))

        return img_a.copy(), img_b.copy(), mask.copy()

    def __getitem__(self, index):
        name = self.image_names[index]
        a_path = self.a_dir / name
        b_path = self.b_dir / name
        label_path = self.label_dir / name

        # Load and resize images
        pil_a = Image.open(a_path).convert("RGB")
        if pil_a.size != self.target_size:
            pil_a = pil_a.resize(self.target_size, Image.Resampling.BILINEAR)
        img_a = np.array(pil_a, dtype=np.float32)

        pil_b = Image.open(b_path).convert("RGB")
        if pil_b.size != self.target_size:
            pil_b = pil_b.resize(self.target_size, Image.Resampling.BILINEAR)
        img_b = np.array(pil_b, dtype=np.float32)

        # Load and resize label
        pil_label = Image.open(label_path).convert("L")
        if pil_label.size != self.target_size:
            pil_label = pil_label.resize(self.target_size, Image.Resampling.NEAREST)
        mask = np.array(pil_label)

        # Binary conversion (Deforestation > 0)
        mask = (mask > 0).astype(np.float32)

        if self.augment:
            img_a, img_b, mask = self._apply_augmentation(img_a, img_b, mask)

        # Normalize images [0, 255] -> [0.0, 1.0]
        img_a = img_a / 255.0
        img_b = img_b / 255.0

        # Channel layout (H, W, C) -> (C, H, W)
        img_a_tensor = torch.tensor(img_a, dtype=torch.float32).permute(2, 0, 1)
        img_b_tensor = torch.tensor(img_b, dtype=torch.float32).permute(2, 0, 1)
        
        # Concatenate T1 and T2 along channel dim -> (6, H, W)
        img_concat = torch.cat([img_a_tensor, img_b_tensor], dim=0)
        
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # (1, H, W)

        return img_concat, mask_tensor, name


def get_temporal_dataloaders(
    dataset_root="data/Forest-Change/extracted/Forest-Change-dataset/images",
    batch_size=4,
    num_workers=0,
    image_size=IMAGE_SIZE
):
    root = Path(dataset_root)

    train_dataset = TemporalForestDataset(
        split_dir=root / "train",
        augment=True,
        target_size=(image_size, image_size)
    )

    val_dataset = TemporalForestDataset(
        split_dir=root / "val",
        augment=False,
        target_size=(image_size, image_size)
    )

    test_dataset = TemporalForestDataset(
        split_dir=root / "test",
        augment=False,
        target_size=(image_size, image_size)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_temporal_dataloaders(batch_size=2)
    print("=" * 50)
    print("TEMPORAL DATASET & DATALOADER TEST")
    print("=" * 50)
    print(f"Train pairs      : {len(train_loader.dataset)}")
    print(f"Validation pairs : {len(val_loader.dataset)}")
    print(f"Test pairs       : {len(test_loader.dataset)}")

    images, masks, names = next(iter(train_loader))
    print("\nSample Batch Information:")
    print(f"Images batch shape : {images.shape} (Expected: B, 6, H, W)")
    print(f"Masks batch shape  : {masks.shape} (Expected: B, 1, H, W)")
    print(f"Image dtype        : {images.dtype}")
    print(f"Mask dtype         : {masks.dtype}")
    print(f"Image value range  : {images.min().item():.4f} to {images.max().item():.4f}")
    print(f"Unique mask classes: {torch.unique(masks).tolist()}")
