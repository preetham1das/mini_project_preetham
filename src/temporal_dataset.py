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

    Args:
        split_dir:    Path to split directory containing A/, B/, label/
        augment:      Apply random spatial augmentation (train only)
        target_size:  Resize target (W, H)
        diff_channels: If True, append |A-B| as 3 extra channels -> (9, H, W)
    """

    def __init__(self, split_dir, augment=False, target_size=TARGET_SIZE, diff_channels=False):
        self.split_dir = Path(split_dir)
        self.augment = augment
        self.target_size = target_size
        self.diff_channels = diff_channels

        self.a_dir = self.split_dir / "A"
        self.b_dir = self.split_dir / "B"
        self.label_dir = self.split_dir / "label"

        if not self.a_dir.exists() or not self.b_dir.exists() or not self.label_dir.exists():
            raise FileNotFoundError(
                f"Missing required subdirectories (A, B, label) in {split_dir}"
            )

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
        """Apply identical spatial augmentations to A, B, and mask."""
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
        pil_b = Image.open(b_path).convert("RGB")
        pil_label = Image.open(label_path).convert("L")
        if pil_a.size != pil_b.size or pil_a.size != pil_label.size:
            raise ValueError(
                f"T1, T2, and label dimensions must match for {name}: "
                f"T1={pil_a.size}, T2={pil_b.size}, label={pil_label.size}"
            )
        if pil_a.size != self.target_size:
            pil_a = pil_a.resize(self.target_size, Image.Resampling.BILINEAR)
        img_a = np.array(pil_a, dtype=np.float32)
        if pil_b.size != self.target_size:
            pil_b = pil_b.resize(self.target_size, Image.Resampling.BILINEAR)
        img_b = np.array(pil_b, dtype=np.float32)

        # Load and resize label (NEAREST to preserve binary values)
        if pil_label.size != self.target_size:
            pil_label = pil_label.resize(self.target_size, Image.Resampling.NEAREST)
        mask = np.array(pil_label)

        # Binary conversion -- any grayscale value > 0 -> deforestation
        # (Label has multi-value grayscale encoding change intensity)
        mask = (mask > 0).astype(np.float32)

        if self.augment:
            img_a, img_b, mask = self._apply_augmentation(img_a, img_b, mask)

        # Normalize images [0, 255] -> [0.0, 1.0]
        img_a = img_a / 255.0
        img_b = img_b / 255.0

        # Channel layout (H, W, C) -> (C, H, W)
        img_a_tensor = torch.tensor(img_a, dtype=torch.float32).permute(2, 0, 1)
        img_b_tensor = torch.tensor(img_b, dtype=torch.float32).permute(2, 0, 1)

        if self.diff_channels:
            # Absolute difference |A - B|: explicit temporal prior (3 extra channels)
            diff_tensor = torch.abs(img_a_tensor - img_b_tensor)
            # Concatenate T1, T2, |T1-T2| along channel dim -> (9, H, W)
            img_concat = torch.cat([img_a_tensor, img_b_tensor, diff_tensor], dim=0)
        else:
            # Standard 6-channel: concatenate T1 and T2 -> (6, H, W)
            img_concat = torch.cat([img_a_tensor, img_b_tensor], dim=0)

        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return img_concat, mask_tensor, name


def compute_pos_weight(dataset_root, image_size=IMAGE_SIZE):
    """
    Compute exact pos_weight = neg_pixels / pos_pixels from full training set.
    Call once before training. Returns a float.
    """
    root = Path(dataset_root)
    ds = TemporalForestDataset(root / "train", augment=False, target_size=(image_size, image_size))
    total_pos = 0
    total_neg = 0
    for _, mask, _ in ds:
        total_pos += mask.sum().item()
        total_neg += (mask == 0).sum().item()
    weight = total_neg / (total_pos + 1e-8)
    # Clamp to a sensible range for numerical stability
    return float(min(max(weight, 1.0), 50.0))


def get_temporal_dataloaders(
    dataset_root="data/Forest-Change/extracted/Forest-Change-dataset/images",
    batch_size=4,
    num_workers=0,
    image_size=IMAGE_SIZE,
    diff_channels=False,
):
    root = Path(dataset_root)

    train_dataset = TemporalForestDataset(
        split_dir=root / "train",
        augment=True,
        target_size=(image_size, image_size),
        diff_channels=diff_channels,
    )

    val_dataset = TemporalForestDataset(
        split_dir=root / "val",
        augment=False,
        target_size=(image_size, image_size),
        diff_channels=diff_channels,
    )

    test_dataset = TemporalForestDataset(
        split_dir=root / "test",
        augment=False,
        target_size=(image_size, image_size),
        diff_channels=diff_channels,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check
    for diff in [False, True]:
        train_loader, val_loader, test_loader = get_temporal_dataloaders(
            batch_size=2, diff_channels=diff
        )
        images, masks, names = next(iter(train_loader))
        ch = 9 if diff else 6
        print(f"diff_channels={diff}: images={images.shape} (expect B,{ch},H,W), masks={masks.shape}")
        print(f"  Image range: {images.min():.3f} to {images.max():.3f}")
        print(f"  Mask unique: {torch.unique(masks).tolist()}")
