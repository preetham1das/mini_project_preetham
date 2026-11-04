import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMAGE_SIZE = 256
TARGET_SIZE = (IMAGE_SIZE, IMAGE_SIZE)
SEED = 42


class ForestDataset(Dataset):
    """
    PyTorch Dataset for Forest Cover Segmentation.
    
    Semantic Mapping:
    - Raw Mask 1 -> Non-Forest / Background (Class 0)
    - Raw Mask 2 -> Forest Cover (Class 1)
    """

    def __init__(self, image_dir, mask_dir=None, augment=False, target_size=TARGET_SIZE):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.augment = augment
        self.target_size = target_size

        self.images = sorted(self.image_dir.glob("*.tiff")) + sorted(self.image_dir.glob("*.tif"))

        if len(self.images) == 0:
            raise RuntimeError(f"No TIFF images found in {self.image_dir}")

        if self.mask_dir:
            for image_path in self.images:
                mask_path = self.mask_dir / (image_path.stem + ".png")
                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Missing mask for image {image_path.name}: expected {mask_path}"
                    )

                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    if image.width == 0 or image.height == 0 or mask.width == 0 or mask.height == 0:
                        raise ValueError(
                            f"Image/mask dimensions are invalid for {image_path.name}: "
                            f"image={image.size}, mask={mask.size}"
                        )
                    if image.size != mask.size:
                        raise ValueError(
                            f"Image/mask dimensions must match exactly for {image_path.name}: "
                            f"image={image.size}, mask={mask.size}"
                        )

    def __len__(self):
        return len(self.images)

    def _apply_augmentation(self, image, mask):
        # Random Horizontal Flip
        if random.random() > 0.5:
            image = np.fliplr(image)
            if mask is not None:
                mask = np.fliplr(mask)

        # Random Vertical Flip
        if random.random() > 0.5:
            image = np.flipud(image)
            if mask is not None:
                mask = np.flipud(mask)

        # Random 90-degree Rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k, axes=(0, 1))
            if mask is not None:
                mask = np.rot90(mask, k, axes=(0, 1))

        return image.copy(), mask.copy() if mask is not None else None

    def __getitem__(self, index):
        image_path = self.images[index]

        # Resize images and masks to the same configurable training resolution.
        pil_image = Image.open(image_path).convert("RGB")
        if pil_image.size != self.target_size:
            pil_image = pil_image.resize(self.target_size, Image.Resampling.BILINEAR)
        
        image = np.array(pil_image, dtype=np.float32)

        # Load the validated mask if this is a labeled split.
        mask = None
        if self.mask_dir:
            mask_path = self.mask_dir / (image_path.stem + ".png")
            pil_mask = Image.open(mask_path)
            if pil_mask.size != self.target_size:
                pil_mask = pil_mask.resize(self.target_size, Image.Resampling.NEAREST)
            mask = np.array(pil_mask)
            mask_values = np.unique(mask)
            if not np.isin(mask_values, (1, 2)).all():
                raise ValueError(
                    f"Unexpected mask classes for {mask_path.name}: {mask_values.tolist()}; "
                    "expected raw classes 1 (non-forest) and 2 (forest)."
                )

        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Normalize image from [0, 255] to [0.0, 1.0]
        image = image / 255.0

        # Convert image (H, W, C) -> (C, H, W) PyTorch layout
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        if mask is not None:
            # Map raw pixel values: 2 -> Forest (1), 1 -> Non-Forest (0)
            mask_binary = (mask == 2).astype(np.int64)
            mask_tensor = torch.tensor(mask_binary, dtype=torch.int64)
            return image_tensor, mask_tensor
        else:
            return image_tensor, image_path.name


def get_dataloaders(
    dataset_root="data/Amazon Forest Dataset",
    batch_size=2,
    num_workers=0,
    image_size=IMAGE_SIZE
):
    """
    Creates and returns DataLoaders for Training, Validation, and Test splits.
    Default batch_size is set to 2 for CPU training efficiency.
    """
    root = Path(dataset_root)

    train_dataset = ForestDataset(
        image_dir=root / "Training" / "images",
        mask_dir=root / "Training" / "masks",
        augment=True,
        target_size=(image_size, image_size)
    )

    val_dataset = ForestDataset(
        image_dir=root / "Validation" / "images",
        mask_dir=root / "Validation" / "masks",
        augment=False,
        target_size=(image_size, image_size)
    )

    test_dataset = ForestDataset(
        image_dir=root / "Test",
        mask_dir=None,
        augment=False,
        target_size=(image_size, image_size)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=2)

    print("=" * 50)
    print("PYTORCH DATASET & DATALOADER TEST")
    print("=" * 50)

    print(f"Train batches      : {len(train_loader)} (Batch size = {train_loader.batch_size})")
    print(f"Validation batches : {len(val_loader)}")
    print(f"Test samples       : {len(test_loader.dataset)}")

    images, masks = next(iter(train_loader))

    print("\nSample Batch Information:")
    print(f"Images batch shape : {images.shape}")
    print(f"Masks batch shape  : {masks.shape}")
    print(f"Image dtype        : {images.dtype}")
    print(f"Mask dtype         : {masks.dtype}")
    print(f"Image value range  : {images.min().item():.4f} to {images.max().item():.4f}")
    print(f"Unique mask classes: {torch.unique(masks).tolist()}")
