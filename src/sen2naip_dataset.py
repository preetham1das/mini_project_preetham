"""
src/sen2naip_dataset.py
=======================
PyTorch Dataset and DataLoader for Sentinel-2 / NAIP super-resolution fine-tuning.

Dataset Properties (SEN2NAIP Cross-Sensor):
  - LR: Sentinel-2 L2A Surface Reflectance (10 m GSD, 121 x 121 x 4)
        Bands: [B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)]
        Native dtype: int32 (scaled x10000)
  - HR: NAIP Aerial Imagery (2.5 m GSD, 484 x 484 x 4)
        Bands: [Red, Green, Blue, NIR]
        Native dtype: uint8 [0, 255]
  - Spatial Scale Factor: exactly 4.0x (121 x 4 = 484)

Band Selection Policy:
  - Default: `rgb_only=True` extracts the first 3 channels (Red, Green, Blue).
    Rationale: Our existing SwinIR architecture (SwinIR-M classical SR) is parameterized
    for 3-channel RGB input/output, and the GIS web dashboard displays true-color RGB.
  - Setting `rgb_only=False` returns all 4 channels [R, G, B, NIR].
    The NIR band is preserved on disk in the GeoTIFF files and is never silently deleted.

Spatial Alignment & SwinIR Window Divisibility:
  - SwinIR uses window-based self-attention with window_size=8.
  - The native LR size 121x121 is not divisible by 8 (121 % 8 = 1).
  - By default, this loader applies a crop of (120, 120) for LR and (480, 480) for HR,
    which satisfies divisibility for both LR (120 % 8 = 0) and HR (480 % 32 = 0)
    while preserving the exact 4x spatial scale.
"""

import json
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import logging
    import tifffile
    logging.getLogger("tifffile").setLevel(logging.ERROR)
except ImportError:
    raise ImportError("tifffile is required for SEN2NAIP GeoTIFF loading. Install via: pip install tifffile")

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "SEN2NAIP"


class SEN2NAIPDataset(Dataset):
    """
    PyTorch Dataset for paired Sentinel-2 (LR) and NAIP (HR) super-resolution.

    Args:
        root_dir:     Path to data/SEN2NAIP directory.
        split:        One of 'train', 'val', 'test', or 'all'.
        splits_file:  Optional path to splits.json. Defaults to root_dir / 'splits.json'.
        rgb_only:     If True, returns 3-channel RGB [R, G, B].
                      If False, returns 4-channel [R, G, B, NIR].
        lr_crop_size: LR crop dimensions (H, W). Default (120, 120) to ensure divisibility by 8.
                      Set to None to keep native 121x121.
        scale:        Super-resolution scale factor (default: 4).
        random_crop:  If True, selects random crop location (recommended for training).
                      If False, selects center crop (deterministic for val/test).
        augment:      If True, applies synchronized spatial augmentations (flips, rotations).
    """

    def __init__(
        self,
        root_dir: Path = DEFAULT_DATASET_DIR,
        split: str = "train",
        splits_file: Optional[Path] = None,
        rgb_only: bool = True,
        lr_crop_size: Optional[Tuple[int, int]] = (120, 120),
        scale: int = 4,
        random_crop: Optional[bool] = None,
        augment: Optional[bool] = None,
    ):
        self.root_dir = Path(root_dir)
        self.cross_dir = self.root_dir / "cross-sensor"
        self.split = split
        self.rgb_only = rgb_only
        self.lr_crop_size = lr_crop_size
        self.scale = scale

        # Defaults for cropping and augmentation based on split
        self.random_crop = (split == "train") if random_crop is None else random_crop
        self.augment = (split == "train") if augment is None else augment

        if not self.cross_dir.exists():
            raise FileNotFoundError(f"SEN2NAIP cross-sensor directory not found at {self.cross_dir}")

        # Resolve splits
        splits_path = Path(splits_file) if splits_file else self.root_dir / "splits.json"
        if splits_path.exists():
            with open(splits_path, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            if split == "all":
                self.roi_ids = sorted(
                    splits_data.get("train", []) + splits_data.get("val", []) + splits_data.get("test", [])
                )
            else:
                self.roi_ids = sorted(splits_data.get(split, []))
        else:
            # Fallback: scan directory directly
            all_rois = sorted([d.name for d in self.cross_dir.iterdir() if d.is_dir() and d.name.startswith("ROI_")])
            if split == "all":
                self.roi_ids = all_rois
            else:
                # 80/10/10 fallback
                n_train = int(len(all_rois) * 0.8)
                n_val = int(len(all_rois) * 0.1)
                if split == "train":
                    self.roi_ids = all_rois[:n_train]
                elif split == "val":
                    self.roi_ids = all_rois[n_train : n_train + n_val]
                else:
                    self.roi_ids = all_rois[n_train + n_val :]

        # Filter to only existing pairs on disk
        self.valid_pairs: List[Tuple[Path, Path, Path]] = []
        for roi_id in self.roi_ids:
            roi_d = self.cross_dir / roi_id
            lr_p = roi_d / "lr.tif"
            hr_p = roi_d / "hr.tif"
            meta_p = roi_d / "metadata.json"
            if lr_p.exists() and hr_p.exists():
                self.valid_pairs.append((lr_p, hr_p, meta_p))

        if len(self.valid_pairs) == 0:
            raise RuntimeError(f"No valid pairs found for split '{split}' in {self.cross_dir}")

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def _apply_augmentation(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronously augment LR and HR pairs."""
        # Random horizontal flip
        if random.random() > 0.5:
            lr = np.fliplr(lr)
            hr = np.fliplr(hr)

        # Random vertical flip
        if random.random() > 0.5:
            lr = np.flipud(lr)
            hr = np.flipud(hr)

        # Random 90-degree rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            lr = np.rot90(lr, k, axes=(0, 1))
            hr = np.rot90(hr, k, axes=(0, 1))

        return lr.copy(), hr.copy()

    def _crop_pair(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crops LR to lr_crop_size and HR to (lr_crop_size * scale).
        Guarantees spatial alignment between LR and HR patches.
        """
        if self.lr_crop_size is None:
            return lr, hr

        lr_h, lr_w = lr.shape[:2]
        crop_h, crop_w = self.lr_crop_size

        if crop_h > lr_h or crop_w > lr_w:
            raise ValueError(f"Crop size {self.lr_crop_size} exceeds LR shape {(lr_h, lr_w)}")

        if self.random_crop:
            top = random.randint(0, lr_h - crop_h)
            left = random.randint(0, lr_w - crop_w)
        else:
            top = (lr_h - crop_h) // 2
            left = (lr_w - crop_w) // 2

        bottom = top + crop_h
        right = left + crop_w

        hr_top = top * self.scale
        hr_left = left * self.scale
        hr_bottom = bottom * self.scale
        hr_right = right * self.scale

        lr_cropped = lr[top:bottom, left:right]
        hr_cropped = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr_cropped, hr_cropped

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lr_path, hr_path, meta_path = self.valid_pairs[idx]

        # Read GeoTIFF arrays
        lr_arr = tifffile.imread(str(lr_path)).astype(np.float32)   # (H, W, 4), int32 surface reflectance
        hr_arr = tifffile.imread(str(hr_path)).astype(np.float32)   # (4H, 4W, 4), uint8 [0, 255]

        # Select bands:
        # Channels 0, 1, 2 = Red, Green, Blue
        # Channel 3 = NIR
        if self.rgb_only:
            lr_arr = lr_arr[:, :, :3]
            hr_arr = hr_arr[:, :, :3]

        # Apply crop (guarantees divisibility for SwinIR windows)
        lr_arr, hr_arr = self._crop_pair(lr_arr, hr_arr)

        # Apply synchronized spatial augmentations
        if self.augment:
            lr_arr, hr_arr = self._apply_augmentation(lr_arr, hr_arr)

        # Normalization:
        # - LR: Sentinel-2 L2A surface reflectance is scaled x10000.
        #       Values are non-negative and rarely exceed 10000 for terrestrial targets.
        #       Normalizing by 10000.0 scales reflectance into [0.0, 1.0].
        lr_norm = np.clip(lr_arr / 10000.0, 0.0, 1.0)

        # - HR: NAIP aerial imagery is standard uint8 [0, 255].
        #       Normalizing by 255.0 scales into [0.0, 1.0].
        hr_norm = np.clip(hr_arr / 255.0, 0.0, 1.0)

        # Layout conversion: (H, W, C) -> (C, H, W) PyTorch tensor
        lr_tensor = torch.from_numpy(lr_norm).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_norm).permute(2, 0, 1).float()

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "roi_id": lr_path.parent.name,
        }


def get_sen2naip_dataloaders(
    root_dir: Path = DEFAULT_DATASET_DIR,
    batch_size: int = 4,
    num_workers: int = 0,
    rgb_only: bool = True,
    lr_crop_size: Tuple[int, int] = (120, 120),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.
    """
    train_ds = SEN2NAIPDataset(
        root_dir=root_dir,
        split="train",
        rgb_only=rgb_only,
        lr_crop_size=lr_crop_size,
        random_crop=True,
        augment=True,
    )
    val_ds = SEN2NAIPDataset(
        root_dir=root_dir,
        split="val",
        rgb_only=rgb_only,
        lr_crop_size=lr_crop_size,
        random_crop=False,
        augment=False,
    )
    test_ds = SEN2NAIPDataset(
        root_dir=root_dir,
        split="test",
        rgb_only=rgb_only,
        lr_crop_size=lr_crop_size,
        random_crop=False,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=" * 60)
    print("SEN2NAIP DATASET LOADER SMOKE TEST")
    print("=" * 60)

    try:
        ds = SEN2NAIPDataset(split="all", rgb_only=True)
        print(f"Loaded 'all' split: {len(ds)} samples found.")
        sample = ds[0]
        print(f"Sample 0 ({sample['roi_id']}):")
        print(f"  LR tensor: shape={sample['lr'].shape}, dtype={sample['lr'].dtype}, min={sample['lr'].min():.4f}, max={sample['lr'].max():.4f}")
        print(f"  HR tensor: shape={sample['hr'].shape}, dtype={sample['hr'].dtype}, min={sample['hr'].min():.4f}, max={sample['hr'].max():.4f}")
        assert sample['hr'].shape[1] == sample['lr'].shape[1] * 4
        assert sample['hr'].shape[2] == sample['lr'].shape[2] * 4
        print("[SUCCESS] SEN2NAIP dataset loader operational.")
    except Exception as e:
        print(f"[NOTE] Dataset not yet populated or error: {e}")
