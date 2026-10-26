"""
test_sen2naip.py
================
Smoke test for the SEN2NAIP Sentinel-2 super-resolution dataset loader.
Verifies:
  1. Train, validation, and test split loading.
  2. LR (120x120) and HR (480x480) spatial alignment (4x scaling).
  3. RGB (3-channel) and RGBN (4-channel) band extraction.
  4. Value ranges: [0.0, 1.0] with no NaN/Inf.
  5. Compatibility forward pass with the existing SwinIR architecture.
"""

import sys
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)
import unittest
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "swinir"))

from src.sen2naip_dataset import SEN2NAIPDataset, get_sen2naip_dataloaders
from src.swinir.network_swinir import SwinIR

DEVICE = torch.device("cpu")


class TestSEN2NAIPDataset(unittest.TestCase):
    def test_split_counts_and_shapes(self):
        train_loader, val_loader, test_loader = get_sen2naip_dataloaders(
            root_dir=PROJECT_ROOT / "data" / "SEN2NAIP",
            batch_size=2,
            rgb_only=True,
            lr_crop_size=(120, 120),
        )

        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(val_loader.dataset), 0)
        self.assertGreater(len(test_loader.dataset), 0)

        total = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        self.assertEqual(total, 30)

        # Inspect one batch from train
        batch = next(iter(train_loader))
        lr, hr = batch["lr"], batch["hr"]

        self.assertEqual(lr.shape, (2, 3, 120, 120))
        self.assertEqual(hr.shape, (2, 3, 480, 480))
        self.assertEqual(lr.dtype, torch.float32)
        self.assertEqual(hr.dtype, torch.float32)

        # Check normalization boundaries
        self.assertTrue(torch.all(lr >= 0.0) and torch.all(lr <= 1.0))
        self.assertTrue(torch.all(hr >= 0.0) and torch.all(hr <= 1.0))
        self.assertFalse(torch.isnan(lr).any())
        self.assertFalse(torch.isnan(hr).any())
        self.assertFalse(torch.isinf(lr).any())
        self.assertFalse(torch.isinf(hr).any())

    def test_four_channel_nir_mode(self):
        ds_4ch = SEN2NAIPDataset(
            root_dir=PROJECT_ROOT / "data" / "SEN2NAIP",
            split="all",
            rgb_only=False,
            lr_crop_size=(120, 120),
        )
        sample = ds_4ch[0]
        self.assertEqual(sample["lr"].shape, (4, 120, 120))
        self.assertEqual(sample["hr"].shape, (4, 480, 480))
        # 4th channel is NIR
        self.assertGreater(sample["lr"][3].mean().item(), 0.0)
        self.assertGreater(sample["hr"][3].mean().item(), 0.0)

    def test_swinir_forward_compatibility(self):
        """Verify that a batch from SEN2NAIP can pass through SwinIR network."""
        ds = SEN2NAIPDataset(
            root_dir=PROJECT_ROOT / "data" / "SEN2NAIP",
            split="val",
            rgb_only=True,
            lr_crop_size=(64, 64),  # Fast forward pass for test
        )
        sample = ds[0]
        lr_tensor = sample["lr"].unsqueeze(0)  # (1, 3, 64, 64)

        # Initialize lightweight SwinIR matching our config
        model = SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[2, 2],
            embed_dim=60,
            num_heads=[6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )
        model.eval()

        with torch.no_grad():
            out = model(lr_tensor)

        self.assertEqual(out.shape, (1, 3, 256, 256))
        self.assertFalse(torch.isnan(out).any())

    def test_full_production_swinir_120_to_480(self):
        """Explicitly verify 120x120 LR -> 480x480 SR matching HR target shape using production checkpoint."""
        from swinir.swinir_inference import _load_model

        ds = SEN2NAIPDataset(
            root_dir=PROJECT_ROOT / "data" / "SEN2NAIP",
            split="all",
            rgb_only=True,
            lr_crop_size=(120, 120),
            random_crop=False,
        )
        sample = ds[0]
        lr_tensor = sample["lr"].unsqueeze(0).to(DEVICE)  # [1, 3, 120, 120]
        hr_tensor = sample["hr"].unsqueeze(0).to(DEVICE)  # [1, 3, 480, 480]

        self.assertEqual(lr_tensor.shape, (1, 3, 120, 120))
        self.assertEqual(hr_tensor.shape, (1, 3, 480, 480))

        # Check input range [0, 1]
        self.assertTrue(lr_tensor.min() >= 0.0 and lr_tensor.max() <= 1.0)
        self.assertTrue(hr_tensor.min() >= 0.0 and hr_tensor.max() <= 1.0)

        # Load production model
        model, err = _load_model(DEVICE)
        self.assertIsNone(err)
        model.eval()

        with torch.no_grad():
            sr_out = model(lr_tensor)

        # Verify exact shape alignment
        self.assertEqual(sr_out.shape, (1, 3, 480, 480))
        self.assertEqual(sr_out.shape, hr_tensor.shape)

        # Verify numerical sanity
        self.assertFalse(torch.isnan(sr_out).any())
        self.assertFalse(torch.isinf(sr_out).any())
        self.assertTrue(sr_out.min() >= 0.0)


if __name__ == "__main__":
    unittest.main()

