"""Unit and integration tests for Kaggle Amazon Sentinel-1 + Sentinel-2 pipeline."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds
import torch

from experiments.kaggle_amazon_deforestation.dataset import (
    resolve_sentinel1_bands,
    resolve_sentinel2_bands,
    align_sentinel_pair,
)
from experiments.kaggle_amazon_deforestation.model import LateFusionSegNet


class KaggleAmazonBandResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.td = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_raster(self, filename, count, dtype="float32", descriptions=None, colorinterp=None, shape=(32, 32)):
        path = self.td / filename
        profile = {
            "driver": "GTiff", "width": shape[1], "height": shape[0],
            "count": count, "dtype": dtype, "crs": "EPSG:4326",
            "transform": from_bounds(-60.0, -10.0, -59.0, -9.0, shape[1], shape[0])
        }
        with rasterio.open(path, "w", **profile) as dst:
            if dtype == "uint8":
                data = np.random.randint(0, 255, (count, *shape), dtype="uint8")
            else:
                data = np.random.randn(count, *shape).astype(dtype)
            dst.write(data)
            if descriptions:
                for b, desc in enumerate(descriptions, 1):
                    dst.set_band_description(b, desc)
            if colorinterp:
                dst.colorinterp = colorinterp
        return path

    def test_sentinel1_accepts_valid_two_bands(self):
        p = self._create_raster("s1_valid.tif", count=2, descriptions=["VV", "VH"])
        with rasterio.open(p) as src:
            data, valid, info = resolve_sentinel1_bands(src)
            self.assertEqual(data.shape, (2, 32, 32))
            self.assertEqual(valid.shape, (32, 32))
            self.assertIn("Sentinel-1", info)

    def test_sentinel1_accepts_three_bands_with_verified_auxiliary(self):
        p = self._create_raster("s1_3band.tif", count=3, descriptions=["VV", "VH", "incidence_angle"])
        with rasterio.open(p) as src:
            data, valid, info = resolve_sentinel1_bands(src)
            self.assertEqual(data.shape, (2, 32, 32))
            self.assertIn("incidence_angle", info)

    def test_sentinel1_rejects_optical_rgb_image(self):
        p = self._create_raster("s1_rgb.tif", count=3, dtype="uint8",
                                colorinterp=[ColorInterp.red, ColorInterp.green, ColorInterp.blue])
        with rasterio.open(p) as src:
            with self.assertRaisesRegex(ValueError, "RGB optical image"):
                resolve_sentinel1_bands(src)

    def test_sentinel1_rejects_actual_stage1_image(self):
        actual_0_tiff = Path("data/Amazon Forest Dataset/Test/0.tiff")
        if actual_0_tiff.exists():
            with rasterio.open(actual_0_tiff) as src:
                with self.assertRaisesRegex(ValueError, "RGB optical image"):
                    resolve_sentinel1_bands(src)

    def test_sentinel1_rejects_single_band(self):
        p = self._create_raster("s1_1band.tif", count=1)
        with rasterio.open(p) as src:
            with self.assertRaisesRegex(ValueError, "only 1 band"):
                resolve_sentinel1_bands(src)

    def test_sentinel1_rejects_unlabelled_three_bands(self):
        p = self._create_raster("s1_unknown3.tif", count=3)
        with rasterio.open(p) as src:
            with self.assertRaisesRegex(ValueError, "could not be unambiguously verified"):
                resolve_sentinel1_bands(src)

    def test_sentinel2_accepts_valid_four_spectral_bands(self):
        p = self._create_raster("s2_valid.tif", count=4, descriptions=["B2", "B3", "B4", "B8"])
        with rasterio.open(p) as src:
            data, valid, info = resolve_sentinel2_bands(src)
            self.assertEqual(data.shape, (4, 32, 32))
            self.assertIn("B2", info)

    def test_sentinel2_rejects_three_band_rgb(self):
        p = self._create_raster("s2_rgb.tif", count=3)
        with rasterio.open(p) as src:
            with self.assertRaisesRegex(ValueError, "only 3 bands"):
                resolve_sentinel2_bands(src)

    def test_sentinel2_rejects_actual_stage1_4_tiff(self):
        actual_4_tiff = Path("data/Amazon Forest Dataset/Test/4.tiff")
        if actual_4_tiff.exists():
            with rasterio.open(actual_4_tiff) as src:
                with self.assertRaisesRegex(ValueError, "only 3 bands"):
                    resolve_sentinel2_bands(src)

    def test_sentinel2_rejects_rgba_image(self):
        p = self._create_raster("s2_rgba.tif", count=4, dtype="uint8",
                                colorinterp=[ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha])
        with rasterio.open(p) as src:
            with self.assertRaisesRegex(ValueError, "RGBA"):
                resolve_sentinel2_bands(src)


class KaggleAmazonAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.td = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_align_identical_rasters(self):
        p1 = self.td / "s1.tif"
        p2 = self.td / "s2.tif"
        t = from_bounds(-60.0, -10.0, -59.0, -9.0, 32, 32)
        with rasterio.open(p1, "w", driver="GTiff", width=32, height=32, count=2, dtype="float32", crs="EPSG:4326", transform=t) as d:
            d.write(np.ones((2, 32, 32), dtype=np.float32))
        with rasterio.open(p2, "w", driver="GTiff", width=32, height=32, count=4, dtype="float32", crs="EPSG:4326", transform=t) as d:
            d.write(np.ones((4, 32, 32), dtype=np.float32))

        with rasterio.open(p1) as s1_src, rasterio.open(p2) as s2_src:
            s1_d = s1_src.read()
            s2_d = s2_src.read()
            v1 = np.ones((32, 32), dtype=bool)
            v2 = np.ones((32, 32), dtype=bool)
            s1, s2, valid, was_resampled, note = align_sentinel_pair(s1_src, s2_src, s1_d, s2_d, v1, v2)
            self.assertFalse(was_resampled)
            self.assertEqual(s1.shape, (2, 32, 32))
            self.assertEqual(s2.shape, (4, 32, 32))
            self.assertTrue(valid.all())

    def test_align_rejects_non_overlapping_rasters(self):
        p1 = self.td / "s1.tif"
        p2 = self.td / "s2.tif"
        t1 = from_bounds(-60.0, -10.0, -59.0, -9.0, 32, 32)
        t2 = from_bounds(10.0, 45.0, 11.0, 46.0, 32, 32)
        with rasterio.open(p1, "w", driver="GTiff", width=32, height=32, count=2, dtype="float32", crs="EPSG:4326", transform=t1) as d:
            d.write(np.ones((2, 32, 32), dtype=np.float32))
        with rasterio.open(p2, "w", driver="GTiff", width=32, height=32, count=4, dtype="float32", crs="EPSG:4326", transform=t2) as d:
            d.write(np.ones((4, 32, 32), dtype=np.float32))

        with rasterio.open(p1) as s1_src, rasterio.open(p2) as s2_src:
            with self.assertRaisesRegex(ValueError, "do not cover the same geographic area"):
                align_sentinel_pair(s1_src, s2_src, s1_src.read(), s2_src.read(), np.ones((32, 32), bool), np.ones((32, 32), bool))

    def test_align_resamples_differing_dimensions(self):
        p1 = self.td / "s1.tif"
        p2 = self.td / "s2.tif"
        t1 = from_bounds(-60.0, -10.0, -59.0, -9.0, 32, 32)
        t2 = from_bounds(-60.0, -10.0, -59.0, -9.0, 64, 64)
        with rasterio.open(p1, "w", driver="GTiff", width=32, height=32, count=2, dtype="float32", crs="EPSG:4326", transform=t1) as d:
            d.write(np.ones((2, 32, 32), dtype=np.float32))
        with rasterio.open(p2, "w", driver="GTiff", width=64, height=64, count=4, dtype="float32", crs="EPSG:4326", transform=t2) as d:
            d.write(np.full((4, 64, 64), 3.0, dtype=np.float32))

        with rasterio.open(p1) as s1_src, rasterio.open(p2) as s2_src:
            s1, s2, valid, was_resampled, note = align_sentinel_pair(
                s1_src, s2_src, s1_src.read(), s2_src.read(), np.ones((32, 32), bool), np.ones((64, 64), bool)
            )
            self.assertTrue(was_resampled)
            self.assertEqual(s2.shape, (4, 32, 32))
            self.assertAlmostEqual(float(s2.mean()), 3.0, places=3)


class KaggleAmazonModelCheckpointTests(unittest.TestCase):
    def test_checkpoint_and_normalization_integrity(self):
        ckpt_path = Path("checkpoints/kaggle_amazon/late_fusion_segnet_v2_corrected_best.pt")
        norm_path = Path("checkpoints/kaggle_amazon/normalization.json")

        self.assertTrue(ckpt_path.is_file(), "Checkpoint file must exist.")
        self.assertTrue(norm_path.is_file(), "Normalization JSON must exist.")

        norm = json.loads(norm_path.read_text(encoding="utf-8"))
        self.assertIn("s1", norm)
        self.assertIn("s2", norm)
        self.assertEqual(len(norm["s1"]["low"]), 2)
        self.assertEqual(len(norm["s1"]["high"]), 2)
        self.assertEqual(len(norm["s2"]["low"]), 4)
        self.assertEqual(len(norm["s2"]["high"]), 4)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model = LateFusionSegNet(s1_channels=2, s2_channels=4, base_channels=32)
        model.load_state_dict(sd)
        model.eval()

        # Run dummy 256x256 patch
        x1 = torch.zeros(1, 2, 256, 256)
        x2 = torch.zeros(1, 4, 256, 256)
        with torch.no_grad():
            out = model(x1, x2)
        self.assertEqual(tuple(out.shape), (1, 1, 256, 256))


if __name__ == "__main__":
    unittest.main()
