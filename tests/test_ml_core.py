import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app import estimate_pixel_area_hectares, parse_geographic_bbox, parse_probability_threshold
from src.dataset import ForestDataset
from src.metrics import CombinedBCEDiceLoss, calculate_confusion_counts, metrics_from_counts
from src.model import UNet
from src.temporal_dataset import TemporalForestDataset
from src.temporal_model import DiceLoss, FocalLoss, TemporalUNet


class ModelShapeTests(unittest.TestCase):
    def test_unet_preserves_odd_input_dimensions_and_backpropagates(self):
        model = UNet(in_channels=3, out_channels=1, init_features=2)
        images = torch.randn(2, 3, 29, 45)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 1, 29, 45))
        loss = CombinedBCEDiceLoss()(logits, torch.zeros(2, 29, 45))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_temporal_unet_supports_six_and_nine_channels_with_odd_dimensions(self):
        for channels in (6, 9):
            with self.subTest(channels=channels):
                model = TemporalUNet(in_channels=channels, out_channels=1, init_features=2)
                logits = model(torch.randn(2, channels, 31, 43))
                self.assertEqual(tuple(logits.shape), (2, 1, 31, 43))

    def test_models_reject_wrong_input_channel_count(self):
        with self.assertRaisesRegex(ValueError, "Expected input"):
            UNet(in_channels=3, init_features=2)(torch.randn(1, 1, 32, 32))
        with self.assertRaisesRegex(ValueError, "Expected input"):
            TemporalUNet(in_channels=6, init_features=2)(torch.randn(1, 9, 32, 32))


class MetricTests(unittest.TestCase):
    def test_confusion_counts_and_zero_division_convention(self):
        # Model predicts no positives although the target contains positives.
        probs = torch.zeros(1, 1, 2, 2)
        targets = torch.tensor([[[1, 0], [0, 0]]])
        counts = calculate_confusion_counts(probs, targets, is_logits=False)
        metrics = metrics_from_counts(counts)
        self.assertEqual(counts, {"tp": 0, "fp": 0, "fn": 1, "tn": 3})
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)

    def test_empty_target_and_empty_prediction_are_not_reported_as_perfect_precision(self):
        metrics = metrics_from_counts({"tp": 0, "fp": 0, "fn": 0, "tn": 8})
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_metric_inputs_are_validated(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            calculate_confusion_counts(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), threshold=1.1)
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            calculate_confusion_counts(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 3, 2))
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            calculate_confusion_counts(torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), 255.0))

    def test_dice_losses_average_per_sample_and_backpropagate(self):
        logits = torch.zeros(2, 1, 3, 5, requires_grad=True)
        targets = torch.zeros(2, 3, 5)
        targets[0, 1, 1] = 1
        for criterion in (DiceLoss(),):
            loss = criterion(logits, targets)
            loss.backward(retain_graph=True)
            self.assertTrue(torch.isfinite(loss))
        focal = FocalLoss()(logits, targets)
        self.assertTrue(torch.isfinite(focal))


class ApiValidationTests(unittest.TestCase):
    def test_probability_threshold_validation(self):
        self.assertEqual(parse_probability_threshold("0.35"), 0.35)
        for value in ("nan", "inf", "-0.1", "1.01", "not-a-number"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_probability_threshold(value)

    def test_geographic_bbox_and_area_estimate_validation(self):
        bbox = parse_geographic_bbox([0, 0, 1, 1])
        self.assertEqual(bbox, [0.0, 0.0, 1.0, 1.0])
        area_per_pixel = estimate_pixel_area_hectares(bbox, 256 * 256)
        self.assertGreater(area_per_pixel, 0)
        self.assertIsNone(estimate_pixel_area_hectares(None, 256 * 256))
        for invalid_bbox in ([0, 1, 0, 2], [181, 0, 182, 1], [0, 0, float("nan"), 1]):
            with self.subTest(bbox=invalid_bbox), self.assertRaises(ValueError):
                parse_geographic_bbox(invalid_bbox)


class DatasetPairingTests(unittest.TestCase):
    def test_temporal_dataset_builds_explicit_nine_channel_difference(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp_dir:
            split = Path(temp_dir)
            for folder in ("A", "B", "label"):
                (split / folder).mkdir()
            t1 = np.zeros((16, 24, 3), dtype=np.uint8)
            t2 = np.full((16, 24, 3), 20, dtype=np.uint8)
            Image.fromarray(t1).save(split / "A" / "pair.png")
            Image.fromarray(t2).save(split / "B" / "pair.png")
            Image.fromarray(np.zeros((16, 24), dtype=np.uint8)).save(split / "label" / "pair.png")

            image, mask, name = TemporalForestDataset(
                split, target_size=(32, 32), diff_channels=True
            )[0]

            self.assertEqual(name, "pair.png")
            self.assertEqual(tuple(image.shape), (9, 32, 32))
            self.assertEqual(tuple(mask.shape), (1, 32, 32))
            self.assertAlmostEqual(float(image[6].mean()), 20.0 / 255.0, places=5)

    def test_temporal_dataset_rejects_misaligned_triplet(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp_dir:
            split = Path(temp_dir)
            for folder in ("A", "B", "label"):
                (split / folder).mkdir()
            Image.new("RGB", (16, 16)).save(split / "A" / "pair.png")
            Image.new("RGB", (20, 16)).save(split / "B" / "pair.png")
            Image.new("L", (16, 16)).save(split / "label" / "pair.png")
            with self.assertRaisesRegex(ValueError, "dimensions must match"):
                TemporalForestDataset(split)[0]

    def test_stage1_dataset_rejects_misaligned_image_and_mask(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp_dir:
            image_dir = Path(temp_dir) / "images"
            mask_dir = Path(temp_dir) / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            Image.new("RGB", (16, 16)).save(image_dir / "tile.tif")
            Image.new("L", (20, 16)).save(mask_dir / "tile.png")
            with self.assertRaisesRegex(ValueError, "must match exactly"):
                ForestDataset(image_dir, mask_dir)

    def test_stage1_dataset_rejects_unknown_mask_classes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp_dir:
            image_dir = Path(temp_dir) / "images"
            mask_dir = Path(temp_dir) / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            Image.new("RGB", (16, 16)).save(image_dir / "tile.tif")
            Image.fromarray(np.full((16, 16), 255, dtype=np.uint8)).save(mask_dir / "tile.png")
            with self.assertRaisesRegex(ValueError, "Unexpected mask classes"):
                ForestDataset(image_dir, mask_dir)[0]


if __name__ == "__main__":
    unittest.main()
